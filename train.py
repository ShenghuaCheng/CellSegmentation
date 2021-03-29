import os
import numpy as np
import argparse
from tqdm import tqdm
# import cv2
# from PIL import Image

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

# Training settings
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--batch_size', type=int, default=64, help='mini-batch size (default: 64)')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--workers', default=4, type=int, help='number of dataloader workers (default: 4)')
parser.add_argument('--test_every', default=1, type=int, help='test on val every (default: 1)')
parser.add_argument('--topk', default=10, type=int,
                    help='top k tiles are assumed to be of the same class as the slide (default: 10, standard MIL)')
parser.add_argument('--interval', type=int, default=20, help='sample interval of patches (default: 20)')
parser.add_argument('--patch_size', type=int, default=32, help='size of each patch (default: 32)')
parser.add_argument('--device', type=str, default='0', help='CUDA device if available (default: \'0\')')
parser.add_argument('--output', type=str, default='.', help='name of output file')
# parser.add_argument('--resume', type=str, default=None, help='continue training from a checkpoint file.pth')
args = parser.parse_args()

if torch.cuda.is_available():
    torch.cuda.manual_seed(1)
    print('\nGPU is available.\n')
else:
    torch.manual_seed(1)

max_acc = 0
resume = False

print('Init Model ...')
model = models.resnet18(pretrained=True)

model.fc = nn.Linear(model.fc.in_features, 2)
# if args.resume:
#     resume = True
#     model.load_state_dict(torch.load(args.resume)['state_dict'])

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
trans = transforms.Compose([transforms.ToTensor(), normalize])

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_VISIBLE_DEVICES'] = args.device
model.to(device)


def train(trainset, valset, batch_size, workers, total_epochs, test_every, model, criterion, optimizer, topk_pos,
          topk_neg, output_path):
    """
    :param trainset:        训练数据集
    :param valset:          验证数据集
    :param batch_size:      Dataloader 打包的小 batch 大小
    :param workers:         Dataloader 使用的进程数
    :param total_epochs:    迭代次数
    :param test_every:      每验证一轮间隔的迭代次数
    :param model:           网络模型
    :param criterion:       损失函数
    :param optimizer:       优化器
    :param topk_pos:        每次在**单个阳性细胞**上选取的 top-k patch 数
    :param topk_neg:        每次在阴性细胞图像上选取的 top-k patch 数
    :param output_path:     保存模型文件的目录
    """

    global max_acc, resume

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=False)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=False)

    # open output file
    fconv = open(os.path.join(output_path, 'convergence.csv'), 'w')
    fconv.write('epoch,metric,value\n')
    fconv.close()
    # 结果保存在output_path/convergence.csv

    print('Start training ...')
    # if resume:
    #     print('Resuming from the checkpoint (epochs: {}).'.format(model['epoch']))

    for epoch in range(1, total_epochs + 1):

        trainset.setmode(1)

        # 获取实例分类概率
        model.eval()
        probs = torch.FloatTensor(len(train_loader.dataset))
        with torch.no_grad():
            # 禁止反向传播
            patch_bar = tqdm(train_loader, total=len(train_loader))
            for i, input in enumerate(patch_bar):
                patch_bar.set_postfix(step="patch forwarding",
                                      epoch="[{}/{}]".format(epoch, total_epochs),
                                      batch="[{}/{}]".format(i + 1, len(train_loader)))
                # softmax 输出 [[a,b],[c,d]] shape = batch_size*2
                output = F.softmax(model(input[0].to(device)), dim=1)
                # detach()[:,1] 取出 softmax 得到的概率，产生：[b, d, ...]
                # input.size(0) 返回 batch 中的实例数量
                probs[i * batch_size:i * batch_size + input[0].size(0)] = output.detach()[:, 1].clone()

        # 找出 top-k
        probs = probs.cpu().numpy()
        groups = np.array(trainset.imageIDX)
        order = np.lexsort((probs, groups))
        dataset_length = trainset.__len__()

        index = np.empty(dataset_length, 'bool')
        for i in range(dataset_length):
            # topk 是实际的 patch 选取数目，如果是阴性样本就直接选取设定的值，如果是阳性样本就使用坐标数目 × 设定的值
            topk = topk_neg if trainset.labels[groups[i]] == 0 else trainset.labels[groups[i]] * topk_pos
            index[i] = groups[i] != groups[(i + topk) % len(groups)]

        # 根据分类，制作迭代使用的数据集
        trainset.make_train_data(list(order[index]))

        trainset.setmode(2)

        # training
        model.train()
        train_loss = 0.
        train_bar = tqdm(train_loader, total=len(train_loader))
        for i, (data, label) in enumerate(train_bar):
            train_bar.set_postfix(step="training",
                                  epoch="[{}/{}]".format(epoch, total_epochs),
                                  batch="[{}/{}]".format(i + 1, len(train_loader)))

            output = model(data.to(device))
            # reset gradients
            optimizer.zero_grad()
            # calculate loss
            loss = criterion(output, label.to(device))
            train_loss += loss.item() * data.size(0)

            # backward pass
            loss.backward()
            # step
            optimizer.step()

        # calculate loss and error for epoch
        train_loss /= len(train_loader.dataset)
        print('Epoch: [{}/{}], Loss: {:.4f}\n'.format(epoch, total_epochs, train_loss))
        fconv = open(os.path.join(output_path, 'convergence.csv'), 'a')
        fconv.write('{},loss,{}\n'.format(epoch, train_loss))
        fconv.close()

        # 验证

        if (epoch + 1) % test_every == 0:

            print('Validating ...')

            valset.setmode(1)
            model.eval()
            val_probs = torch.FloatTensor(len(val_loader.dataset))
            with torch.no_grad():
                # 禁止反向传播
                bar = tqdm(val_loader, total=len(val_loader))
                for i, input in enumerate(bar):
                    bar.set_postfix(epoch="[{}/{}]".format(epoch, total_epochs),
                                    batch="[{}/{}]".format(i + 1, len(val_loader)))
                    # 把训练过的 model 在验证集上做一下
                    val_output = F.softmax(model(input[0].to(device)), dim=1)
                    val_probs[i * batch_size:i * batch_size + input[0].size(0)] \
                        = val_output.detach()[:, 1].clone()

            val_probs = val_probs.cpu().numpy()
            val_groups = np.array(valset.imageIDX)

            max_prob = np.empty(len(valset.labels))  # 模型预测的实例最大概率列表，每张切片取最大概率的 patch
            max_prob[:] = np.nan
            order = np.lexsort((val_probs, val_groups))
            # 排序
            val_groups = val_groups[order]
            val_probs = val_probs[order]
            # 取最大
            val_index = np.empty(len(val_groups), 'bool')
            val_index[-1] = True
            val_index[:-1] = val_groups[1:] != val_groups[:-1]
            max_prob[val_groups[val_index]] = val_probs[val_index]

            pred = [1 if prob >= 0.5 else 0 for prob in max_prob]  # 每张切片由最大概率的 patch 得到的标签
            err, fpr, fnr = calc_err(pred, valset.labels)
            # 计算
            print('\nEpoch: [{}/{}]\tError: {}\tFPR: {}\tFNR: {}\n'
                  .format(epoch, total_epochs, err, fpr, fnr))
            fconv = open(os.path.join(args.output, 'convergence.csv'), 'a')
            fconv.write('{},error,{}\n'.format(epoch, err))
            fconv.write('{},fpr,{}\n'.format(epoch, fpr))
            fconv.write('{},fnr,{}\n'.format(epoch, fnr))
            fconv.close()

            # Save the best model
            acc = 1 - (fpr + fnr) / 2.
            if acc >= max_acc:
                max_acc = acc
                obj = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'max_accuracy': max_acc,
                    'optimizer': optimizer.state_dict()
                }
                torch.save(obj, os.path.join(output_path, 'checkpoint_best.pth'))


def calc_err(pred, real):
    pred = np.array(pred)
    real = np.array(real)
    neq = np.not_equal(pred, real)
    err = float(neq.sum()) / pred.shape[0]
    fpr = float(np.logical_and(pred == 1, neq).sum()) / (real == 0).sum()
    fnr = float(np.logical_and(pred == 0, neq).sum()) / (real == 1).sum()
    return err, fpr, fnr


if __name__ == "__main__":
    from dataset.datasets import LystoDataset

    print('Loading Dataset ...')
    imageSet = LystoDataset(filepath="data/training.h5", transform=trans,
                            interval=args.interval, size=args.patch_size, num_of_imgs=51)
    imageSet_val = LystoDataset(filepath="data/training.h5", transform=trans, train=False,
                                interval=args.interval, size=args.patch_size, num_of_imgs=51)

    train(imageSet, imageSet_val, batch_size=args.batch_size, workers=args.workers, total_epochs=args.epochs,
          test_every=args.test_every, model=model, criterion=criterion, optimizer=optimizer, topk=args.topk,
          output_path=args.output)
