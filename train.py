import os
import numpy as np
import argparse
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms
import sklearn.metrics as metrics
from torch.utils.tensorboard import SummaryWriter

import model.resnet as models

# Training settings
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('-e', '--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('-b', '--batch_size', type=int, default=64, help='mini-batch size (default: 64)')
parser.add_argument('-s', '--slide_mode', action='store_true', help='using the slide training part or not')
parser.add_argument('-l', '--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('-w', '--workers', default=4, type=int, help='number of dataloader workers (default: 4)')
parser.add_argument('-t', '--test_every', default=1, type=int, help='test on val every (default: 1)')
parser.add_argument('-p', '--patches_per_pos', default=1, type=int,
                    help='k tiles are from a single positive cell (default: 1, standard MIL)')
parser.add_argument('-n', '--topk_neg', default=30, type=int,
                    help='top k tiles from a negative slide (default: 30, standard MIL)')
parser.add_argument('--interval', type=int, default=20, help='sample interval of patches (default: 20)')
parser.add_argument('--patch_size', type=int, default=32, help='size of each patch (default: 32)')
parser.add_argument('-d', '--device', type=str, default='0', help='CUDA device if available (default: \'0\')')
parser.add_argument('-o', '--output', type=str, default='.', help='name of output file')
parser.add_argument('-r', '--resume', action='store_true', help='continue training from a checkpoint file.pth')
args = parser.parse_args()

if torch.cuda.is_available():
    torch.cuda.manual_seed(1)
    print('\nGPU is available.\n')
else:
    torch.manual_seed(1)

max_acc = 0
resume = False
verbose = True

print('Init Model ...')

model = models.MILresnet18(pretrained=True)

if args.resume:
    resume = True
    model.load_state_dict(torch.load(args.resume)['state_dict'])

# normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# trans = transforms.Compose([transforms.ToTensor(), normalize])
trans = transforms.ToTensor()

criterion_cls = nn.CrossEntropyLoss()
criterion_reg = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

os.environ['CUDA_VISIBLE_DEVICES'] = args.device
device = torch.device("cuda:" + args.device if torch.cuda.is_available() else "cpu")
model.to(device)


def train(trainset, valset, batch_size, slide_mode, workers, total_epochs, test_every, model,
          criterion_cls, criterion_reg, optimizer, patches_per_pos, topk_neg, output_path):
    """one training epoch = patch mode -> slide mode

    :param trainset:        训练数据集
    :param valset:          验证数据集
    :param batch_size:      DataLoader 打包的小 batch 大小
    :param slide_mode:      是否启用全图训练模式
    :param workers:         DataLoader 使用的进程数
    :param total_epochs:    迭代总次数
    :param test_every:      每验证一轮间隔的迭代次数
    :param model:           网络模型
    :param criterion_cls:   分类器损失函数
    :param criterion_reg:   回归损失函数
    :param optimizer:       优化器
    :param patches_per_pos: 在**单个阳性细胞**上选取的 patch 数 (topk_pos = patches_per_pos * label)
    :param topk_neg:        每次在阴性细胞图像上选取的 top-k patch **总数**
    :param output_path:     保存模型文件和训练数据结果的目录
    """

    global device, resume

    # shuffle 只能是 False
    # 暂定对 patch 的训练和对 slide 的训练所用的 batch_size 是一样的
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=workers,
                              pin_memory=False)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=workers,
                            pin_memory=False)

    # open output file
    fconv = open(os.path.join(output_path, 'training.csv'), 'w')
    fconv.write('epoch,mode,value\n')
    fconv.close()
    # 训练结果保存在 output_path/training.csv
    fconv = open(os.path.join(output_path, 'validation.csv'), 'w')
    fconv.write('epoch,mode,value\n')
    fconv.close()
    # 验证结果保存在 output_path/validation.csv

    print('Start training ...')
    # if resume:
    #     print('Resuming from the checkpoint (epochs: {}).'.format(model['epoch']))

    with SummaryWriter() as writer:
        for epoch in range(1, total_epochs + 1):

            # Forwarding step
            trainset.setmode(1)
            model.setmode("patch")
            model.eval()
            # 把 ResNet 源码中的分为 1000 类改为二分类（由于预训练模型文件的限制，只能在外面改）
            model.fc_patch = nn.Linear(model.fc_patch.in_features, 2).to(device)
            probs = predict_patch(train_loader, batch_size, epoch, total_epochs)
            sample(trainset, probs, patches_per_pos, topk_neg)

            # Alternative training step
            trainset.setmode(2)
            model.train()
            patch_loss = train_patch(train_loader, epoch, total_epochs, model, criterion_cls,
                                     optimizer, output_path)
            writer.add_scalar('patch loss', patch_loss, epoch)

            if slide_mode:
                trainset.setmode(3)
                model.setmode("slide")
                model.train()
                slide_loss = train_slide(train_loader, batch_size, epoch, total_epochs, model, criterion_cls,
                                         criterion_reg, optimizer, 1, 1, output_path)
                writer.add_scalar('slide loss', slide_loss, epoch)

            # Validating step
            if (epoch + 1) % test_every == 0:
                valset.setmode(1)
                model.setmode("patch")
                model.eval()
                print('Validating ...')
                # 把训练过的 model 在验证集上做一下
                probs_p = predict_patch(val_loader, batch_size, epoch, total_epochs)
                err_p, fpr_p, fnr_p = validation_patch(valset, probs_p, epoch, total_epochs, output_path)
                writer.add_scalar('patch error rate', err_p, epoch)
                writer.add_scalar('patch false positive rate', fpr_p, epoch)
                writer.add_scalar('patch false negative rate', fnr_p, epoch)

                if slide_mode:
                    valset.setmode(3)
                    model.setmode("slide")
                    probs_i, reg, seg = predict_slide(val_loader, batch_size, epoch, total_epochs)
                    err_i, fpr_i, fnr_i, mae, mse = validation_slide(valset, probs_i, reg, seg, epoch, total_epochs,
                                                                     output_path)
                    writer.add_scalar('slide error rate', err_i, epoch)
                    writer.add_scalar('slide false positive rate', fpr_i, epoch)
                    writer.add_scalar('slide false negative rate', fnr_i, epoch)
                    writer.add_scalar('slide mae', mae, epoch)
                    writer.add_scalar('slide mse', mse, epoch)

                # 每验证一次，保存模型
                obj = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                torch.save(obj, os.path.join(output_path, 'checkpoint_best.pth'))


def predict_patch(loader, batch_size, epoch, total_epochs):
    """前馈推导一次模型，获取实例分类概率。

    :param loader:          训练集的迭代器
    :param batch_size:      DataLoader 打包的小 batch 大小
    :param epoch:           当前迭代次数
    :param total_epochs:    迭代总次数
    """
    global device

    probs = torch.Tensor(len(loader.dataset))
    with torch.no_grad():
        patch_bar = tqdm(loader, total=len(loader.dataset) // batch_size + 1)
        for i, input in enumerate(patch_bar):
            patch_bar.set_postfix(step="patch forwarding",
                                  epoch="[{}/{}]".format(epoch, total_epochs),
                                  batch="[{}/{}]".format(i + 1, len(loader.dataset) // batch_size + 1))
            # softmax 输出 [[a,b],[c,d]] shape = batch_size*2
            output = model(input[0].to(device)) # input: [2, b, c, h, w]
            output = F.softmax(output, dim=1)
            # detach()[:,1] 取出 softmax 得到的概率，产生：[b, d, ...]
            # input.size(0) 返回 batch 中的实例数量
            probs[i * batch_size:i * batch_size + input[0].size(0)] = output.detach()[:, 1].clone()
    return probs.cpu().numpy()


def sample(trainset, probs, patches_per_pos, topk_neg):
    """找出概率为 top-k 的补丁，制作迭代使用的数据集。

    :param trainset:        训练数据集
    :param probs:           predict_patch() 得到的补丁概率
    :param patches_per_pos: 在**单个阳性细胞**上选取的 patch 数 (topk_pos = patches_per_pos * label)
    :param topk_neg:        每次在阴性细胞图像上选取的 top-k patch **总数**
    """

    global verbose

    groups = np.array(trainset.patchIDX)
    order = np.lexsort((probs, groups))

    index = np.empty(len(trainset), 'bool')
    for i in range(len(trainset)):
        topk = topk_neg if trainset.labels[groups[i]] == 0 else trainset.labels[groups[i]] * patches_per_pos
        index[i] = groups[i] != groups[(i + topk) % len(groups)]

    p, n = trainset.make_train_data(list(order[index]))
    if verbose:
        print("Training data is sampled. \nPos samples: {} | Neg samples: {}".format(p, n))

def predict_slide(loader, batch_size, epoch, total_epochs):
    """前馈推导一次模型，获取图像级的分类概率和回归预测值。
    """

    probs = torch.tensor(())
    nums = torch.tensor(())
    feats = torch.tensor(())
    with torch.no_grad():
        slide_bar = tqdm(loader, total=len(loader.dataset) // batch_size + 1)
        for i, (data, label_cls, label_num) in enumerate(slide_bar):
            slide_bar.set_postfix(step="slide forwarding",
                                  epoch="[{}/{}]".format(epoch, total_epochs),
                                  batch="[{}/{}]".format(i + 1, len(loader.dataset) // batch_size + 1))
            output = model(data.to(device))
            output_cls = F.softmax(output[0], dim=1)
            probs = torch.cat((probs, output_cls.detach()[:, 1].clone().cpu()), dim=0)
            nums = torch.cat((nums, output[1].detach()[:, 0].clone().cpu()), dim=0)
            feats = torch.cat((feats, output[2].detach().clone().cpu()), dim=0)
    return probs.numpy(), nums.numpy(), feats.numpy()


def train_patch(loader, epoch, total_epochs, model, criterion, optimizer, output_path):
    """Patch training for one epoch.

    :param loader:          训练集的迭代器
    :param epoch:           当前迭代次数
    :param total_epochs:    迭代总次数
    :param model:           网络模型
    :param criterion:       用于补丁级训练的损失函数（criterion_cls）
    :param optimizer:       优化器
    :param output_path:     保存训练结果数据的目录
    """
    global device

    train_loss = 0.
    train_bar = tqdm(loader, total=len(loader))
    for i, (data, label, _) in enumerate(train_bar):
        train_bar.set_postfix(step="patch training",
                              epoch="[{}/{}]".format(epoch, total_epochs),
                              batch="[{}/{}]".format(i + 1, len(loader)))

        output = model(data.to(device))
        optimizer.zero_grad()
        loss = criterion(output, label.to(device))
        train_loss += loss.item() * data.size(0)
        loss.backward()
        optimizer.step()

    train_loss /= len(loader.dataset)
    print('Epoch: [{}/{}], Loss: {:.4f}'.format(epoch, total_epochs, train_loss))
    fconv = open(os.path.join(output_path, 'training.csv'), 'a')
    fconv.write('{},patch,{}\n'.format(epoch, train_loss))
    fconv.close()

    return train_loss

def train_slide(loader, batch_size, epoch, total_epochs, model, criterion_cls, criterion_reg, optimizer, alpha, beta,
                output_path):
    """slide training for one epoch. slide mode = slide_cls + slide_reg + slide_seg

    :param loader:          训练集的迭代器
    :param batch_size:      DataLoader 打包的小 batch 大小
    :param epoch:           当前迭代次数
    :param total_epochs:    迭代总次数
    :param model:           网络模型
    :param criterion_cls:   分类器损失函数
    :param criterion_reg:   回归损失函数
    :param optimizer:       优化器
    :param alpha:           loss 参数
    :param beta:            loss 参数
    :param output_path:     保存训练结果数据的目录
    """
    global device

    train_loss = 0.
    train_bar = tqdm(loader, total=len(loader.dataset) // batch_size + 1)
    for i, (data, label_cls, label_num) in enumerate(train_bar):
        train_bar.set_postfix(step="slide training",
                              epoch="[{}/{}]".format(epoch, total_epochs),
                              batch="[{}/{}]".format(i + 1, len(loader.dataset) // batch_size + 1))

        output = model(data.to(device))
        optimizer.zero_grad()

        loss_cls = criterion_cls(output[0], label_cls.to(device))  # 图片分类损失
        loss_reg = criterion_reg(output[1].squeeze(), label_num.to(device, dtype=torch.float32))  # 回归数目损失
        loss = alpha * loss_cls + beta * loss_reg

        train_loss += loss.item() * data.size(0)
        loss.backward()
        optimizer.step()

    train_loss /= len(loader.dataset)
    print('Epoch: [{}/{}], Loss_cls: {:.4f}, Loss_reg: {:.4f}'.format(epoch, total_epochs, loss_cls, loss_reg))
    fconv = open(os.path.join(output_path, 'training.csv'), 'a')
    fconv.write('{},slide_cls,{}\n'.format(epoch, loss_cls))
    fconv.write('{},slide_reg,{}\n'.format(epoch, loss_reg))
    fconv.close()

    return train_loss

def validation_patch(valset, probs, epoch, total_epochs, output_path):
    """patch mode 的验证"""

    val_groups = np.array(valset.patchIDX)

    max_prob = np.empty(len(valset.labels))  # 模型预测的实例最大概率列表，每张切片取最大概率的 patch
    max_prob[:] = np.nan
    order = np.lexsort((probs, val_groups))
    # 排序
    val_groups = val_groups[order]
    val_probs = probs[order]
    # 取最大
    val_index = np.empty(len(val_groups), 'bool')
    val_index[-1] = True
    val_index[:-1] = val_groups[1:] != val_groups[:-1]
    max_prob[val_groups[val_index]] = val_probs[val_index]

    # 计算错误率、FPR、FNR
    probs = np.round(max_prob)  # 每张切片由最大概率的 patch 得到的标签
    err, fpr, fnr = calc_err(probs, np.sign(valset.labels))
    print('Epoch: [{}/{}]\tpatch Error: {}\tpatch FPR: {}\tpatch FNR: {}'
          .format(epoch, total_epochs, err, fpr, fnr))
    fconv = open(os.path.join(output_path, 'validation.csv'), 'a')
    fconv.write('{},patch_error,{}\n'.format(epoch, err))
    fconv.write('{},patch_fpr,{}\n'.format(epoch, fpr))
    fconv.write('{},patch_fnr,{}\n'.format(epoch, fnr))
    fconv.close()

    return err, fpr, fnr

def validation_slide(valset, probs, reg, seg, epoch, total_epochs, output_path):

    """slide mode 的验证"""
    probs = np.round(probs)
    err, fpr, fnr = calc_err(probs, np.sign(valset.labels))
    mae = metrics.mean_absolute_error(valset.labels, reg)
    mse = metrics.mean_squared_error(valset.labels, reg)
    print('\nEpoch: [{}/{}]\tslide Error: {}\tslide FPR: {}\tslide FNR: {}\nMAE: {}\tMSE: {}'
          .format(epoch, total_epochs, err, fpr, fnr, mae, mse))
    fconv = open(os.path.join(output_path, 'validation.csv'), 'a')
    fconv.write('{},slide_err,{}\n'.format(epoch, err))
    fconv.write('{},slide_fpr,{}\n'.format(epoch, fpr))
    fconv.write('{},slide_fnr,{}\n'.format(epoch, fnr))
    fconv.write('{},mae,{}\n'.format(epoch, mae))
    fconv.write('{},mse,{}\n'.format(epoch, mse))
    fconv.close()

    return err, fpr, fnr, mae, mse

def calc_err(pred, real):
    """计算分类任务的错误率、假阳性率、假阴性率"""
    pred = np.asarray(pred)
    real = np.asarray(real)
    neq = np.not_equal(pred, real)
    err = float(neq.sum()) / pred.shape[0] # 错误率 = 预测错误的和 / 总和
    fpr = float(np.logical_and(pred == 1, neq).sum()) / (real == 0).sum() # 假阳性率 = 假阳性 / 所有的阴性
    fnr = float(np.logical_and(pred == 0, neq).sum()) / (real == 1).sum() # 假阴性率 = 假阴性 / 所有的阳性
    return err, fpr, fnr


if __name__ == "__main__":
    from dataset.datasets import LystoDataset

    print("Training settings: ")
    print("Epochs: {} | Validate every {} iteration(s) | Patches batch size: {} | Slide mode: {} | Negative top-k: {}"
          .format(args.epochs, args.test_every, args.batch_size, "on" if args.slide_mode else "off", args.topk_neg))

    print('Loading Dataset ...')
    imageSet = LystoDataset(filepath="data/training.h5", transform=trans,
                            interval=args.interval, size=32)
    imageSet_val = LystoDataset(filepath="data/training.h5", transform=trans, train=False,
                                interval=args.interval, size=32)


    train(imageSet, imageSet_val,
          batch_size=args.batch_size,
          slide_mode=args.slide_mode,
          workers=args.workers,
          total_epochs=args.epochs,
          test_every=args.test_every,
          model=model,
          criterion_cls=criterion_cls,
          criterion_reg=criterion_reg,
          optimizer=optimizer,
          patches_per_pos=args.patches_per_pos,
          topk_neg=args.topk_neg,
          output_path=args.output)
