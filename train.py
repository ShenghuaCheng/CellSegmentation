import warnings
import os
import numpy as np
import argparse
from tqdm import tqdm
import time
from datetime import datetime

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms
import sklearn.metrics as metrics
from torch.utils.tensorboard import SummaryWriter

import model.resnet as models
from utils.collate import default_collate

warnings.filterwarnings("ignore")

# Training settings
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('-e', '--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('-b', '--batch_size', type=int, default=32, help='mini-batch size of images (default: 32)')
parser.add_argument('-l', '--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('-w', '--workers', default=4, type=int, help='number of dataloader workers (default: 4)')
parser.add_argument('-t', '--test_every', default=1, type=int, help='test on val every (default: 1)')
parser.add_argument('-p', '--patches_per_pos', default=1, type=int,
                    help='k tiles are from a single positive cell (default: 1, standard MIL)')
parser.add_argument('-n', '--topk_neg', default=30, type=int,
                    help='top k tiles from a negative slide (default: 30, standard MIL)')
parser.add_argument('--patch_size', type=int, default=32, help='size of each slide (default: 32)')
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

trainset = None
valset = None

model = models.MILresnet18(pretrained=True)

if args.resume:
    resume = True
    model.load_state_dict(torch.load(args.resume)['state_dict'])

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
trans = transforms.Compose([transforms.ToTensor(), normalize])
# trans = transforms.ToTensor()

crit_cls = nn.CrossEntropyLoss()
crit_reg = nn.SmoothL1Loss()
crit_seg = None # TODO
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

os.environ['CUDA_VISIBLE_DEVICES'] = args.device
device = torch.device("cuda:" + args.device if torch.cuda.is_available() else "cpu")
model.to(device)


def train(batch_size, workers, total_epochs, test_every, model,
          crit_cls, crit_reg, crit_seg, optimizer, patches_per_pos, topk_neg, output_path):
    """one training epoch = patch mode -> slide mode

    :param batch_size:      DataLoader ???????????? batch ??????
    :param workers:         DataLoader ??????????????????
    :param total_epochs:    ???????????????
    :param test_every:      ????????????????????????????????????
    :param model:           ????????????
    :param crit_cls:        ?????????????????????
    :param crit_reg:        ??????????????????
    :param crit_seg:        ??????????????????
    :param optimizer:       ?????????
    :param patches_per_pos: ???**??????????????????**???????????? patch ??? (topk_pos = patches_per_pos * label)
    :param topk_neg:        ??????????????????????????????????????? top-k patch **??????**
    :param output_path:     ????????????????????????????????????????????????
    """

    global device, resume

    # shuffle ????????? False
    # ????????? patch ??????????????? slide ?????????????????? batch_size ????????????
    collate_fn = default_collate
    train_loader_forward = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=workers,
                                      pin_memory=True)
    train_loader_backward = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=workers,
                                       pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=workers,
                            pin_memory=True)

    # open output file
    fconv = open(os.path.join(output_path, 'training.csv'), 'w')
    fconv.write('epoch,patch_loss,slide_cls_loss,slide_reg_loss,slide_seg_loss,total_loss\n')
    fconv.close()
    # ????????????????????? output_path/training.csv
    fconv = open(os.path.join(output_path, 'validation.csv'), 'w')
    fconv.write('epoch,patch_error,patch_fpr,patch_fnr,slide_error,slide_fpr,slide_fnr,mae,mse\n')
    fconv.close()
    # ????????????????????? output_path/validation.csv

    print('Start training ...')
    # if resume:
    #     print('Resuming from the checkpoint (epochs: {}).'.format(model['epoch']))

    with SummaryWriter() as writer:
        for epoch in range(1, total_epochs + 1):
            start = time.time()

            # Forwarding step
            # ??? ResNet ?????????????????? 1000 ?????????????????????????????????????????????????????????????????????????????????
            model.fc_patch = nn.Linear(model.fc_patch.in_features, 2).to(device)
            trainset.setmode(1)
            probs = predict_patch(train_loader_forward, batch_size, epoch, total_epochs)
            sample(probs, patches_per_pos, topk_neg)

            # Alternative training step
            trainset.setmode(2)
            if epoch == total_epochs:
                trainset.visualize_bboxes()  # patch visualize testing
            alpha = 1.
            beta = 0.1
            gamma = 0.1
            delta = 0.1
            loss = train_alternative(train_loader_backward, batch_size, epoch, total_epochs, model, crit_cls,
                                     crit_reg, crit_seg, optimizer, alpha, beta, gamma, delta)

            end = time.time()

            print("""patch loss: {:.4f} | slide cls loss: {:.4f} | slide reg loss: {:.4f} | slide seg loss: {:.4f}
total loss: {:.4f}""".format(*loss))
            print("Runtime: {}s".format((end - start) / 1000))
            fconv = open(os.path.join(output_path, 'training.csv'), 'a')
            fconv.write('{},{},{},{},{},{}\n'.format(epoch, *loss))
            fconv.close()

            writer.add_scalar("loss", loss[4], epoch)
            writer.add_scalar("patch loss", loss[0], epoch)
            writer.add_scalar("slide cls loss", loss[1], epoch)
            writer.add_scalar("slide reg loss", loss[2], epoch)
            writer.add_scalar("slide seg loss", loss[3], epoch)

            # Validating step
            if (epoch + 1) % test_every == 0:
                valset.setmode(1)
                print('Validating ...')

                probs_p = predict_patch(val_loader, batch_size, epoch, total_epochs)
                metrics_p = validation_patch(probs_p)
                print('patch error: {} | patch FPR: {} | patch FNR: {}'.format(*metrics_p))

                writer.add_scalar('patch error rate', metrics_p[0], epoch)
                writer.add_scalar('patch false positive rate', metrics_p[1], epoch)
                writer.add_scalar('patch false negative rate', metrics_p[2], epoch)

                # slide validating
                valset.setmode(4)
                probs_s, reg, seg = predict_slide(val_loader, batch_size, epoch, total_epochs)
                metrics_s = validation_slide(probs_s, reg, seg)
                print('slide error: {} | slide FPR: {} | slide FNR: {}\nMAE: {} | MSE: {}\n'.format(*metrics_s))
                fconv = open(os.path.join(output_path, 'validation.csv'), 'a')
                fconv.write('{},{},{},{},{},{},{},{},{}\n'.format(epoch, *(metrics_p + metrics_s)))
                fconv.close()

                writer.add_scalar('slide error rate', metrics_s[0], epoch)
                writer.add_scalar('slide false positive rate', metrics_s[1], epoch)
                writer.add_scalar('slide false negative rate', metrics_s[2], epoch)
                writer.add_scalar('slide mae', metrics_s[3], epoch)
                writer.add_scalar('slide mse', metrics_s[4], epoch)

                # ??????????????????????????????
                obj = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                torch.save(obj, os.path.join(output_path, 'checkpoint_{}epochs.pth'.format(epoch)))


def predict_patch(loader, batch_size, epoch, total_epochs):
    """??????????????????????????????????????????????????????

    :param loader:          ?????????????????????
    :param batch_size:      DataLoader ???????????? batch ??????
    :param epoch:           ??????????????????
    :param total_epochs:    ???????????????
    """
    global device

    model.setmode("patch")
    model.eval()

    probs = torch.Tensor(len(loader.dataset))
    with torch.no_grad():
        patch_bar = tqdm(loader, total=len(loader.dataset) // batch_size + 1)
        for i, input in enumerate(patch_bar):
            patch_bar.set_postfix(step="patch forwarding",
                                  epoch="[{}/{}]".format(epoch, total_epochs),
                                  batch="[{}/{}]".format(i + 1, len(loader.dataset) // batch_size + 1))
            # softmax ?????? [[a,b],[c,d]] shape = batch_size*2
            output = model(input[0].to(device)) # input: [2, b, c, h, w]
            output = F.softmax(output, dim=1)
            # detach()[:,1] ?????? softmax ???????????????????????????[b, d, ...]
            # input.size(0) ?????? batch ??????????????????
            probs[i * batch_size:i * batch_size + input[0].size(0)] = output.detach()[:, 1].clone()
    return probs.cpu().numpy()


def sample(probs, patches_per_pos, topk_neg):
    """??????????????? top-k ?????????????????????????????????????????????

    :param probs:           predict_patch() ?????????????????????
    :param patches_per_pos: ???**??????????????????**???????????? patch ??? (topk_pos = patches_per_pos * label)
    :param topk_neg:        ??????????????????????????????????????? top-k patch **??????**
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
    """??????????????????????????????????????????????????????????????????????????????

    :param loader:          ?????????????????????
    :param batch_size:      DataLoader ???????????? batch ??????
    :param epoch:           ??????????????????
    :param total_epochs:    ???????????????
    :return:                ????????????????????????????????????????????????
    """

    model.setmode("slide")
    model.eval()

    probs = torch.tensor(())
    nums = torch.tensor(())
    feats = torch.tensor(())
    with torch.no_grad():
        slide_bar = tqdm(loader, total=len(loader.dataset) // batch_size + 1)
        for i, (data, label_cls, label_num, _) in enumerate(slide_bar):
            slide_bar.set_postfix(step="slide forwarding",
                                  epoch="[{}/{}]".format(epoch, total_epochs),
                                  batch="[{}/{}]".format(i + 1, len(loader.dataset) // batch_size + 1))
            output = model(data.to(device))
            output_cls = F.softmax(output[0], dim=1)
            probs = torch.cat((probs, output_cls.detach()[:, 1].clone().cpu()), dim=0)
            nums = torch.cat((nums, output[1].detach()[:, 0].clone().cpu()), dim=0)
            feats = torch.cat((feats, output[2].detach().clone().cpu()), dim=0)
    return probs.numpy(), nums.numpy(), feats.numpy()


def train_patch(loader, epoch, total_epochs, model, criterion, optimizer):
    """Patch training for one epoch.

    :param loader:          ?????????????????????
    :param epoch:           ??????????????????
    :param total_epochs:    ???????????????
    :param model:           ????????????
    :param criterion:       ???????????????????????????????????????criterion_cls???
    :param optimizer:       ?????????
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
    return train_loss


def train_alternative(loader, batch_size, epoch, total_epochs, model, crit_cls, crit_reg, crit_seg, optimizer, alpha, beta, gamma, delta):
    """patch + slide training for one epoch. slide mode = slide_cls + slide_reg + slide_seg

    :param loader:          ?????????????????????
    :param batch_size:      DataLoader ???????????? batch ??????
    :param epoch:           ??????????????????
    :param total_epochs:    ???????????????
    :param model:           ????????????
    :param crit_cls:        ?????????????????????
    :param crit_reg:        ??????????????????
    :param crit_seg:        ??????????????????
    :param optimizer:       ?????????
    :param alpha:           patch_loss ??????
    :param beta:            slide_cls_loss ??????
    :param gamma:           slide_reg_loss ??????
    :param delta:           slide_seg_loss ??????
    """

    global device

    model.train()

    patch_num = 0
    patch_loss = 0.
    slide_cls_loss = 0.
    slide_reg_loss = 0.
    slide_seg_loss = 0.
    total_loss = 0.

    train_bar = tqdm(loader, total=len(loader.dataset) // batch_size + 1)
    for i, (data, labels) in enumerate(train_bar):
        train_bar.set_postfix(step="alternative training",
                              epoch="[{}/{}]".format(epoch, total_epochs),
                              batch="[{}/{}]".format(i + 1, len(loader.dataset) // batch_size + 1))

        # Patch training
        model.setmode("patch")
        # print("slides pack size:", data[0].size())
        # print("patches pack size:", data[1].size())
        output = model(data[1].to(device))

        patch_loss_i = crit_cls(output, labels[3].to(device))
        patch_loss += patch_loss_i.item() * data[1].size(0)
        patch_num += data[1].size(0)

        # Slide training
        model.setmode("slide")

        output = model(data[0].to(device))
        optimizer.zero_grad()

        slide_cls_loss_i = crit_cls(output[0], labels[0].to(device))
        slide_reg_loss_i = crit_reg(output[1].squeeze(), labels[1].to(device, dtype=torch.float32))
        # slide_seg_loss_i = crit_seg(output[2], labels[2].to(device))

        # total_loss_i = alpha * patch_loss_i + beta * slide_cls_loss_i + \
        #                gamma * slide_reg_loss_i + delta * slide_seg_loss_i
        total_loss_i = alpha * patch_loss_i + beta * slide_cls_loss_i + gamma * slide_reg_loss_i
        total_loss_i.backward()
        optimizer.step()

        slide_cls_loss += slide_cls_loss_i.item() * data[0].size(0)
        slide_reg_loss += slide_reg_loss_i.item() * data[0].size(0)
        # slide_seg_loss += slide_seg_loss_i.item() * slide_data[0].size(0)
        total_loss += total_loss_i.item() * data[0].size(0)

        # print("slide data size:", data[0].size(0))
        # print("patch data size:", data[1].size(0))

    # print("Total patches:", patch_num)
    # print("Total slides:", len(loader.dataset))

    total_loss /= len(loader.dataset)
    patch_loss /= patch_num
    slide_cls_loss /= len(loader.dataset)
    slide_reg_loss /= len(loader.dataset)
    # slide_seg_loss /= len(loader.dataset)
    slide_seg_loss = 0.
    return patch_loss, slide_cls_loss, slide_reg_loss, slide_seg_loss, total_loss

def validation_patch(probs):
    """patch mode ?????????"""

    val_groups = np.array(valset.patchIDX)

    max_prob = np.empty(len(valset.labels))  # ???????????????????????????????????????????????????????????????????????? patch
    max_prob[:] = np.nan
    order = np.lexsort((probs, val_groups))
    # ??????
    val_groups = val_groups[order]
    val_probs = probs[order]
    # ?????????
    val_index = np.empty(len(val_groups), 'bool')
    val_index[-1] = True
    val_index[:-1] = val_groups[1:] != val_groups[:-1]
    max_prob[val_groups[val_index]] = val_probs[val_index]

    # ??????????????????FPR???FNR
    probs = np.round(max_prob)  # ?????????????????????????????? patch ???????????????
    err, fpr, fnr = calc_err(probs, np.sign(valset.labels))
    return err, fpr, fnr

def validation_slide(probs, reg, seg):
    """slide mode ?????????"""

    probs = np.round(probs)
    err, fpr, fnr = calc_err(probs, np.sign(valset.labels))
    mae = metrics.mean_absolute_error(valset.labels, reg)
    mse = metrics.mean_squared_error(valset.labels, reg)
    return err, fpr, fnr, mae, mse

def calc_err(pred, real):
    """????????????????????????????????????????????????????????????"""

    pred = np.asarray(pred)
    real = np.asarray(real)
    neq = np.not_equal(pred, real)
    err = float(neq.sum()) / pred.shape[0] # ????????? = ?????????????????? / ??????
    fpr = float(np.logical_and(pred == 1, neq).sum()) / (real == 0).sum() # ???????????? = ????????? / ???????????????
    fnr = float(np.logical_and(pred == 0, neq).sum()) / (real == 1).sum() # ???????????? = ????????? / ???????????????
    return err, fpr, fnr


if __name__ == "__main__":
    from dataset.dataset import LystoDataset

    print("Training settings: ")
    print("Epochs: {} | Validate every {} iteration(s) | Slide batch size: {} | Negative top-k: {}"
          .format(args.epochs, args.test_every, args.batch_size, args.topk_neg))

    print('Loading Dataset ...')
    trainset = LystoDataset(filepath="data/training.h5", transform=trans)
    valset = LystoDataset(filepath="data/training.h5", train=False, transform=trans)

    train(batch_size=args.batch_size,
          workers=args.workers,
          total_epochs=args.epochs,
          test_every=args.test_every,
          model=model,
          crit_cls=crit_cls,
          crit_reg=crit_reg,
          crit_seg=crit_seg,
          optimizer=optimizer,
          patches_per_pos=args.patches_per_pos,
          topk_neg=args.topk_neg,
          output_path=args.output)
