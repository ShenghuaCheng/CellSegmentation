import os
import numpy as np
import argparse
from tqdm import tqdm
import csv
import cv2
from PIL import Image

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms

import model.resnet as models
from train import predict_patch

parser = argparse.ArgumentParser(description='Testing & Heatmap')
parser.add_argument('-m', '--model', type=str, default='checkpoint_10epochs.pth', help='path to pretrained model')
parser.add_argument('-b', '--batch_size', type=int, default=64, help='mini-batch size (default: 64)')
parser.add_argument('-w', '--workers', default=4, type=int, help='number of dataloader workers (default: 4)')
parser.add_argument('-k', '--topk', default=10, type=int,
                    help='top k tiles are assumed to be of the same class as the slide (default: 10, standard MIL)')
parser.add_argument('--interval', type=int, default=20, help='sample interval of patches (default: 20)')
parser.add_argument('--patch_size', type=int, default=32, help='size of each patch (default: 32)')
parser.add_argument('-d', '--device', type=str, default='0', help='CUDA device if available (default: \'0\')')
parser.add_argument('-o', '--output', type=str, default='./output', help='path of output details .csv file')
args = parser.parse_args()

if torch.cuda.is_available():
    torch.cuda.manual_seed(1)
    print('\nGPU is available.\n')
else:
    torch.manual_seed(1)

print("Testing settings: ")
print("Model: {} | Patches batch size: {} | Top-k: {}".format(args.model, args.batch_size, args.topk))
model = models.MILresnet18(pretrained=True)
model.fc_patch = nn.Linear(model.fc_patch.in_features, 2)
model.load_state_dict(torch.load(args.model)['state_dict'])

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
trans = transforms.Compose([transforms.ToTensor(), normalize])
# trans = transforms.ToTensor()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_VISIBLE_DEVICES'] = args.device
model.to(device)


def test(testset, batch_size, workers, model, topk, output_path):
    """
    :param testset:         ???????????????
    :param batch_size:      Dataloader ???????????? batch ??????
    :param workers:         Dataloader ??????????????????
    :param model:           ????????????
    :param topk:            ??????????????? k ?????????
    :param output_path:     ???????????????????????????
    """

    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=False)

    # open output file
    fconv = open(os.path.join(output_path, 'pred.csv'), 'w', newline="")
    w = csv.writer(fconv)
    w.writerow(['patch_size', '{}'.format(testset.size)])
    w.writerow(['interval', '{}'.format(testset.interval)])
    w.writerow(['idx', 'grid', 'prob'])
    fconv.close()
    # ??????????????? patch ?????????????????? output_path/pred.csv

    print('Start testing ...')

    # ?????????????????????
    model.setmode("patch")
    model.eval()
    probs = predict_patch(test_loader, batch_size)
    max_patches, max_probs = rank(testset, probs, topk)

    # ????????????
    heatmap(testset, max_patches, max_probs, topk, output_path)

def predict_patch(loader, batch_size):
    """???????????????????????????

    :param loader:          ?????????????????????
    :param batch_size:      DataLoader ???????????? batch ??????
    """
    global device

    probs = torch.Tensor(len(loader.dataset))
    with torch.no_grad():
        patch_bar = tqdm(loader, total=len(loader))
        for i, input in enumerate(patch_bar):
            patch_bar.set_postfix(step="testing",
                                  batch="[{}/{}]".format(i + 1, len(loader)))
            output = model(input.to(device)) # input: [b, c, h, w]
            output = F.softmax(output, dim=1)
            probs[i * batch_size:i * batch_size + input.size(0)] = output.detach()[:, 1].clone()
    return probs.cpu().numpy()


def rank(testset, probs, topk):
    """????????????????????? k ??? patch ??????????????????

    :param testset:     ?????????
    :param probs:       ???????????????
    :param topk:        ??????????????????
    :return:            ????????????????????????????????????
    """

    groups = np.array(testset.patchIDX)
    patches = np.array(testset.patches)

    order = np.lexsort((probs, groups))
    groups = groups[order]
    probs = probs[order]
    patches = patches[order]

    index = np.empty(len(groups), 'bool')
    index[-topk:] = True
    index[:-topk] = groups[topk:] != groups[:-topk]

    return patches[index], probs[index]


def heatmap(testset, patches, probs, topk, output_path):
    """???????????????????????????????????????????????????

    :param testset:         ?????????
    :param patches:         ??????????????????
    :param probs:           ?????????????????????
    :param topk:            ??????????????????
    :param output_path:     ??????????????????
    """

    for i, img in enumerate(testset.images):
        mask = np.zeros((img.shape[0], img.shape[1]))
        for idx in range(topk):
            patch_mask = np.full((testset.size, testset.size), probs[idx + i * topk])
            grid = list(map(int, patches[idx + i * topk]))
            mask[grid[0]: grid[0] + testset.size,
                 grid[1]: grid[1] + testset.size] = patch_mask
            # ????????????
            print("prob_{}:{}".format(i, probs[idx + i * topk]))
            fconv = open(os.path.join(output_path, 'pred.csv'), 'a', newline="")
            w = csv.writer(fconv)
            w.writerow([i, '{}'.format(grid), probs[idx + i * topk]])
            fconv.close()

        mask = cv2.applyColorMap(255 - np.uint8(255 * mask), cv2.COLORMAP_JET)
        img = img * 0.5 + mask * 0.5
        Image.fromarray(np.uint8(img)).save(os.path.join(output_path, "test_{}.png".format(i)))


if __name__ == "__main__":
    from dataset.datasets import LystoTestset


    print('Loading Dataset ...')
    imageSet_test = LystoTestset(filepath="data/testing.h5", transform=trans,
                                 interval=args.interval, size=args.patch_size, num_of_imgs=20)

    test(imageSet_test, batch_size=args.batch_size, workers=args.workers, model=model, topk=args.topk,
         output_path=args.output)
