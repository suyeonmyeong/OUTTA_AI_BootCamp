from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import random
import os


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    # Calculate the scaling factor based on the kernel size
    factor = (kernel_size + 1) // 2

    # Determine the center of the kernel based on whether the kernel size is odd or even
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5

    # Create an open grid of kernel size dimensions
    og = np.ogrid[:kernel_size, :kernel_size]

    # Calculate the bilinear filter using the open grid and scaling factor
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

    # Initialize the weight matrix with zeros and set the diagonal elements to the calculated filter
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt

    # Convert the NumPy array to a PyTorch tensor and return it
    return torch.from_numpy(weight).float()


def segmentation_output(mask, num_classes=21):
    label_colours = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (64, 0, 0), (128, 0, 128), (0, 128, 128), (128, 128, 128), (0, 0, 128), (192, 0, 0), (
        64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)]
    # 0=background
    # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
    # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
    # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
    # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor

    h, w = mask.shape

    img = Image.new('RGB', (w, h))
    pixels = img.load()
    for j_, j in enumerate(mask[:, :]):
        for k_, k in enumerate(j):
            if k < num_classes:
                pixels[k_, j_] = label_colours[k]
    output = np.array(img)

    return output


def read_file(path_to_file):
    with open(path_to_file) as f:
        img_list = []
        for line in f:
            img_list.append(line[:-1])
    return img_list


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def flip(I, flip_p):
    if flip_p > 0.5:
        return np.fliplr(I)
    else:
        return I


def scale_im(img_temp, scale):
    new_dims = (int(img_temp.shape[0] * scale), int(img_temp.shape[1] * scale))
    return cv2.resize(img_temp, new_dims).astype(float)


def get_data(chunk, gt_path='./gt', img_path='./img'):
    assert len(chunk) == 1

    scale = random.uniform(0.5, 1.3)
    flip_p = random.uniform(0, 1)

    images = cv2.imread(os.path.join(
        img_path, chunk[0] + '.jpg')).astype(float)

    images = cv2.resize(images, (321, 321)).astype(float)
    images = scale_im(images, scale)
    images[:, :, 0] = images[:, :, 0] - 104.008
    images[:, :, 1] = images[:, :, 1] - 116.669
    images[:, :, 2] = images[:, :, 2] - 122.675
    images = flip(images, flip_p)
    images = images[:, :, :, np.newaxis]
    images = images.transpose((3, 2, 0, 1))
    images = torch.from_numpy(images.copy()).float()

    gt = cv2.imread(os.path.join(gt_path, chunk[0] + '.png'))[:, :, 0]
    gt[gt == 255] = 0
    gt = flip(gt, flip_p)

    dim = int(321 * scale)

    gt = cv2.resize(
        gt, (dim, dim), interpolation=cv2.INTER_NEAREST).astype(float)

    labels = gt[np.newaxis, :].copy()

    return images, labels


def validation_miou(model):
    max_label = 20
    hist = np.zeros((max_label + 1, max_label + 1))

    def fast_hist(a, b, n):
        k = (a >= 0) & (a < n)
        return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

    val_list = open('./list/val.txt').readlines()
    print('mIoU 계산 시작!')
    with torch.no_grad():
        for idx, i in enumerate(val_list):
            img = cv2.imread(os.path.join(
                './img', i[:-1] + '.jpg')).astype(float)

            img[:, :, 0] -= 104.008
            img[:, :, 1] -= 116.669
            img[:, :, 2] -= 122.675

            data = torch.from_numpy(img.transpose(
                (2, 0, 1))).float().cuda().unsqueeze(0)
            score = model(data)

            output = score.cpu().data[0].numpy().transpose(1, 2, 0)
            output = np.argmax(output, axis=2)
            gt = cv2.imread(os.path.join('./gt', i[:-1] + '.png'), 0)

            hist += fast_hist(gt.flatten(), output.flatten(), max_label + 1)

        miou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        print("mIoU = ", np.sum(miou) / len(miou))

    return np.sum(miou) / len(miou)
