import random

import cv2
import numpy as np
import os

import torch
from PIL import Image

from torch.utils.data import Dataset

EXTENSIONS = ['.jpg', '.png']

def load_image(file):
    return Image.open(file)

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def is_label(filename):
    return filename.endswith("_labelTrainIds.png")

def image_path(root, basename, extension):
    return os.path.join(root, f'{basename}{extension}')

def image_path_city(root, name):
    return os.path.join(root, f'{name}')

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

class VOC12(Dataset):

    def __init__(self, root, input_transform=None, target_transform=None):
        self.images_root = os.path.join(root, 'images')
        self.labels_root = os.path.join(root, 'labels')

        self.filenames = [image_basename(f)
            for f in os.listdir(self.labels_root) if is_image(f)]
        self.filenames.sort()

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        filename = self.filenames[index]

        with open(image_path(self.images_root, filename, '.jpg'), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path(self.labels_root, filename, '.png'), 'rb') as f:
            label = load_image(f).convert('P')

        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.filenames)


class cityscapes(Dataset):

    def __init__(self, root, co_transform=None, subset='train'):
        self.images_root = os.path.join(root, 'leftImg8bit/')
        self.labels_root = os.path.join(root, 'gtFine/')
        
        self.images_root += subset
        self.labels_root += subset

        print (self.images_root)
        #self.filenames = [image_basename(f) for f in os.listdir(self.images_root) if is_image(f)]
        self.filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.images_root)) for f in fn if is_image(f)]
        self.filenames.sort()

        #[os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(".")) for f in fn]
        #self.filenamesGt = [image_basename(f) for f in os.listdir(self.labels_root) if is_image(f)]
        self.filenamesGt = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.labels_root)) for f in fn if is_label(f)]
        self.filenamesGt.sort()

        self.co_transform = co_transform # ADDED THIS


    def __getitem__(self, index):
        filename = self.filenames[index]
        filenameGt = self.filenamesGt[index]

        with open(image_path_city(self.images_root, filename), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path_city(self.labels_root, filenameGt), 'rb') as f:
            label = load_image(f).convert('P')

        if self.co_transform is not None:
            image, label = self.co_transform(image, label)

        return image, label

    def __len__(self):
        return len(self.filenames)


class AutoVpDataset(Dataset):
    def __init__(self, label_file_path, root_dir, length=0):
        self.length = length
        self.root_dir = root_dir
        with open(label_file_path, 'r') as label:
            self.data = [x.strip().split('\t') for x in label.readlines()]

    def __getitem__(self, index):
        image = cv2.imread(self.root_dir + self.data[index][0])
        image = cv2.resize(image, (820, 295)).astype(np.float)
        image /= 255
        image -= 0.5
        # cv2.imshow("resized image", image)
        label = np.load(self.root_dir + self.data[index][1])
        label = cv2.resize(label, (820, 295))
        # cv2.imshow("resized label", label)

        horizontal_flip = random.randint(0, 1)
        vertical_shift = random.randint(0, 1)
        if horizontal_flip == 0:
            image = image[:, ::-1, :].copy()  # it will cause memory discontinuous after channel changing
            label = label[:, ::-1].copy()
            # cv2.imshow("flipped image", image)
            # cv2.imshow("flipped label", label)
        if vertical_shift == 0:
            vy = int(float(self.data[index][3]) / 2)
            offset = random.randint(-vy, 295 - vy)
            if offset <= 0:
                y_new1 = 0
                y_new2 = 295 - 1 + offset
                y_old1 = -offset
                y_old2 = 295 - 1
            else:
                y_new1 = offset
                y_new2 = 295 - 1
                y_old1 = 0
                y_old2 = 295 - offset - 1

            shifted_image = np.zeros_like(image)
            shifted_label = np.zeros_like(label)
            shifted_image[y_new1:y_new2, :, :] = image[y_old1: y_old2, :, :]
            shifted_label[y_new1:y_new2, :] = label[y_old1: y_old2, :]
            image = shifted_image
            label = shifted_label
            # cv2.imshow("shifted image", image)
            # cv2.imshow("shifted label", label)
            # cv2.waitKey()
        image = image.transpose((2, 0, 1))
        label = label[np.newaxis, :, :]

        return torch.from_numpy(image).float(), torch.from_numpy(label).float()

    def __len__(self):
        if self.length > 0:
            return self.length
        else:
            return len(self.data)
        #return 10
