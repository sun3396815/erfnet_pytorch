import cv2
import numpy as np
import torch
import os
import importlib

from PIL import Image
from argparse import ArgumentParser

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage

from dataset import cityscapes, AutoVpDataset
from erfnet import ERFNet
from transform import Relabel, ToLabel, Colorize

import visdom

NUM_CHANNELS = 3
NUM_CLASSES = 1

def main(args):

    weightspath = "/media/sunhuibo/data/Work/src/erfnet_pytorch/ckpt/last.pth.tar"

    #Import ERFNet model from the folder
    #Net = importlib.import_module(modelpath.replace("/", "."), "ERFNet")
    model = ERFNet(NUM_CLASSES)
  
    model = torch.nn.DataParallel(model)
    if (not args.cpu):
        model = model.cuda()

    #model.load_state_dict(torch.load(args.state))
    #model.load_state_dict(torch.load(weightspath)) #not working if missing key

    def load_my_state_dict(m, state_dict):  #custom function to load model when not all dict elements
        own_state = m.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            own_state[name].copy_(param)
        return m

    model = load_my_state_dict(model, torch.load(weightspath))
    print("Model and weights LOADED successfully")

    model.eval()

    if(not os.path.exists(args.datadir)):
        print("Error: datadir could not be loaded")

    # TODO: remove param root_dir
    loader = DataLoader(AutoVpDataset(args.datadir, "/media/sunhuibo/data/Work/data/3rdparty/CULaneSlim/"),
        num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    # For visualizer:
    # must launch in other window "python3.6 -m visdom.server -port 8097"
    # and access localhost:8097 to see it
    if (args.visualize):
        vis = visdom.Visdom()

    for step, (images, labels) in enumerate(loader):
        if (not args.cpu):
            images = images.cuda()
            labels = labels.cuda()

        cv2.imshow("gt heat map", np.array(labels.cpu()[0][0]) * 255)

        inputs = Variable(images)
        #targets = Variable(labels)
        with torch.no_grad():
            outputs = model(inputs)

        _, _, th, tw = images.shape
        _, _, oh, ow = outputs.shape
        start_h = (oh - th) // 2
        start_w = (ow - tw) // 2

        croped_outputs = outputs[:, :, start_h:th + start_h, start_w:tw + start_w]

        reshaped_outputs = croped_outputs.reshape((croped_outputs.shape[0], -1))
        reshaped_targets = labels.reshape((labels.shape[0], -1))
        _, peak_out_index = torch.max(reshaped_outputs, 1)
        _, peak_tar_index = torch.max(reshaped_targets, 1)
        ox, oy, tx, ty = peak_out_index % tw, peak_out_index // tw, peak_tar_index % tw, peak_tar_index // tw
        norm_dist = ((ox - tx) ** 2 + (ty - oy) ** 2) ** 0.5 / (tw ** 2 + th ** 2) ** 0.5
        print(f'norm dist: {str(norm_dist)}')

        original_image = np.array((images[0].cpu() + 0.5) * 255, dtype=np.uint8)
        original_image = original_image.transpose((1, 2, 0)).copy()  # c,h,w to h,w,c
        cv2.drawMarker(original_image, np.int32((ox, oy)), (0, 255, 0), thickness=2)
        cv2.drawMarker(original_image, np.int32((tx, ty)), (0, 0, 255), thickness=2)

        cv2.imshow("original image", original_image)
        cv2.waitKey()

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--state')

    parser.add_argument('--loadDir', default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")  #can be val, test, train, demoSequence

    parser.add_argument('--datadir', default="/media/sunhuibo/data/Work/data/3rdparty/CULaneSlim/test_heatmap.txt")
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true', default=False)

    parser.add_argument('--visualize', action='store_true')
    main(parser.parse_args())
