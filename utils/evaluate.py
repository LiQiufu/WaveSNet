#from logger import *
#from models.deeplabv3plus import Deeplab_v3plus
#from cityscapes import CityScapes
#from configs import config_factory
from dataloaders.datasets import cityscapes, coco, combine_dbs, pascal, sbd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch.distributed as dist
import cv2

import os
import sys
import os.path as osp
import logging
import time
import numpy as np
from tqdm import tqdm
import argparse
from DeepLabV3Plus.deeplab import DeepLab as DeepLabV3P
from WDeepLabV3Plus.deeplab import DeepLab as WDeepLabV3P

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type = str, default = 'deeplab',
                        choices = ['deeplabv3p', 'wdeeplabv3p'])
    parser.add_argument('--wn', type = str, default = 'none',
                        choices = ['none', 'haar', 'bior2.2', 'bior3.3', 'bior4.4', 'bior5.5', 'db2'])
    parser.add_argument('--p_dropout', type=str, default='0.5, 0.1',
                        help='use which gpu to train, must be a \
                        comma-separated list of floats only')
    parser.add_argument('--backbone', type = str, default = 'resnet101',
                        choices = ['resnet50', 'resnet101', 'vgg16bn'],
                        help = 'backbone name (default: resnet101)')
    parser.add_argument('--out_stride', type = int, default = 16,
                        help = 'network output stride (default: 8)')
    parser.add_argument('--dataset', type = str, default = 'pascal',
                        choices = ['pascal', 'coco', 'cityscapes'],
                        help = 'dataset name (default: pascal)')
    parser.add_argument('--datamode', type = str, default = 'val',
                        choices = ['val', 'test'],
                        help = 'dataset mode (default: val)')
    parser.add_argument('--workers', type = int, default = 8,
                        metavar = 'N', help = 'dataloader threads')
    parser.add_argument('--batch_size', type=int, default=8,
                        metavar='N', help='input batch size for training (default: auto)')
    parser.add_argument('--test_batch_size', type=int, default=4,
                        metavar='N', help='input batch size for testing (default: auto)')
    parser.add_argument('--base_size', type = int, default = 768,
                        help = 'base image size')
    parser.add_argument('--crop_size', type = int, default = 768,
                        help = 'crop image size')
    parser.add_argument('--multi_scale', action='store_true', default = False,
                        help='whether to use multi scale input (default: False)')
    parser.add_argument('--eval_flip', action='store_true', default = False,
                        help='whether to flip the input (default: False)')
    parser.add_argument('-gpu', '--gpu_ids', type = str, default = '0',
                        help = 'use which gpu to train, must be a comma-separated list of integers only (default=0)')
    parser.add_argument('--resume', type = str, default = None,
                        help = 'put the path to resuming file if needed')

    return parser.parse_args()


class MscEval(object):
    def __init__(self, args):
        self.args = args
        self.get_gpu()
        self.get_dataloader()
        self.get_net()
        self.scales = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0) if self.args.multi_scale else (1.0,)

    def get_dataloader(self):
        if self.args.dataset == 'pascal':
            assert self.args.datamode == 'val'
            dataset = pascal.VOCSegmentation(self.args, split = 'val')
            self.n_classes = dataset.NUM_CLASSES
            self.name_classes = dataset.class_names
            self.dataloader = DataLoader(dataset, batch_size = self.args.test_batch_size, shuffle = False)
        elif self.args.dataset == 'cityscapes':
            dataset = cityscapes.CityscapesSegmentation(self.args, split = self.args.datamode)
            self.n_classes = dataset.NUM_CLASSES
            self.name_classes = dataset.class_names
            self.dataloader = DataLoader(dataset, batch_size = self.args.test_batch_size, shuffle = False)
        elif self.args.dataset == 'coco':
            dataset = coco.COCOSegmentation(self.args, split = self.args.datamode)
            self.n_classes = dataset.NUM_CLASSES
            self.name_classes = dataset.class_names
            self.dataloader =  DataLoader(dataset, batch_size = self.args.test_batch_size, shuffle = False)
        else:
            raise NotImplementedError('不适用于其他数据集')

    def get_net(self):
        self.args.p_dropout = tuple(float(s) for s in self.args.p_dropout.split(','))
        # (0.5, 0.25) for voc in deeplabv3+
        # (0.25, 0.1) for cityscape in deeplabv3+
        # (0.5, 0.25) for voc in deeplabv3+wavelets
        # (0, 0.1) for cityscape in deeplabv3+haar
        if self.args.net == 'deeplabv3p':
            self.net = DeepLabV3P(num_classes = self.n_classes, backbone = self.args.backbone, output_stride = self.args.out_stride,
                                 sync_bn = None, freeze_bn = False, p_dropout = self.args.p_dropout)
        elif self.args.net == 'wdeeplabv3p':
            self.net = WDeepLabV3P(num_classes = self.n_classes, backbone = self.args.backbone, output_stride = self.args.out_stride, wavename = self.args.wn,
                                    sync_bn = None, freeze_bn = False, p_dropout = self.args.p_dropout)
        else:
            raise NotImplementedError('当前指定网络没有实现')
        self.load_pre_mode()
        self.net = torch.nn.DataParallel(self.net.cuda(), device_ids = self.gpus, output_device = self.out_gpu)

    def load_pre_mode(self):
        if not os.path.isfile(self.args.resume):
            raise RuntimeError("=> no checkpoint found at '{}'".format(self.args.resume))
        checkpoint = torch.load(self.args.resume, map_location = self.gpu_map)
        checkpoint = checkpoint['state_dict']
        model_dict = self.net.state_dict()
        for key in model_dict:
            print('AAAA - key in model --> {}'.format(key))
        for key in checkpoint:
            print('BBBB - key in pre_model --> {}'.format(key))
        pre_layers = [(k, v) for k, v in checkpoint.items() if
                      k in model_dict and model_dict[k].shape == checkpoint[k].shape]
        model_dict.update(pre_layers)
        self.net.load_state_dict(model_dict)

    def get_gpu(self):
        self.gpu_ids = [int(s) for s in self.args.gpu_ids.split(',')]
        self.gpu_map = {}
        for index, gpu_id in enumerate(self.gpu_ids):
            self.gpu_map['cuda:{}'.format(gpu_id)] = 'cuda:{}'.format(index)
        self.gpus = list([i for i in range(len(self.gpu_ids))])
        self.out_gpu = 0
        print(self.gpu_map, self.gpus, self.out_gpu)

    def __call__(self, save = False, saved_root = None):
        ## evaluate
        self.net.eval()
        hist_size = (self.n_classes, self.n_classes)
        hist = np.zeros(hist_size, dtype=np.float32)
        for i, sample in enumerate(self.dataloader):
            imgs = sample['image']
            label = sample['label']
            factor_resize = (1,1)
            try:
                factor_resize = sample['factor']
            except:
                pass
            label = label.unsqueeze(1)
            N, _, H, W = label.size()
            probs = torch.zeros((N, self.n_classes, H, W))
            probs.requires_grad = False
            for sc in self.scales:
                new_hw = [int(H*sc), int(W*sc)]
                with torch.no_grad():
                    im = F.interpolate(imgs, new_hw, mode='bilinear', align_corners=True)
                    im = im.cuda()
                    out = self.net(im)
                    out = F.interpolate(out, (H, W), mode='bilinear', align_corners=True)
                    prob = F.softmax(out, 1)
                    probs += prob.cpu()
                    if self.args.eval_flip:
                        out = self.net(torch.flip(im, dims=(3,)))
                        out = torch.flip(out, dims=(3,))
                        out = F.interpolate(out, (H, W), mode='bilinear', align_corners=True)
                        prob = F.softmax(out, 1)
                        probs += prob.cpu()
                    del out, prob
            probs = probs.data.numpy()
            preds = np.argmax(probs, axis=1)
            hist_once = self.compute_hist(preds, label.data.numpy().squeeze(1))
            hist = hist + hist_once

            if save:
                if saved_root == None:
                    saved_root = './'
                image_path = os.path.join(saved_root, 'image')
                label_t_path = os.path.join(saved_root, 'label')
                label_p_path = os.path.join(saved_root, 'label_pre')
                build_path(image_path)
                build_path(label_t_path)
                build_path(label_p_path)
                image_names = sample['image_name']
                for index in range(N):
                    image_name = image_names[index]
                    image = imgs[index, :, :, :]
                    label_truth = label[index, :, :, :]
                    label_pre = preds[index, :, :]
                    label_pre = torch.tensor(label_pre)
                    label_pre = label_pre.unsqueeze(0)
                    if not factor_resize == (1,1):
                        factor = [factor_resize[0][index], factor_resize[1][index]]
                        image = F.interpolate(image.unsqueeze(0), (int(H / factor[0]), int(W / factor[1])), mode = 'bilinear', align_corners = True)
                        label_truth = F.interpolate(label_truth.unsqueeze(0), (int(H / factor[0]), int(W / factor[1])), mode = 'nearest')
                        label_pre = F.interpolate(label_pre.unsqueeze(0).float(), (int(H / factor[0]), int(W / factor[1])), mode = 'nearest')
                        image = image.squeeze()
                        image = np.array(image).astype(np.uint8).transpose((1, 2, 0))
                        label_truth = np.array(label_truth.squeeze().data.cpu(), dtype = np.uint8)
                        label_pre = np.array(label_pre.squeeze().data.cpu(), dtype = np.uint8)
                        image = image[:,:,-1::-1]
                        cv2.imwrite(os.path.join(image_path, image_name + '.png'), image)
                        cv2.imwrite(os.path.join(label_t_path, image_name + '.png'), label_truth)
                        cv2.imwrite(os.path.join(label_p_path, image_name + '.png'), label_pre)
            del imgs, label
            IOUs = np.diag(hist) / (np.sum(hist, axis = 0) + np.sum(hist, axis = 1) - np.diag(hist))
            print('{} / {} -- {}'.format(i, len(self.dataloader), np.mean(IOUs)))
        IOUs = np.diag(hist) / (np.sum(hist, axis=0) + np.sum(hist, axis=1) - np.diag(hist))
        mIOU = np.mean(IOUs)
        Object_names = '\t'.join(self.name_classes)
        Object_IoU = '\t'.join(['{:0.3f}'.format(IoU * 100) for IoU in IOUs])
        print('------------ ' + Object_names)
        print('------------ ' + Object_IoU)
        return mIOU

    def compute_hist(self, pred, lb):
        n_classes = self.n_classes
        keep = np.logical_not(lb == 255)
        merge = pred[keep] * n_classes + lb[keep]
        merge = np.array(merge, dtype = np.int64)
        hist = np.bincount(merge, minlength = n_classes**2)
        hist = hist.reshape((n_classes, n_classes))
        return hist


def evaluate(save = True, saved_root = './'):
    ## setup
    args = parse_args()
    evaluator = MscEval(args)
    mIOU = evaluator(save = save, saved_root = saved_root)
    print('mIOU is: {:.6f}'.format(mIOU))

def build_path(path):
    if not os.path.isdir(path):
        os.makedirs(path)

if __name__ == "__main__":
    evaluate()
