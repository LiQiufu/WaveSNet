import argparse
import os, torch
import numpy as np
from tqdm import tqdm
from datetime import datetime

from mypath import Path, my_mkdir
from dataloaders import make_data_loader
from DeepLabV3Plus.sync_batchnorm.replicate import patch_replication_callback
from DeepLabV3Plus.deeplab import DeepLab as DeepLabV3P
from WDeepLabV3Plus.deeplab import DeepLab as WDeepLabV3P
from SegNet_UNet.SegNet import segnet_vgg16_bn as SegNet
from SegNet_UNet.SegNet import wsegnet_vgg16_bn as WSegNet
from SegNet_UNet.U_Net import unet_vgg16_bn as UNet
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.metrics import Evaluator
from utils.printer import Printer

class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        #self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.printer = args.printer
        
        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass, self.class_names = make_data_loader(args, **kwargs)

        # Define network
        self.model = self.get_net()
        if args.net in {'deeplabv3p', 'wdeeplabv3p', 'wsegnet', 'segnet', 'unet'}:
            train_params = [{'params': self.model.get_1x_lr_params(), 'lr': args.lr},
                            {'params': self.model.get_10x_lr_params(), 'lr': args.lr * 10}]
        elif args.net in {'segnet', 'waveunet', 'unet', 'waveunet_v2'}:
            weight_p, bias_p = [], []
            for name, p in self.model.named_parameters():
                if 'bias' in name:
                    bias_p.append(p)
                else:
                    weight_p.append(p)
            train_params = [{'params': weight_p, 'weight_decay': args.weight_decay, 'lr': args.lr},
                            {'params':  bias_p, 'weight_decay': 0, 'lr': args.lr}]
        else:
            train_params = None
            assert args.net in {'deeplabv3p', 'wdeeplabv3p', 'wsegnet', 'segnet', 'unet'}

        optimizer = torch.optim.SGD(train_params, momentum = args.momentum, nesterov = args.nesterov)
        self.optimizer = optimizer
        # Define Optimizer

        # Define Criterion
        # whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset+'_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        #self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.criterion = SegmentationLosses(weight = weight, cuda = args.cuda, batch_average = self.args.batch_average).build_loss(mode = args.loss_type)
        
        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        self.args.printer.pprint('Using {} LR Scheduler!, initialization lr = {}'.format(args.lr_scheduler, args.lr))
        if self.args.net.startswith('deeplab'):
            self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(self.train_loader))
        else:
            self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(self.train_loader), net = self.args.net)

        for key, value in self.args.__dict__.items():
            if not key.startswith('_'):
                self.printer.pprint('{} ==> {}'.format(key.rjust(24), value))

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu, output_device = args.out_gpu)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if args.dataset in ['pascal', 'cityscapes']:
                #self.load_pretrained_model()
            #elif args.dataset == 'cityscapes':
                self.load_pretrained_model_cityscape()

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    def get_net(self):
        model = None
        if self.args.net == 'deeplabv3p':
            model = DeepLabV3P(num_classes = self.nclass,
                              backbone = self.args.backbone,
                              output_stride = self.args.out_stride,
                              sync_bn = self.args.sync_bn,
                              freeze_bn = self.args.freeze_bn,
                              p_dropout = self.args.p_dropout)
        elif self.args.net == 'wdeeplabv3p':
            model = WDeepLabV3P(num_classes = self.nclass,
                                 backbone = self.args.backbone,
                                 output_stride = self.args.out_stride,
                                 sync_bn = self.args.sync_bn,
                                 freeze_bn = self.args.freeze_bn,
                                 wavename = self.args.wn,
                                 p_dropout = self.args.p_dropout)
        elif self.args.net == 'segnet':
            model = SegNet(num_classes = self.nclass, wavename = self.args.wn)
        elif self.args.net == 'unet':
            model = UNet(num_classes = self.nclass, wavename = self.args.wn)
        elif self.args.net == 'wsegnet':
            model = WSegNet(num_classes = self.nclass, wavename = self.args.wn)
        return model

    def load_pretrained_model(self):
        if not os.path.isfile(self.args.resume):
            raise RuntimeError("=> no checkpoint found at '{}'".format(self.args.resume))
        checkpoint = torch.load(self.args.resume, map_location = self.args.gpu_map)
        try:
            self.args.start_epoch = checkpoint['epoch']
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            pre_model = checkpoint['state_dict']
        except:
            self.printer.pprint('What happened ?!')
            self.args.start_epoch = 0
            if self.args.net in {'deeplabv3p', 'wdeeplabv3p', 'wsegnet', 'segnet', 'unet'}:
                pre_model = checkpoint
            elif self.args.net in {'waveunet','waveunet_v2'}:
                pre_model = checkpoint['state_dict']
        model_dict = self.model.state_dict()
        self.printer.pprint("=> loaded checkpoint '{}' (epoch {})".format(self.args.resume, self.args.start_epoch))
        for key in model_dict:
            self.printer.pprint('AAAA - key in model --> {}'.format(key))
        for key in pre_model:
            self.printer.pprint('BBBB - key in pre_model --> {}'.format(key))
        if self.args.net in {'deeplabv3p', 'wdeeplabv3p'}:
            pre_layers = [('module.' + k, v) for k, v in pre_model.items() if 'module.' + k in model_dict]
            for key in pre_layers:
                self.printer.pprint('CCCC - key in pre_model --> {}'.format(key[0]))
            model_dict.update(pre_layers)
            self.model.load_state_dict(model_dict)
        elif self.args.net in {'segnet', 'unet'}:
            pre_layers = [('module.' + k, v) for k, v in pre_model.items() if 'module.' + k in model_dict]
            for key in pre_layers:
                self.printer.pprint('CCCC - key in pre_model --> {}'.format(key[0]))
            model_dict.update(pre_layers)
            self.model.load_state_dict(model_dict)
        elif self.args.net in {'wsegnet'}:
            pre_layers = [('module.features.' + k[16:], v) for k, v in pre_model.items() if 'module.features.' + k[16:] in model_dict]
            for key in pre_layers:
                self.printer.pprint('CCCC - key in pre_model --> {}'.format(key[0]))
            model_dict.update(pre_layers)
            self.model.load_state_dict(model_dict)
        elif self.args.net == 'wdeeplabv3p':
            pre_layers = [('module.backbone.' + k[7:], v) for k, v in pre_model.items() if
                          'module.backbone.' + k[7:] in model_dict and (
                                      v.shape == model_dict['module.backbone.' + k[7:]].shape)]
            for key in pre_layers:
                self.printer.pprint('CCCC - key in pre_model --> {}'.format(key[0]))
            model_dict.update(pre_layers)
            self.model.load_state_dict(model_dict)


    def load_pretrained_model_cityscape(self):
        if not os.path.isfile(self.args.resume):
            raise RuntimeError("=> no checkpoint found at '{}'".format(self.args.resume))
        checkpoint = torch.load(self.args.resume, map_location = self.args.gpu_map)
        try:
            self.args.start_epoch = 0
            pre_model = checkpoint['state_dict']
        except:
            self.printer.pprint('What happened ?!')
            self.args.start_epoch = 0
            pre_model = checkpoint
        self.printer.pprint("=> loaded checkpoint '{}' (epoch {})".format(self.args.resume, self.args.start_epoch))
        if self.args.net == 'deeplabv3p' or 'wdeeplabv3p_per':
            model_dict = self.model.state_dict()
            for key in model_dict:
                self.printer.pprint('AAAA - key in model --> {}'.format(key))
            for key in pre_model:
                self.printer.pprint('BBBB - key in pre_model --> {}'.format(key))
            pre_layers = [('module.backbone.'+k, v) for k,v in pre_model.items() if 'module.backbone.'+k in model_dict and model_dict['module.backbone.'+k].shape == pre_model[k].shape]
            model_dict.update(pre_layers)
            self.model.load_state_dict(model_dict)

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        num_img_tr = len(self.train_loader)
        time_epoch_begin = datetime.now()
        for i, sample in enumerate(self.train_loader):
            time_iter_begin = datetime.now()
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            time_iter_end = datetime.now()
            time_iter = time_iter_end - time_iter_begin
            time_iter_during = time_iter_end - self.args.time_begin
            if i % 10 == 0:
                self.printer.pprint('train: epoch = {:3d} / {:3d}, '
                                    'iter = {:4d} / {:5d}, '
                                    'loss = {:.3f} / {:.3f}, '
                                    'time = {} / {}, '
                                    'lr = {:.6f}'.format(epoch, self.args.epochs, i, num_img_tr,
                                                         loss.item(), train_loss / (i+1),
                                                         time_iter, time_iter_during, self.optimizer.param_groups[0]['lr']))
        self.printer.pprint('------------ Train_total_loss = {}, epoch = {}, Time = {}'.format(train_loss, epoch, datetime.now() - time_epoch_begin))
        self.printer.pprint(' ')
        if epoch % 10 == 0:
            filename = os.path.join(self.args.weight_root, 'epoch_{}'.format(epoch) + '.pth.tar')
            torch.save({'epoch': epoch + 1,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'best_pred': self.best_pred,}, filename)

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        num_img_val = len(self.val_loader)
        test_loss = 0.0
        time_epoch_begin = datetime.now()
        for i, sample in enumerate(self.val_loader):
            time_iter_begin = datetime.now()
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            #test_loss += loss.item()
            test_loss += loss
            _, pred = output.topk(1, dim = 1)
            pred = pred.squeeze(dim = 1)
            pred = pred.cpu().numpy()
            target = target.cpu().numpy()
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)
            time_iter_end = datetime.now()
            time_iter = time_iter_end - time_iter_begin
            time_iter_during = time_iter_end - self.args.time_begin
            self.printer.pprint('validation: epoch = {:3d} / {:3d}, '
                                'iter = {:4d} / {:5d}, '
                                'loss = {:.3f} / {:.3f}, '
                                'time = {} / {}'.format(epoch, self.args.epochs,
                                                        i, num_img_val,
                                                        loss.item(), test_loss / (i + 1),
                                                        time_iter, time_iter_during))

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.printer.pprint('Validation, epoch = {}, Time = {}'.format(epoch, datetime.now() - time_epoch_begin))
        self.printer.pprint('------------ Total_loss = {}'.format(test_loss))
        self.printer.pprint("------------ Acc: {:.4f}, mIoU: {:.4f}, fwIoU: {:.4f}".format(Acc, mIoU, FWIoU))
        self.printer.pprint('------------ Acc_class = {}'.format(Acc_class))
        Object_names = '\t'.join(self.class_names)
        Object_IoU = '\t'.join(['{:0.3f}'.format(IoU * 100) for IoU in self.evaluator.IoU_class])
        self.printer.pprint('------------ ' + Object_names)
        self.printer.pprint('------------ ' + Object_IoU)

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

def main():
    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument('--net', type = str, default = 'deeplab',
                        choices = ['deeplabv3p', 'wdeeplabv3p', 'wsegnet', 'segnet', 'unet'])
    parser.add_argument('--p_dropout', type=str, default='0.5, 0.1',
                        help='use which gpu to train, must be a \
                        comma-separated list of floats only')
    parser.add_argument('--parameter_per', type=str, default='0.5, 0.75',
                        help='use which gpu to train, must be a \
                        comma-separated list of floats only')
    parser.add_argument('--wn', type = str, default = 'none',
                        choices = ['none', 'haar', 'bior2.2', 'bior3.3', 'bior4.4', 'bior5.5', 'db2'])
    parser.add_argument('--backbone', type=str, default='resnet101',
                        choices=['resnet50', 'resnet101', 'vgg16bn'],
                        help='backbone name (default: resnet101)')
    parser.add_argument('--out_stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'coco', 'cityscapes'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--use_sbd', action='store_true', default=True,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--batch_average', action='store_true', default=True,
                        help='whether to average the loss of the batch (default: True)')
    parser.add_argument('--workers', type=int, default=8,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base_size', type=int, default=768,
                        help='base image size')
    parser.add_argument('--crop_size', type=int, default=768,
                        help='crop image size')
    parser.add_argument('--sync_bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze_bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss_type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch_size', type=int, default=8,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test_batch_size', type=int, default=4,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use_balanced_weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr', '--learning_ratio', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr_scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no_cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', '--fine_tuning', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval_interval', type=int, default=5,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no_val', action='store_true', default=False,
                        help='skip validation during training')

    args = parser.parse_args()  #在执行这条命令之前，所有命令行参数都给不会生效
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
            args.gpu_map = {}
            for index, gpu_id in enumerate(args.gpu_ids):
                args.gpu_map['cuda:{}'.format(gpu_id)] = 'cuda:{}'.format(index)
            args.gpu = list([i for i in range(len(args.gpu_ids))])
            args.out_gpu = 0
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    args.p_dropout = tuple(float(s) for s in args.p_dropout.split(','))
    args.parameter_per = tuple(float(s) for s in args.parameter_per.split(','))
    # (0.5, 0.25) for voc in deeplabv3+
    # (0.5, 0.25) or (0.5, 0.1) for voc in deeplabv3+wavelets
    # (0.25, 0.1) for cityscape in deeplabv3+
    # (0, 0.1) for cityscape in deeplabv3+haar

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'cityscapes': 200,
            'pascal': 50,
        }
        args.epochs = epoches[args.dataset.lower()]

    args.weight_class = None
    if args.net == 'segnet' and args.dataset == 'pascal':
        args.weight_class = torch.tensor([0.1369, 1.4151, 3.0612, 1.5385, 1.7606,
                                          1.7969, 0.7062, 1.0345, 0.5535, 1.699,
                                          1.6026, 1.0081, 0.8621, 1.5306, 0.9167,
                                          0.4124, 1.875, 1.506, 1.1029, 0.6452, 1.4286]).cuda()

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'coco': 0.1,
            'cityscapes': 0.01,
            'pascal': 0.007,
        }
        #args.lr = lrs[args.dataset.lower()] / len(args.gpu_ids)
        args.lr = lrs[args.dataset.lower()]
    if args.checkname is None:
        args.checkname = args.net + '_' + str(args.backbone) + '_' + args.wn
    args.info_file = os.path.join('.', 'info', args.dataset, args.net, args.checkname + '_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S.info'))
    args.weight_root = os.path.join('.', 'weight', args.dataset, args.net, args.checkname + '_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    my_mkdir(args.info_file, mode = 'file')
    my_mkdir(args.weight_root, mode = 'path')
    args.printer = Printer(args.info_file)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    trainer.args.printer.pprint('Starting Epoch: {}'.format(trainer.args.start_epoch))
    trainer.args.printer.pprint('Total Epoches: {}'.format(trainer.args.epochs))
    args.time_begin = datetime.now()
    trainer.validation(epoch = trainer.args.start_epoch)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and (epoch % args.eval_interval == (args.eval_interval - 1) or epoch >= (trainer.args.epochs - 10)):
            trainer.validation(epoch)

    #trainer.writer.close()

if __name__ == "__main__":
   main()
