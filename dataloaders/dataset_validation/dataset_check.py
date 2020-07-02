import os
import cv2
import numpy as np
import copy
#from dataloaders.datasets.pascal import VOCSegmentation

class Checker():
    def __init__(self, image_root, label_root, saved_root, source, class_num = 21):
        self.image_root = image_root
        self.label_root = label_root
        self.saved_root = saved_root
        self.source = source
        self.class_num = class_num

        self.name_list = self.name_list_()
        self.chect()

    def chect(self):
        assert os.path.isdir(self.saved_root)
        for i in range(self.class_num):
            if not os.path.isdir(os.path.join(self.saved_root, str(i))):
                os.mkdir(os.path.join(self.saved_root, str(i)))

    def name_list_(self):
        name_list = open(self.source).readlines()
        name_list = [line.strip() for line in name_list if not line.startswith('#')]
        return name_list

    def get_item(self, index):
        image_name = os.path.join(self.image_root, self.name_list[index] + '.jpg')
        label_name = os.path.join(self.label_root, self.name_list[index] + '.png')
        image = cv2.imread(image_name, 1)
        label = cv2.imread(label_name, 1)
        return image, label

    def run(self):
        for index, name in enumerate(self.name_list):
            if index % 20 == 0:
                print('processing {}, {}/{}'.format(os.path.join(self.image_root, name), index, len(self.name_list)))
            image, label = self.get_item(index)
            for i in range(self.class_num):
                if np.sum(label[label == i]) == 0:
                    continue
                image_sub = copy.copy(image)
                image_sub[label != i] = 0
                image_sub_name = os.path.join(self.saved_root, str(i), name + '.jpg')
                if index % 20 == 0:
                    print('----- {}'.format(image_sub_name))
                cv2.imwrite(image_sub_name, image_sub)


if __name__ == '__main__':
    image_root = '/raid/liqiufu/DATA/VOC/JPEGImages'
    label_root = '/raid/liqiufu/DATA/VOC/SegmentationClass'
    source = '/raid/liqiufu/DATA/VOC/train.txt'
    saved_root = '/raid/liqiufu/DATA/VOC/Check/train'
    checker = Checker(image_root = image_root, label_root = label_root, saved_root = saved_root, source = source, class_num = 21)
    checker.run()