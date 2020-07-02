import numpy as np
import os, cv2
from dataloaders.utils import decode_segmap


def decode_segmentation(root, save_root):
    image_names = os.listdir(root)
    image_names.sort()
    for image_name in image_names:
        print('processing {} ...'.format(image_name))
        label = cv2.imread(os.path.join(root, image_name),0)
        print(label.shape)
        label_color = decode_segmap(label, dataset = 'pascal', plot = False)
        #label_color.save(os.path.join(save_root, image_name))
        cv2.imwrite(os.path.join(save_root, image_name), np.array(label_color * 255, np.uint8))


if __name__ == '__main__':
    root = '/home/liqiufu/Desktop/my_paper/VOC_result/label_pre_wdeeplab'
    save_root = '/home/liqiufu/Desktop/my_paper/VOC_result/label_pre_wdeeplab_color'
    decode_segmentation(root = root, save_root = save_root)
    label = cv2.imread(os.path.join(save_root, '2007_000033.png'))
    cv2.imshow('image', label)
    cv2.waitKey(0)