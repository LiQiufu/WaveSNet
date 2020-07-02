import os
import matplotlib.pyplot as plt
from PIL import Image

image_root = '/home/liqiufu/Desktop/my_paper/VOC_result/image'
label_root = '/home/liqiufu/Desktop/my_paper/VOC_result/label_color'
label_deeplab_root = '/home/liqiufu/Desktop/my_paper/VOC_result/label_pre_deeplab_color'
label_wdeelab_root = '/home/liqiufu/Desktop/my_paper/VOC_result/label_pre_wdeeplab_color'

def show_segmentation():
    image_names = os.listdir(image_root)
    image_names = ['2007_000663.png', '2007_000830.png', '2007_000925.png', '2007_001299.png',
                   '2007_001430.png', '2007_001763.png', '2007_002719.png', '2007_003106.png',
                   '2007_003506.png', '2007_004902.png', '2007_004189.png', '2007_006241.png',
                   '2008_004624.png',
                   '2009_000732.png', '2009_002165.png', '2009_003804.png', '2009_005302.png',
                   '2011_002178.png', '2011_002358.png']
    image_names.sort()
    for index, image_name in enumerate(image_names):
        image = Image.open(os.path.join(image_root, image_name))
        label = Image.open(os.path.join(label_root, image_name))
        label_deeplab = Image.open(os.path.join(label_deeplab_root, image_name))
        label_wdeeplab = Image.open(os.path.join(label_wdeelab_root, image_name))
        fig, axs = plt.subplots(2,2)
        axs[0,0].imshow(image)
        axs[0,1].imshow(label)
        axs[1,0].imshow(label_deeplab)
        axs[1,1].imshow(label_wdeeplab)
        axs[0,0].set_title('image')
        axs[0, 1].set_title('label')
        axs[1, 0].set_title('label_deeplab')
        axs[1, 1].set_title('label_wdeeplab')
        plt.title(image_name)
        if (index + 1) % 20 == 0 or (index + 1) == len(image_names):
            plt.show()

if __name__ == '__main__':
    show_segmentation()