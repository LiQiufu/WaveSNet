import os

class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return '/raid/liqiufu/DATA/VOC'  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return '/path/to/datasets/benchmark_RELEASE'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return '/raid/liqiufu/DATA/Cityscapes'     # foler that contains leftImg8bit/
        elif dataset.lower() == 'coco':
            return '/raid/liqiufu/DATA/COCO/2017'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError


def my_mkdir(file_name, mode = 'file'):
    """
    创建根路径
    :param mode: 'path', 'file'
    """
    if mode == 'path':
        if not os.path.isdir(file_name):
            os.makedirs(file_name)
            return
    elif mode == 'file':
        root, name = os.path.split(file_name)
        if not os.path.isdir(root):
            os.makedirs(root)
        return
    else:
        assert mode in ['path', 'file']

if __name__ == '__main__':
    print(Path.db_root_dir('coco'))