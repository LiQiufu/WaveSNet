"""
这个脚本用于计算深度网络模型所需要的可训练参数个数
"""

def counter(net):
    params = list(net.parameters())
    k = 0
    for index, i in enumerate(params):
        l = 1
        print("{} -- 该层的结构：".format(index) + str(list(i.size())))
        for j in i.size():
            l *= j
        k = k + l
        print("{} -- 该层参数和： {} / {}".format(index, l, k))
    print("总参数数量和：{} ==> {} K ==> {} M".format(k, k / 1000, k / 1000 / 1000))

from WDeepLabV3Plus.deeplab import DeepLab
#from modeling.deeplab import DeepLab

if __name__ == '__main__':
    net = DeepLab(num_classes = 21)
    counter(net)