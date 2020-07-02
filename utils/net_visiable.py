"""
这个脚本对构建的网络结构进行可视化，这个脚本是从网上获得
"""

from graphviz import Digraph
import torch
from torch.autograd import Variable


def make_dot(vars, params = None):
    """ Produces Graphviz representation of PyTorch autograd graph
    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    if params is not None:
        assert isinstance(params.values()[0], Variable)
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '('+(', ').join(['%d' % v for v in size])+')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                name = param_map[id(u)] if params is not None else ''
                node_name = '%s\n %s' % (name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)
    for var in vars:
        try:
            add_nodes(var.grad_fn)
        except:
            print('got {}'.format(var))
    return dot

from torchvision.models import segmentation
from modeling.deeplab import *
if __name__ == '__main__':
    from datetime import datetime

    model_ = DeepLab(num_classes = 21,
                    backbone = 'resnet',
                    output_stride = 16,
                    sync_bn = True,
                    freeze_bn = True)
    model__ = segmentation.deeplabv3_resnet50(pretrained = False, progress = True, pretrained_backbone = False)
    #model.train()
    model = torch.hub.load('pytorch/vision:v0.4.2', 'deeplabv3_resnet101', pretrained = False)
    t0 = datetime.now()
    x = torch.randn(1,3,512,512)
    #model = model()
    output0 = model(x)
    for key in output0:
        print(key)
    t1 = datetime.now()
    print('taking {} secondes'.format(t1 - t0))
    g = make_dot((output0))
    g.view()