"""
创建根路径
"""

import os

def my_mkdir(file_name):
    root, name = os.path.split(file_name)
    if not os.path.isdir(root):
        os.makedirs(root)

if __name__ == '__main__':
    file_name = '/a/b/c/d/e'
    root, name = os.path.split(file_name)
    print(root, name)