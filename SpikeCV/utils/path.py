# -*- encoding: utf-8 -*-
'''
@File    :   path.py
@Time    :   2022/07/26 10:29:36
@Author  :   Jiyuan Zhang 
@Version :   0.0
@Contact :   jyzhang@stu.pku.edu.cn
'''

# here put the import lib
import os


def split_path_into_pieces(path: str):
    pieces = []
    
    while True:
        splits = os.path.split(path)
        if splits[0] == '':
            pieces.insert(0, splits[-1])
            break
        pieces.insert(0, splits[-1])
        path = splits[0]
    
    return pieces

def replace_identifier(path: list, src: str, dst: str):
    new_path = []
    for piece in path:
        added_piece = piece
        if piece == src:
            added_piece = dst
        new_path.append(added_piece)
    
    return new_path