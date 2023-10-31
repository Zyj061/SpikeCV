# -*- encoding: utf-8 -*-

# here put the import lib
import os

def seek_file(search_dirs, filename):
    search_dir_split = split_path_into_pieces(search_dirs)
    dir_level = len(search_dir_split)

    for i_dir in range(0, dir_level):
        if i_dir > 0:
            search_dir_split.pop(-1)
        # search_dir = os.path.join(str(search_dir_split[0:-i_dir]))
        search_dir = os.path.join(*search_dir_split)
        for root, dirs, files in os.walk(search_dir):
            if filename in files:
                print('{0}/{1}'.format(root, filename))
                filepath = os.path.join(root, filename)
                return filepath

def split_path_into_pieces(path: str):
    pieces = []
    if path[-1] == '/':
        path = path[0:-1]

    while True:
        splits = os.path.split(path)
        if splits[0] == '':
            pieces.insert(0, splits[-1])
            break
        if splits[-1] == '':
            pieces.insert(0, splits[0])
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