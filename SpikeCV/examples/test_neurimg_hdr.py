import sys
sys.path.append("..")

import os
import numpy as np
from spkProc.reconstruction.NeurImg_HDR.options.test_options import TestOptions
from spkProc.reconstruction.NeurImg_HDR.data import create_dataset
from spkProc.reconstruction.NeurImg_HDR.models import create_model
from spkProc.reconstruction.NeurImg_HDR.util.util import *

def save_image(visuals, im_h_pad, im_w_pad, savedir, file_path):
    (filepath, filename) = os.path.split(file_path[0])
    (name, extension) = os.path.splitext(filename)
    for label, image in visuals.items():
        height_pad = im_h_pad * 2
        width_pad = im_w_pad * 2
        image = delPadding(image, height_pad, -height_pad, width_pad, -width_pad)
        if(label.endswith('output_hdr_rgb')):
            image_numpy = tensor2hdr(image)
            exr_path = os.path.join(savedir, '%s_%s.exr' % (name, label))
            writeEXR(image_numpy, exr_path)

def white_balance(save_dir):
    orig_dir = "%s/exr_results/" % (save_dir)
    files = sorted(os.listdir(orig_dir))
    for i, file in enumerate(files):
        hdr_img = readEXR(os.path.join(orig_dir, file))[:,:,::-1]
        if hdr_img.min() < 0:
            hdr_img = hdr_img - hdr_img.min()
        # white balabce
        if i == 0:
            r_max = hdr_img[:,:,0].max()
            g_max = hdr_img[:,:,1].max()
            b_max = hdr_img[:,:,2].max()
            mat = [[g_max/r_max, 0, 0], [0, 1.0, 0], [0,0,g_max/b_max]]
        hdr_img_wb = whiteBalance_mat(hdr_img, mat)
        writeEXR(hdr_img_wb[:,:,::-1], os.path.join(orig_dir, file))
        

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    
    dataset_dirs = str.split(opt.dataroot, '/')
    save_dir = os.path.join(opt.results_dir, opt.name, dataset_dirs[-1])
    img_dir = os.path.join(save_dir, 'exr_results/')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    if opt.eval:
        model.eval()
        
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        if opt.netColor == 'image':
            model.set_input(data)  # test for images 
        elif opt.netColor == 'video':
            model.set_testvideo_input(data) # test for video frames
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        if i % 5 == 0: 
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_image(visuals, data['im_h_pad'], data['im_w_pad'], img_dir, data['paths'])
    if opt.netColor == 'video':
        white_balance(save_dir)