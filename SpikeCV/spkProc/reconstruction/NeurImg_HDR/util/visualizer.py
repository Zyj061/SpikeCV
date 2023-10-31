import numpy as np
import os
import time
import cv2
from . import util

class Visualizer():
    def __init__(self, opt):
        self.opt = opt  # cache the option
        self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
        self.img_dir = os.path.join(self.web_dir, 'images')
        print('create web directory %s...' % self.web_dir)
        util.mkdirs([self.web_dir, self.img_dir])
        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)



    # Save training samples to disk
    def save_image_to_disk(self, visuals, iteration, epoch):
        for label, image in visuals.items():
            if(label[-1] == 'B'):
                image_numpy = util.tensor2hdr(image)
                tonemapped = util.hdr2tonemapped(image_numpy)
                # hdr_path = os.path.join(self.img_dir, 'epoch%.3d_iter%.4d_%s.exr' % (epoch, iteration, label))
                tonemap_path = os.path.join(self.img_dir, 'epoch%.3d_iter%.4d_%s_tmp.jpg' % (epoch, iteration, label))
#                cv2.imwrite(hdr_path, image_numpy[:,:,::-1])
                # util.writeEXR(image_numpy, hdr_path)
                cv2.imwrite(tonemap_path, tonemapped[:,:,::-1])
            elif(label[-3:] == 'B_Y'):
                image_numpy = util.tensor2hdr(image)
                tonemapped = util.hdr2tonemapped(image_numpy)
                tonemap_path = os.path.join(self.img_dir, 'epoch%.3d_iter%.4d_%s_tmp.jpg' % (epoch, iteration, label))
                cv2.imwrite(tonemap_path, tonemapped[:,:,::-1])
            elif(label[-3:] == 'A_Y'):
                image_numpy = image.cpu().float().numpy()
                AYimg = np.transpose(image_numpy[0], (1, 2, 0))
                AYimg = (AYimg + 1.0) / 2.0
                AYimg = (AYimg-AYimg.min()) / (AYimg.max()-AYimg.min())
                AYimg = AYimg ** (1/2.2)
                AYimg = (AYimg*255).astype(np.uint8)
                img_path = os.path.join(self.img_dir, 'epoch%.3d_iter%.4d_%s.jpg' % (epoch, iteration, label))
                cv2.imwrite(img_path, AYimg)
            elif(label[-1] == 'A'):
                image_numpy = util.tensor2im(image)
                img_path = os.path.join(self.img_dir, 'epoch%.3d_iter%.4d_%s.jpg' % (epoch, iteration, label))
                cv2.imwrite(img_path, image_numpy[:,:,::-1])
            elif(label[:6] == 'spikes'):
                image_numpy = image.detach().cpu().float().numpy()
                spikes_img = np.transpose(image_numpy[0], (1, 2, 0))
                spikes_img = (spikes_img + 1.0) / 2.0

                spikes_img = (spikes_img-spikes_img.min()) / (spikes_img.max()-spikes_img.min())

                spikes_img = (spikes_img*255).astype(np.uint8)
                img_path = os.path.join(self.img_dir, 'epoch%.3d_iter%.4d_%s.jpg' % (epoch, iteration, label))
                cv2.imwrite(img_path, spikes_img)
            else:
                att_map = util.tensor2im(image)
                att_map = cv2.applyColorMap(att_map, cv2.COLORMAP_HOT)
                img_path  = os.path.join(self.img_dir, 'epoch%.3d_iter%.4d_%s.jpg' % (epoch, iteration, label))
                cv2.imwrite(img_path, att_map)
    
    def newvideo_save_image(self, visuals, iteration, epoch):
        dest_dir = os.path.join(self.img_dir, 'epoch%.3d_iter%.4d' % (epoch, iteration))
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        for label, image in visuals.items():
            if(label.endswith('Bs')):
                for i in range(len(image)):
                    image_numpy = util.tensor2hdr(image[i])
                    tonemapped = util.hdr2tonemapped(image_numpy)
                    tonemap_path = os.path.join(self.img_dir, 'epoch%.3d_iter%.4d' % (epoch, iteration), '%04d_%s_tmp.jpg' % (i, label))
                    cv2.imwrite(tonemap_path, tonemapped[:,:,::-1])
            elif(label.endswith('As')):
                for i in range(len(image)):
                    image_numpy = util.tensor2im(image[i])
                    img_path = os.path.join(self.img_dir, 'epoch%.3d_iter%.4d' % (epoch, iteration), '%04d_%s.jpg' % (i, label))
                    cv2.imwrite(img_path, image_numpy[:,:,::-1])
            elif(label.startswith('spikes')):
                for i in range(len(image)):
                    image_numpy = image[i].cpu().float().numpy()
                    spikes_img = np.transpose(image_numpy[0], (1, 2, 0))
                    spikes_img = (spikes_img + 1.0) / 2.0
                    spikes_img = (spikes_img-spikes_img.min()) / (spikes_img.max()-spikes_img.min())
                    spikes_img = (spikes_img*255).astype(np.uint8)
                    img_path = os.path.join(self.img_dir, 'epoch%.3d_iter%.4d' % (epoch, iteration), '%04d_%s.jpg' % (i, label))
                    cv2.imwrite(img_path, spikes_img)
