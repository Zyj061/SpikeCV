import skimage.morphology as smor
from skimage.feature import peak_local_max
from skimage.morphology import erosion
from skimage.measure import label, regionprops
import numpy as np
import torch
from torchvision.transforms import Resize


# obtain 2D gaussian filter
def get_kernel(filter_size, sigma):

    assert (filter_size + 1) % 2 == 0, '2D filter size must be odd number!'
    g = np.zeros((filter_size, filter_size), dtype=np.float32)
    half_width = int((filter_size - 1) / 2)
    # center location

    xc = (filter_size + 1) / 2
    yc = (filter_size + 1) / 2
    for i in range(-half_width, half_width + 1, 1):
        for j in range(-half_width, half_width + 1, 1):
            x = int(xc + i)
            y = int(yc + j)
            g[y - 1, x - 1] = np.exp(- (i ** 2 + j ** 2) / 2 / sigma / sigma)

    g = (g - g.min()) / (g.max() - g.min())
    return g


# detect moving connected regions
class SaccadeInput:

    def __init__(self, spike_h, spike_w, box_size, device):

        self.spike_h = spike_h
        self.spike_w = spike_w
        self.device = device

        self.U = torch.zeros(self.spike_h, self.spike_w, dtype=torch.float32)
        self.tau_u = 0.5
        self.global_inih = 0.01
        self.box_width = box_size  # attention box width
        self.Jxx_size = self.box_width * 2 + 1
        self.Jxx = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(self.Jxx_size, self.Jxx_size),
                                   padding=(self.box_width, self.box_width), bias=False)

        tmp_filter = get_kernel(self.Jxx_size, round(self.box_width / 2) + 1)
        tmp_filter = tmp_filter.reshape((1, 1, self.Jxx_size, self.Jxx_size))
        self.Jxx.weight.data = torch.from_numpy(tmp_filter)
        self.resizer = Resize((self.Jxx_size, self.Jxx_size))

        self.U = self.U.to(self.device)
        self.Jxx = self.Jxx.to(self.device)

        self.attentionThr = 40
        self.extend_edge = 7
        self.peak_width = int(self.extend_edge)

    def update_dnf(self, spike):
        inputSpk = torch.reshape(spike, (1, 1, self.spike_h, self.spike_w)).float()

        maxU = torch.relu(self.U)
        squareU = torch.square(maxU)
        r = squareU / (1 + self.global_inih * torch.sum(squareU))
        conv_fired = self.Jxx(inputSpk)
        conv_fired = torch.squeeze(conv_fired).to(self.device)
        du = conv_fired - self.U

        r = torch.reshape(r, (1, 1, self.spike_h, self.spike_w))
        conv_r = self.Jxx(r)
        conv_r = torch.squeeze(conv_r).to(self.device)
        du = conv_r + du
        self.U += (du * self.tau_u).detach()

        del inputSpk, maxU, squareU, r, conv_r, conv_fired, du

    def get_attention_location(self):

        tmpU = torch.relu(self.U - self.attentionThr)
        tmpU = tmpU.cpu()
        tmpU = tmpU.detach().numpy()
        dilated_u = erosion(tmpU, smor.square(self.peak_width))
        peak_cord = peak_local_max(dilated_u, min_distance=self.box_width)
        num_max = len(peak_cord)
        # print('detect %d attention location' % num_max)
        dilated_u[dilated_u > 1] = 1
        dilated_u[dilated_u < 1] = 0
        region_labels = label(dilated_u)
        regions = regionprops(region_labels)
        num_box = len(regions)

        attentionBox = torch.zeros((num_box, 4), dtype=torch.int)

        for region, iBox in zip(regions, range(num_box)):
            minr, minc, maxr, maxc = region.bbox
            beginX = minr - self.extend_edge >= 0 and minr - self.extend_edge or 0
            beginY = minc - self.extend_edge >= 0 and minc - self.extend_edge or 0
            endX = maxr + self.extend_edge < self.spike_h and maxr + self.extend_edge or self.spike_h - 1
            endY = maxc + self.extend_edge < self.spike_w and maxc + self.extend_edge or self.spike_w - 1

            attentionBox[iBox, :] = torch.tensor([beginX, beginY, endX, endY])

        del tmpU, dilated_u, peak_cord

        return attentionBox
