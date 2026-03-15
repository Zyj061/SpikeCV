# from config import *
import numpy as np
import torch
import torch.nn.functional as F
from collections import namedtuple

detect_box = namedtuple('detect_box', ['zId', 'box', 'velocity'])
tracks = namedtuple('tracks', ['id', 'color', 'bbox', 'predbox', 'visible', 'vel', 'age', 'unvisible_count'])
trajectories = namedtuple('trajectories', ['id', 'x', 'y', 't', 'color'])
tracks_bbox = namedtuple('tracks_bbox', ['id', 't', 'x', 'y', 'h', 'w'])

DTYPE = torch.float64

class stdp_cluster():

    def __init__(self, spike_h, spike_w, box_size, device):

        self.spike_h = spike_h
        self.spike_w = spike_w
        self.box_size = box_size
        self.K1 = 1
        self.K2 = 5
        # self.InputSize = (2 * box_size + 1)**2
        self.InputSize = (2 * box_size + 1) * (2 * box_size + 5)
        self.device = device
        # self.InputSize = box_size * (box_size + 4)
        # self.InputSize = box_size**2

        # self.synaptic_weight = torch.ones(self.K1, self.InputSize, dtype=torch.float64) / self.K1
        self.synaptic_weight = torch.rand(self.K2, self.K1, self.InputSize, dtype=DTYPE)
        # self.synaptic_weight = self.synaptic_weight / torch.sum(self.synaptic_weight)
        # self.synaptic_weight = torch.unsqueeze(self.synaptic_weight, dim=0)
        # self.synaptic_weight = self.synaptic_weight.repeat(self.K2, 1, 1)
        self.bias_weight = torch.ones(self.K2, 1, dtype=DTYPE) / self.K2

        self.synaptic_weight = self.normalization_w(self.synaptic_weight)
        self.bias_weight = self.normalization_w(self.bias_weight)
        self.synaptic_weight = self.synaptic_weight.to(device)
        self.bias_weight = self.bias_weight.to(device)

        self.learning_rate = 0.001
        self.iter_num = 2
        self.stdp_coefficience = 1

        self.w_up = 1
        self.w_low = -8

        self.w_up_tensor = torch.ones_like(self.synaptic_weight, dtype=DTYPE) * self.w_up
        self.w_low_tensor = torch.ones_like(self.synaptic_weight, dtype=DTYPE) * self.w_low

        self.tracks = []
        self.trajectories = []
        self.tracks_bbox = []
        # self.seed_everything(5)
        self._zbuf = torch.zeros(self.K2, device=self.device, dtype=DTYPE)
        self._last_idx = None  # 用张量而不是 python int 更好

        self.background_occ_fr = 10  # background oscillation rate　20 Hz
        self.occ_fr = torch.Tensor([self.background_occ_fr / 20000.0])

        for i_neuron in range(self.K2):
            self.tracks.append(tracks(i_neuron, 255 * np.random.rand(1, 3), torch.zeros((1, 4), dtype=torch.float64),
                                      torch.zeros((1, 4), dtype=torch.float64), 0,
                                      torch.zeros((2,)), 0, 0))
            self.trajectories.append(trajectories(i_neuron, [], [], [], self.tracks[i_neuron].color))
            self.tracks_bbox.append(tracks_bbox(i_neuron, [], [], [], [], []))


    def normalization_w(self, weight):

        return F.log_softmax(weight, dim=-1)

    # winner-take-all
    # @staticmethod
    def wta(self, attention_spikes, synaptic_weight, bias_weight):
        # 与你原来一致：按项目的全局 DTYPE 转
        x = attention_spikes.view(-1)
        # x = attention_spikes.to(dtype=DTYPE).view(-1)  # (InputSize,)

        # (K2, K1, InputSize) -> (K2, InputSize)，不再加 abs(self.w_low)
        W = synaptic_weight.squeeze(1)  # (K2, InputSize)
        b = bias_weight.squeeze(1)  # (K2,)

        # 线性项 + 偏置，做稳定化（减最大值）
        # logits = torch.mv(W, x) + b  # (K2,)
        logits = torch.addmv(b, W, x)

        # logits = logits - logits.max()

        # Gumbel(0,1) = -log(Exp(1))
        g = -torch.empty_like(logits).exponential_().log_()  # 取代 rand_like+两次log
        idx = (logits + g).argmax(dim=0, keepdim=True)

        Z = self._zbuf
        Z.zero_()  # 简单可靠；若想更快再做“只清一位”的优化
        Z.scatter_(0, idx, 1.0)
        self._last_idx = idx  # 保持为张量，避免 .item() 同步
        return Z

    @staticmethod
    def intersect(box_a, box_b):
        """ We resize both tensors to [A,B,2] without new malloc:
        [A,2] -> [A,1,2] -> [A,B,2]
        [B,2] -> [1,B,2] -> [A,B,2]
        Then we compute the area of intersect between box_a and box_b.
        Args:
          box_a: (tensor) bounding boxes, Shape: [A,4].
          box_b: (tensor) bounding boxes, Shape: [B,4].
        Return:
          (tensor) intersection area, Shape: [A,B].
        """
        A = box_a.size(0)
        B = box_b.size(0)
        max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                           box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
        min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                           box_b[:, :2].unsqueeze(0).expand(A, B, 2))
        inter = torch.clamp((max_xy - min_xy), min=0)
        return inter[:, :, 0] * inter[:, :, 1]

    def jaccard(self, box_a, box_b):
        """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
        is simply the intersection over union of two boxes.  Here we operate on
        ground truth boxes and default boxes.
        E.g.:
            A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
        Args:
            box_a: (tensor) Ground truth bounding boxes, Shape: [A,4]
            box_b: (tensor) Prior boxes from priorbox layers, Shape: [B,4]
        Return:
            jaccard overlap: (tensor) Shape: [A, B]
        """
        inter = self.intersect(box_a, box_b)
        area_a = ((box_a[:, 2] - box_a[:, 0]) *
                  (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
        area_b = ((box_b[:, 2] - box_b[:, 0]) *
                  (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
        union = area_a + area_b - inter
        return inter / union  # [A,B]

    @torch.no_grad()
    def update_weight(self, attention_input):

        attention_input = attention_input.to(dtype=DTYPE, device=self.device)
        n_attention = attention_input.shape[2]
        predict_fire = torch.zeros(n_attention, self.K2)
        synaptic_weight = self.synaptic_weight.detach().clone()
        bias_weight = self.bias_weight.detach().clone()
        lr_weight = torch.zeros(self.K2).to(self.device)
        has_fired = np.zeros((self.K2, 1))

        if False:
            print(f"n_attention: {n_attention}")            #5
            print(f"self.iter_num: {self.iter_num}")        #1
            print(f"self.K2: {self.K2}")                    #5

        for iPattern in range(n_attention):
            detected = -1
            input_spike = torch.reshape(attention_input[:, :, iPattern], (-1, 1))
            # background_noise = (torch.rand(input_spike.shape) < self.occ_fr).to(device)
            # input_spike = (torch.logical_or((input_spike).type(torch.bool), background_noise)).type(torch.float32)
            # input_spike = torch.reshape(attention_input, (-1, n_attention))
            confusion_flag = 0
            for i in range(self.iter_num):
                # z_spike = self.wta(input_spike, synaptic_weight, bias_weight).to(self.device)
                z_spike = self.wta(input_spike, synaptic_weight, bias_weight)
                dw_bias = self.learning_rate * (z_spike * torch.squeeze(torch.exp(-bias_weight)) - 1)
                # tmp_sum = torch.sum(dw_bias, dim=1)
                bias_weight += torch.unsqueeze(dw_bias, dim=1).detach()

                # for iZ in range(1) :  # observed >1.0ms saving in badminton
                for iZ in range(self.K2): 
                    if z_spike[iZ] != 0 and has_fired[iZ] == 0 and (detected == -1 or iZ == detected):
                        has_fired[iZ] = 1
                        detected = iZ
                        # fire_idx = torch.where(z_spike[iZ, :]!=0)
                        tmpE = torch.exp(-synaptic_weight[iZ, :, :])
                        dw = self.stdp_coefficience * tmpE * torch.transpose(input_spike, 0, 1) - 1
                        lr_weight[iZ] += 1

                        synaptic_weight[iZ, :, :] += ((1.0 / lr_weight[iZ]) * dw.to(self.device))
                        # synaptic_weight[iZ, :, :] += self.learning_rate * dw.to(device)
                        synaptic_weight.clamp_(min=self.w_low, max=self.w_up)

                        # synaptic_weight = torch.where(synaptic_weight < self.w_up, synaptic_weight,
                        #                               self.w_up_tensor).detach()
                        # synaptic_weight = torch.where(synaptic_weight < self.w_low, self.w_low_tensor,
                        #                               synaptic_weight).detach()

                        synaptic_weight = self.normalization_w(synaptic_weight)
                        bias_weight = self.normalization_w(bias_weight)
                        predict_fire[iPattern, iZ] = torch.Tensor([1])

                # predict_fire[iPattern, :] = z_spike.detach()
            # predict_fire = z_spike.detach()
        # print(synaptic_weight.max())
        # print(synaptic_weight.min())

        del n_attention, lr_weight
        return predict_fire, synaptic_weight, bias_weight

    def seed_everything(self, seed=11):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        return

    @staticmethod
    def detect_object(predict_fire, attention_box, motion_id, motion_vector, **kwargs):

        spike_h = kwargs.get('spike_h')
        spike_w = kwargs.get('spike_w')
        device = kwargs.get('device')
        nAttention = attention_box.shape[0]
        boxId = torch.zeros(nAttention, 1)
        predBox = torch.zeros((nAttention, 4), dtype=torch.int)
        velocities = torch.zeros(nAttention, 2).to(device)
        predict_box = []

        for iPattern in range(nAttention):
            z_spike = predict_fire[iPattern, :]
            if torch.any(z_spike != 0):
                if len(torch.where(z_spike != 0)[0]) > 1:
                    print('check')

                tmp_fired = torch.where(z_spike != 0)[0]
                boxId[iPattern] = tmp_fired[0] + 1

                x = attention_box[iPattern, 0]
                y = attention_box[iPattern, 1]
                end_x = attention_box[iPattern, 2]
                end_y = attention_box[iPattern, 3]

                tmp_motion = torch.zeros(spike_h, spike_w)
                tmp_motion[x:end_x + 1, y:end_y + 1] = motion_id[x:end_x + 1, y:end_y + 1].clone()

                motion_index2d = torch.where(tmp_motion != 0)
                if len(motion_index2d[0]) == 0:
                    continue

                motion_num = len(motion_index2d[0])
                block_veloctiy = torch.zeros(motion_num, 2).to(device)
                block_veloctiy[:, 0] = motion_vector[motion_index2d[0], motion_index2d[1], 0].clone()
                block_veloctiy[:, 1] = motion_vector[motion_index2d[0], motion_index2d[1], 1].clone()
                tmp_veloctiy = torch.mean(block_veloctiy, dim=0)
                velocities[iPattern, :] = tmp_veloctiy.data
                predBox[iPattern, :] = attention_box[iPattern, :]

                predict_box.append(detect_box(boxId[iPattern],
                                              torch.unsqueeze(predBox[iPattern], dim=0),
                                              velocities[iPattern]))
            # else:
            #     print('no tracking neuron fire..')

        del boxId, predBox, velocities
        # torch.cuda.empty_cache()

        return predict_box

    def update_tracks(self, detect_objects, sw, bw, timestep):

        objects_num = len(detect_objects)
        id_check = torch.zeros(self.K2, 1)
        AssignTrk = []

        for iObject in range(objects_num):
            tmp_object = detect_objects[iObject]
            id = int(tmp_object.zId.detach().item())
            box = tmp_object.box
            velocity = tmp_object.velocity

            if id_check[id - 1] != 0:
                if id in AssignTrk:
                    # print('id %d repeat' % (id-1))
                    AssignTrk.remove(id)
                continue
            else:
                id_check[id - 1] = 1

            pred_box = self.tracks[id - 1].predbox
            boxes_iou = self.jaccard(box, pred_box)
            unvisible_count = self.tracks[id - 1].unvisible_count
            if ~(self.tracks[id - 1].predbox[0, 3] != 0 and self.tracks[id - 1].age > 15
                 and boxes_iou < 0.6):
                self.tracks[id - 1] = self.tracks[id - 1]._replace(bbox=box)
                beginX = box[0, 0]
                beginY = box[0, 1]
                endX = box[0, 2]
                endY = box[0, 3]

                beginX = beginX + velocity[0] >= 0 and (beginX + velocity[0]) or 0
                beginY = beginY + velocity[1] >= 0 and (beginY + velocity[1]) or 0
                # endX = beginX + self.box_size * 2 < self.spike_h and (beginX + self.box_size * 2) or (self.spike_h - 1)
                # endY = beginY + self.box_size * 2 < self.spike_w and (beginY + self.box_size * 2) or (self.spike_w - 1)
                endX = endX + velocity[0] < self.spike_h and (endX + velocity[0]) or (self.spike_h - 1)
                endY = endY + velocity[1] < self.spike_w and (endY + velocity[1]) or (self.spike_w - 1)

                tmp_box = torch.tensor([beginX, beginY, endX, endY])
                tmp_box = torch.unsqueeze(tmp_box, dim=0)
                self.tracks[id - 1] = self.tracks[id - 1]._replace(predbox=tmp_box)

                self.tracks[id - 1] = self.tracks[id - 1]._replace(visible=1)
                self.tracks[id - 1] = self.tracks[id - 1]._replace(vel=velocity)
                self.tracks[id - 1] = self.tracks[id - 1]._replace(unvisible_count=0)
                self.tracks[id - 1] = self.tracks[id - 1]._replace(age=self.tracks[id - 1].age + 1)

                # update the trajectories
                self.trajectories[id - 1].x.append((box[0, 0] + self.box_size).item())
                self.trajectories[id - 1].y.append((box[0, 1] + self.box_size).item())
                self.trajectories[id - 1].t.append(timestep)

                # Check if beginX, beginY, endX, endY are int; otherwise, use .item()
                self.tracks_bbox[id - 1].x.append(beginY if isinstance(beginY, int) else beginY.item())
                self.tracks_bbox[id - 1].y.append(beginX if isinstance(beginX, int) else beginX.item())
                self.tracks_bbox[id - 1].h.append(
                    (endX - beginX) if isinstance(endX, int) and isinstance(beginX, int) else (endX - beginX).item())
                self.tracks_bbox[id - 1].w.append(
                    (endY - beginY) if isinstance(endY, int) and isinstance(beginY, int) else (endY - beginY).item())

                self.tracks_bbox[id - 1].t.append(timestep)
                AssignTrk.append(id)
                # print('tracks %d velocity dx: %f dy: %f' % (id, velocity[0], velocity[1]))

        all_id = list(range(1, self.K2 + 1, 1))
        noAssign = [x for x in all_id if x not in AssignTrk]

        noAssign_num = self.K2 - len(AssignTrk)

        for iObject in range(noAssign_num):
            id = noAssign[iObject]
            unvisible_count = self.tracks[id - 1].unvisible_count
            if unvisible_count > 5:
                self.tracks[id - 1] = self.tracks[id - 1]._replace(age=0)
                self.tracks[id - 1] = self.tracks[id - 1]._replace(visible=0)
                # sw[id-1, :, :] = 1 / self.K1
                # bw[id-1] = 1 / self.K2
            else:
                if self.tracks[id - 1].predbox[0, 2] != 0:
                    self.tracks[id - 1] = self.tracks[id - 1]._replace(bbox=self.tracks[id - 1].predbox)
                    beginX = self.tracks[id - 1].predbox[0, 0].item()
                    beginY = self.tracks[id - 1].predbox[0, 1].item()
                    endX = self.tracks[id - 1].predbox[0, 2].item()
                    endY = self.tracks[id - 1].predbox[0, 3].item()

                    beginX = beginX + self.tracks[id - 1].vel[0] >= 0 and (beginX + self.tracks[id - 1].vel[0]) or 0
                    beginY = beginY + self.tracks[id - 1].vel[1] >= 0 and (beginY + self.tracks[id - 1].vel[1]) or 0
                    # endX = beginX + self.box_size * 2 < self.spike_h and (beginX + self.box_size * 2) or (self.spike_h - 1)
                    # endY = beginY + self.box_size * 2 < self.spike_w and (beginY + self.box_size * 2) or (self.spike_w - 1)
                    endX = endX + self.tracks[id - 1].vel[0] < self.spike_h and (endX + self.tracks[id - 1].vel[0]) or (
                                self.spike_h - 1)
                    endY = endY + self.tracks[id - 1].vel[1] < self.spike_w and (endY + self.tracks[id - 1].vel[1]) or (
                                self.spike_w - 1)

                    pred_box = torch.tensor([beginX, beginY, endX, endY])
                    pred_box = torch.unsqueeze(pred_box, dim=0)
                    self.tracks[id - 1] = self.tracks[id - 1]._replace(predbox=pred_box)

                    self.trajectories[id - 1].x.append(
                        (beginX + self.box_size) if isinstance(beginX, int) else (beginX + self.box_size).item())
                    self.trajectories[id - 1].y.append(
                        (beginY + self.box_size) if isinstance(beginY, int) else (beginY + self.box_size).item())
                    self.trajectories[id - 1].t.append(timestep)

                    # Check if beginX, beginY, endX, endY are int; otherwise, use .item()
                    self.tracks_bbox[id - 1].x.append(beginY if isinstance(beginY, int) else beginY.item())
                    self.tracks_bbox[id - 1].y.append(beginX if isinstance(beginX, int) else beginX.item())
                    self.tracks_bbox[id - 1].h.append(
                        (endX - beginX) if isinstance(endX, int) and isinstance(beginX, int) else (
                                endX - beginX).item())
                    self.tracks_bbox[id - 1].w.append(
                        (endY - beginY) if isinstance(endY, int) and isinstance(beginY, int) else (
                                endY - beginY).item())
                    self.tracks_bbox[id - 1].t.append(timestep)
                    # print('predicting location of object %d the %d time' % (id, unvisible_count))

                # print('tracks %d predictive velocity dx: %f, dy: %f' % (
                # id, self.tracks[id - 1].vel[0], self.tracks[id - 1].vel[1]))
                if ~(torch.all(self.synaptic_weight == 1)):
                    sw[id - 1, :, :] = self.synaptic_weight[id - 1, :, :].detach().clone()
                    bw[id - 1] = self.bias_weight[id - 1].detach().clone()
                    # print('correct the weight')
            self.tracks[id - 1] = self.tracks[id - 1]._replace(unvisible_count=self.tracks[id - 1].unvisible_count + 1)

        return sw, bw
