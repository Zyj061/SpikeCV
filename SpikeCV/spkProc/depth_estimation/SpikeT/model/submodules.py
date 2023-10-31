import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn import init
import math


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation='relu', norm=None,
                 BN_momentum=0.1):
        super(ConvLayer, self).__init__()

        bias = False if norm == 'BN' else True
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        if activation is not None:
            self.activation = getattr(torch, activation)
        else:
            self.activation = None

        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nn.BatchNorm2d(out_channels, momentum=BN_momentum)
        elif norm == 'IN':
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        out = self.conv2d(x)

        if self.norm in ['BN', 'IN']:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out


class TransposedConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation='relu', norm=None):
        super(TransposedConvLayer, self).__init__()

        bias = False if norm == 'BN' else True
        self.transposed_conv2d = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=2, padding=padding, output_padding=1, bias=bias)

        if activation is not None:
            self.activation = getattr(torch, activation)
        else:
            self.activation = None

        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        out = self.transposed_conv2d(x)

        if self.norm in ['BN', 'IN']:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out


class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation='relu', norm=None):
        super(UpsampleConvLayer, self).__init__()

        bias = False if norm == 'BN' else True
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        if activation is not None:
            self.activation = getattr(torch, activation)
        else:
            self.activation = None

        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        x_upsampled = f.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        out = self.conv2d(x_upsampled)

        if self.norm in ['BN', 'IN']:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out


class RecurrentConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 recurrent_block_type='convlstm', activation='relu', norm=None, BN_momentum=0.1):
        super(RecurrentConvLayer, self).__init__()

        assert(recurrent_block_type in ['convlstm', 'convgru'])
        self.recurrent_block_type = recurrent_block_type
        if self.recurrent_block_type == 'convlstm':
            RecurrentBlock = ConvLSTM
        else:
            RecurrentBlock = ConvGRU

        # self.conv = ConvLayer(in_channels, out_channels, kernel_size, stride, padding, activation, norm,
        #                      BN_momentum=BN_momentum)
        self.recurrent_block = RecurrentBlock(input_size=out_channels, hidden_size=out_channels, kernel_size=3)

    def forward(self, x, prev_state):
        # x = self.conv(x)
        state = self.recurrent_block(x, prev_state)
        x = state[0] if self.recurrent_block_type == 'convlstm' else state
        return x, state

class Recurrent2ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 recurrent_block_type='convlstm', activation='relu', norm=None, BN_momentum=0.1):
        super(Recurrent2ConvLayer, self).__init__()

        assert(recurrent_block_type in ['convlstm', 'convgru'])
        self.recurrent_block_type = recurrent_block_type
        if self.recurrent_block_type == 'convlstm':
            RecurrentBlock = ConvLSTM
        else:
            RecurrentBlock = ConvGRU

        self.conv = ConvLayer(in_channels, out_channels, kernel_size, stride, padding, activation, norm,
                              BN_momentum=BN_momentum)
        self.recurrent_block = RecurrentBlock(input_size=out_channels, hidden_size=out_channels, kernel_size=3)

    def forward(self, x, prev_state):
        x = self.conv(x)
        state = self.recurrent_block(x, prev_state)
        x = state[0] if self.recurrent_block_type == 'convlstm' else state
        return x, state


class RecurrentPhasedConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 activation='relu', norm=None, BN_momentum=0.1):
        super(RecurrentPhasedConvLayer, self).__init__()

        self.conv = ConvLayer(in_channels, out_channels, kernel_size, stride, padding, activation, norm,
                              BN_momentum=BN_momentum)
        self.recurrent_block = PhasedConvLSTMCell(input_channels=out_channels, hidden_channels=out_channels, kernel_size=3)

    def forward(self, x, times, prev_state):
        x = self.conv(x)
        x, state = self.recurrent_block(x, times, prev_state)
        return x, state


class DownsampleRecurrentConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, recurrent_block_type='convlstm', padding=0, activation='relu'):
        super(DownsampleRecurrentConvLayer, self).__init__()

        self.activation = getattr(torch, activation)

        assert(recurrent_block_type in ['convlstm', 'convgru'])
        self.recurrent_block_type = recurrent_block_type
        if self.recurrent_block_type == 'convlstm':
            RecurrentBlock = ConvLSTM
        else:
            RecurrentBlock = ConvGRU
        self.recurrent_block = RecurrentBlock(input_size=in_channels, hidden_size=out_channels, kernel_size=kernel_size)

    def forward(self, x, prev_state):
        state = self.recurrent_block(x, prev_state)
        x = state[0] if self.recurrent_block_type == 'convlstm' else state
        x = f.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        return self.activation(x), state


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, norm=None,
                 BN_momentum=0.1):
        super(ResidualBlock, self).__init__()
        bias = False if norm == 'BN' else True
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.norm = norm
        if norm == 'BN':
            self.bn1 = nn.BatchNorm2d(out_channels, momentum=BN_momentum)
            self.bn2 = nn.BatchNorm2d(out_channels, momentum=BN_momentum)
        elif norm == 'IN':
            self.bn1 = nn.InstanceNorm2d(out_channels)
            self.bn2 = nn.InstanceNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        if self.norm in ['BN', 'IN']:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.norm in ['BN', 'IN']:
            out = self.bn2(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class PhasedLSTMCell(nn.Module):
    """Phased LSTM recurrent network cell.
    """

    def __init__(
        self,
        hidden_size,
        leak=0.001,
        ratio_on=0.1,
        period_init_min=0.02,
        period_init_max=50.0
    ):
        """
        Args:
            hidden_size: int, The number of units in the Phased LSTM cell.
            leak: float or scalar float Tensor with value in [0, 1]. Leak applied
                during training.
            ratio_on: float or scalar float Tensor with value in [0, 1]. Ratio of the
                period during which the gates are open.
            period_init_min: float or scalar float Tensor. With value > 0.
                Minimum value of the initialized period.
                The period values are initialized by drawing from the distribution:
                e^U(log(period_init_min), log(period_init_max))
                Where U(.,.) is the uniform distribution.
            period_init_max: float or scalar float Tensor.
                With value > period_init_min. Maximum value of the initialized period.
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.ratio_on = ratio_on
        self.leak = leak

        # initialize time-gating parameters
        period = torch.exp(
            torch.Tensor(hidden_size).uniform_(
                math.log(period_init_min), math.log(period_init_max)
            )
        )
        #self.tau = nn.Parameter(period)
        self.register_parameter("tau", nn.Parameter(period))

        phase = torch.Tensor(hidden_size).uniform_() * period
        self.register_parameter("phase", nn.Parameter(phase))
        self.phase.requires_grad = True
        self.tau.requires_grad = True
        #self.phase = nn.Parameter(phase)

    def _compute_phi(self, t):
        t_ = t.view(-1, 1).repeat(1, self.hidden_size)
        phase_ = self.phase.view(1, -1).repeat(t.shape[0], 1)
        tau_ = self.tau.view(1, -1).repeat(t.shape[0], 1)
        tau_.to(t_.device)
        phase_.to(t_.device)
        phi = self._mod((t_ - phase_), tau_)
        phi = torch.abs(phi) / tau_
        return phi

    def _mod(self, x, y):
        """Modulo function that propagates x gradients."""
        return x + (torch.fmod(x, y) - x).detach()

    def set_state(self, c, h):
        self.h0 = h
        self.c0 = c

    def forward(self, c_s, h_s, t):
        # print(c_s.size(), h_s.size(), t.size())
        phi = self._compute_phi(t)

        # Phase-related augmentations
        k_up = 2 * phi / self.ratio_on
        k_down = 2 - k_up
        k_closed = self.leak * phi

        k = torch.where(phi < self.ratio_on, k_down, k_closed)
        k = torch.where(phi < 0.5 * self.ratio_on, k_up, k)
        k = k.view(c_s.shape[0], -1)
        c_s_new = k * c_s + (1 - k) * self.c0
        h_s_new = k * h_s + (1 - k) * self.h0

        return h_s_new, c_s_new


class ConvLSTM(nn.Module):
    """Adapted from: https://github.com/Atcold/pytorch-CortexNet/blob/master/model/ConvLSTMCell.py """

    def __init__(self, input_size, hidden_size, kernel_size):
        super(ConvLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        pad = kernel_size // 2

        # cache a tensor filled with zeros to avoid reallocating memory at each inference step if --no-recurrent is enabled
        self.zero_tensors = {}

        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size, padding=pad)

    def forward(self, input_, prev_state=None):
        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:

            # create the zero tensor if it has not been created already
            state_size = tuple([batch_size, self.hidden_size] + list(spatial_size))
            if state_size not in self.zero_tensors:
                # allocate a tensor with size `spatial_size`, filled with zero (if it has not been allocated already)
                self.zero_tensors[state_size] = (
                    torch.zeros(state_size, dtype=input_.dtype).to(input_.device),
                    torch.zeros(state_size, dtype=input_.dtype).to(input_.device)
                )

            prev_state = self.zero_tensors[tuple(state_size)]

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return hidden, cell


class PhasedConvLSTMCell(nn.Module):
    def __init__(
        self,
        input_channels,
        hidden_channels,
        kernel_size=3
    ):
        super().__init__()
        self.hidden_channels = hidden_channels

        self.lstm = ConvLSTM(
            input_size=input_channels,
            hidden_size=hidden_channels,
            kernel_size=kernel_size
        )

        # as soon as spatial dimension is known, phased lstm cell is instantiated
        self.phased_cell = None
        self.hidden_size = None

    def forward(self, input, times, prev_state=None):
        # input: B x C x H x W
        # times: B
        # returns: output: B x C_out x H x W,   prev_state: (B x C_out x H x W, B x C_out x H x W)

        B, C, H, W = input.shape

        if self.hidden_size is None:
            self.hidden_size = self.hidden_channels * W * H
            self.phased_cell = PhasedLSTMCell(hidden_size=self.hidden_size)
            self.phased_cell = self.phased_cell.to(input.device)
            self.phased_cell.requires_grad = True

        if prev_state is None:
            h0 = input.new_zeros((B, self.hidden_channels, H, W))
            c0 = input.new_zeros((B, self.hidden_channels, H, W))
        else:
            c0, h0 = prev_state

        self.phased_cell.set_state(c0.view(B, -1), h0.view(B, -1))

        c_t, h_t = self.lstm(input, (c0, h0))

        # reshape activation maps such that phased lstm can use them
        (c_s, h_s) = self.phased_cell(c_t.view(B, -1), h_t.view(B, -1), times)

        # reshape to feed to conv lstm
        c_s = c_s.view(B, -1, H, W)
        h_s = h_s.view(B, -1, H, W)

        return h_t, (c_s, h_s)


class ConvGRU(nn.Module):
    """
    Generate a convolutional GRU cell
    Adapted from: https://github.com/jacobkimmel/pytorch_convgru/blob/master/convgru.py
    """

    def __init__(self, input_size, hidden_size, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)

        init.orthogonal_(self.reset_gate.weight)
        init.orthogonal_(self.update_gate.weight)
        init.orthogonal_(self.out_gate.weight)
        init.constant_(self.reset_gate.bias, 0.)
        init.constant_(self.update_gate.bias, 0.)
        init.constant_(self.out_gate.bias, 0.)

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = torch.zeros(state_size, dtype=input_.dtype).to(input_.device)

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        update = torch.sigmoid(self.update_gate(stacked_inputs))
        reset = torch.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = torch.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=1)))
        new_state = prev_state * (1 - update) + out_inputs * update

        return new_state


class RecurrentResidualLayer(nn.Module):
    def __init__(self, in_channels, out_channels,
                 recurrent_block_type='convlstm', norm=None, BN_momentum=0.1):
        super(RecurrentResidualLayer, self).__init__()

        assert(recurrent_block_type in ['convlstm', 'convgru'])
        self.recurrent_block_type = recurrent_block_type
        if self.recurrent_block_type == 'convlstm':
            RecurrentBlock = ConvLSTM
        else:
            RecurrentBlock = ConvGRU
        self.conv = ResidualBlock(in_channels=in_channels,
                                  out_channels=out_channels,
                                  norm=norm,
                                  BN_momentum=BN_momentum)
        self.recurrent_block = RecurrentBlock(input_size=out_channels,
                                              hidden_size=out_channels,
                                              kernel_size=3)

    def forward(self, x, prev_state):
        x = self.conv(x)
        state = self.recurrent_block(x, prev_state)
        x = state[0] if self.recurrent_block_type == 'convlstm' else state
        return x, state
