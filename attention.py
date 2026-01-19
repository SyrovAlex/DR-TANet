import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

def unfold_vertical(x, kernel_size, stride=1):
    """Эквивалент x.unfold(2, kernel_size, stride)"""
    B, C, H, W = x.shape
    if H < kernel_size:
        raise ValueError(f"Input height {H} < kernel_size {kernel_size}")
    unfolded = F.unfold(x, kernel_size=(kernel_size, 1), stride=(stride, 1))
    H_out = (H - kernel_size) // stride + 1
    unfolded = unfolded.view(B, C, kernel_size, H_out, W)
    unfolded = unfolded.permute(0, 1, 3, 4, 2).contiguous()  # (B, C, H_out, W, k)
    return unfolded


def unfold_horizontal(x, kernel_size, stride=1):
    """Эквивалент x.unfold(3, kernel_size, stride)"""
    B, C, H, W = x.shape
    if W < kernel_size:
        raise ValueError(f"Input width {W} < kernel_size {kernel_size}")
    unfolded = F.unfold(x, kernel_size=(1, kernel_size), stride=(1, stride))
    W_out = (W - kernel_size) // stride + 1
    unfolded = unfolded.view(B, C, kernel_size, H, W_out)
    unfolded = unfolded.permute(0, 1, 3, 4, 2).contiguous()  # (B, C, H, W_out, k)
    return unfolded


def unfold_2d(x, kernel_size, stride=1):
    """Эквивалент x.unfold(2, k, s).unfold(3, k, s)"""
    B, C, H, W = x.shape
    if H < kernel_size or W < kernel_size:
        raise ValueError(f"Input spatial size ({H}, {W}) < kernel_size {kernel_size}")
    unfolded = F.unfold(x, kernel_size=kernel_size, stride=stride)
    H_out = (H - kernel_size) // stride + 1
    W_out = (W - kernel_size) // stride + 1
    unfolded = unfolded.view(B, C, kernel_size, kernel_size, H_out, W_out)
    unfolded = unfolded.permute(0, 1, 4, 5, 2, 3).contiguous()  # (B, C, H_out, W_out, k, k)
    return unfolded


class Temporal_Attention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                 groups=1, bias=False, refinement=False):
        super(Temporal_Attention, self).__init__()
        self.outc = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.refinement = refinement

        assert self.outc % self.groups == 0, 'out_channels should be divided by groups.'

        self.w_q = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.w_k = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.w_v = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

        # Relative positional encoding
        self.rel_h = nn.Parameter(torch.randn(self.outc // 2, 1, 1, self.kernel_size, 1), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(self.outc // 2, 1, 1, 1, self.kernel_size), requires_grad=True)
        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)

        init.kaiming_normal_(self.w_q.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.w_k.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.w_v.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, feature_map):
        # Разделение на t0 и t1 (без assert для ONNX)
        fm_t0, fm_t1 = torch.split(feature_map, feature_map.size(1) // 2, dim=1)
        batch, _, h, w = fm_t0.shape

        padded_fm_t0 = F.pad(fm_t0, [self.padding, self.padding, self.padding, self.padding])
        q_out = self.w_q(fm_t1)
        k_out = self.w_k(padded_fm_t0)
        v_out = self.w_v(padded_fm_t0)

        if self.refinement:
            padding = self.kernel_size
            padded_fm_col = F.pad(fm_t0, [0, 0, padding, padding])  # top/bottom
            padded_fm_row = F.pad(fm_t0, [padding, padding, 0, 0])  # left/right

            k_out_col = self.w_k(padded_fm_col)
            k_out_row = self.w_k(padded_fm_row)
            v_out_col = self.w_v(padded_fm_col)
            v_out_row = self.w_v(padded_fm_row)

            large_kernel = 2 * self.kernel_size + 1
            k_out_col = unfold_vertical(k_out_col, kernel_size=large_kernel, stride=self.stride)
            k_out_row = unfold_horizontal(k_out_row, kernel_size=large_kernel, stride=self.stride)
            v_out_col = unfold_vertical(v_out_col, kernel_size=large_kernel, stride=self.stride)
            v_out_row = unfold_horizontal(v_out_row, kernel_size=large_kernel, stride=self.stride)

        # Base attention
        q_out_base = q_out.view(batch, self.groups, self.outc // self.groups, h, w, 1)
        q_out_base = q_out_base.expand(-1, -1, -1, -1, -1, self.kernel_size * self.kernel_size)

        k_out = unfold_2d(k_out, kernel_size=self.kernel_size, stride=self.stride)
        v_out = unfold_2d(v_out, kernel_size=self.kernel_size, stride=self.stride)

        k_out_h, k_out_w = torch.split(k_out, self.outc // 2, dim=1)
        k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)
        k_out = k_out.view(batch, self.groups, self.outc // self.groups, h, w, -1)
        v_out = v_out.view(batch, self.groups, self.outc // self.groups, h, w, -1)

        inter_out = (q_out_base * k_out).sum(dim=2)
        out_weights = F.softmax(inter_out, dim=-1)
        out = torch.einsum('bnhwk,bnchwk -> bnchw', out_weights, v_out).contiguous().view(batch, -1, h, w)

        if self.refinement:
            k_out_row = k_out_row.view(batch, self.groups, self.outc // self.groups, h, w, -1)
            k_out_col = k_out_col.view(batch, self.groups, self.outc // self.groups, h, w, -1)
            v_out_row = v_out_row.view(batch, self.groups, self.outc // self.groups, h, w, -1)
            v_out_col = v_out_col.view(batch, self.groups, self.outc // self.groups, h, w, -1)

            q_out_ref = q_out.view(batch, self.groups, self.outc // self.groups, h, w, 1)
            q_out_ref = q_out_ref.expand(-1, -1, -1, -1, -1, 2 * self.kernel_size + 1)

            out_row_weights = F.softmax((q_out_ref * k_out_row).sum(dim=2), dim=-1)
            out_col_weights = F.softmax((q_out_ref * k_out_col).sum(dim=2), dim=-1)

            out += torch.einsum('bnhwk,bnchwk -> bnchw', out_row_weights, v_out_row).contiguous().view(batch, -1, h, w)
            out += torch.einsum('bnhwk,bnchwk -> bnchw', out_col_weights, v_out_col).contiguous().view(batch, -1, h, w)

        return out





