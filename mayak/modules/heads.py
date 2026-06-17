import torch
import torch.nn as nn
import torch.nn.functional as F

from mayak.constants import DZ, NQ, inv_softplus


class Heads(nn.Module):

    def __init__(self):
        super().__init__()
        self.zproj = nn.Linear(DZ, 4)
        self.fc1 = nn.Linear(16, 48)
        self.fc2 = nn.Linear(48, 2 + (NQ - 1))
        gaps = torch.tensor([0.674, 0.608, 0.363, 0.674, 0.608, 0.363])
        with torch.no_grad():
            self.fc2.weight.zero_()
            self.fc2.bias.zero_()
            self.fc2.bias[1] = 0.9
            self.fc2.bias[2:] = inv_softplus(gaps)

    def forward(self, o, Eg, sun_fut, log_sigma, z, e):
        Bsz, Hn = o.shape
        zp = self.zproj(z)[:, None, :].expand(Bsz, Hn, 4)
        hn = (torch.arange(1, Hn + 1, dtype=o.dtype, device=o.device) / Hn
              )[None, :, None].expand(Bsz, Hn, 1)
        le = torch.log1p(e.mean(-1))[:, None, None].expand(Bsz, Hn, 1)

        x = torch.cat([o[..., None], Eg, Eg.sum(-1, keepdim=True), sun_fut,
                       log_sigma[..., None], zp, hn, le], dim=-1)
        out = self.fc2(F.gelu(self.fc1(x)))

        r = 0.6 * torch.tanh(out[..., 0])
        ratio = 0.08 + torch.sigmoid(out[..., 1] + 1.5)
        gaps = F.softplus(out[..., 2:])

        lo = torch.flip(torch.cumsum(gaps[..., :3], dim=-1), dims=(-1,))
        hi = torch.cumsum(gaps[..., 3:], dim=-1)
        off = torch.cat([-lo, torch.zeros_like(r)[..., None], hi], dim=-1)
        return r, ratio, off