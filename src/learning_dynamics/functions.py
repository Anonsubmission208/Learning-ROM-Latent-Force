import torch
import math
import numpy as np
import scipy as sp

def truncated_normal(t, mean=0.0, std=0.01):
    torch.nn.init.normal_(t, mean=mean, std=std)
    while True:
        cond = torch.logical_or(t < mean - 2*std, t > mean + 2*std)
        if not torch.sum(cond):
            break
        t = torch.where(cond, torch.nn.init.normal_(torch.ones(t.shape), mean=mean, std=std), t)
    return t

def gauss_cross_entropy(mu1, var1, mu2, var2):
    term0 = math.log(2*math.pi)
    # term1 = torch.log(torch.abs(var2))
    term1 = torch.log(var2)
    # term2 = (var1 + mu1**2 - 2 * mu1 * mu2 + mu2**2) / (torch.abs(var2)) 
    term2 = (var1 + mu1**2 - 2 * mu1 * mu2 + mu2**2) / var2
    cross_entropy = -0.5*(term0 + term1 + term2)

    return cross_entropy

class Faddeeva(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z):
        z_np = z.detach().cpu().numpy()
        w_np = np.array(sp.special.wofz(z_np), dtype=z_np.dtype)
        w = torch.from_numpy(w_np).to(z.device, dtype=z.dtype)
        ctx.save_for_backward(z, w)
        return w

    @staticmethod
    def backward(ctx, grad_output):
        z, w = ctx.saved_tensors
        pi_tensor = torch.tensor(torch.pi, device=z.device).double()
        two_i = torch.complex(
            torch.tensor(0.0, device=z.device).double(),
            torch.tensor(2.0, device=z.device).double()
        )
        sqrt_pi = torch.sqrt(pi_tensor)
        dw_dz = (two_i / sqrt_pi) - 2 * z * w
        gradient = dw_dz.conj() * grad_output
        
        return gradient
wofz = Faddeeva.apply