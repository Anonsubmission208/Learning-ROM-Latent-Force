import torch
from ..functions import wofz

def kernel_ode(X,Y,l=1,freq=0.6,damping=0.05):
    def hfcn(l,gamma_q,gamma_p,t,tp):
        out=(gfcn(l,gamma_q,tp,t)-torch.exp(-gamma_p*t)*gfcn(l,gamma_q,tp,0))/(gamma_q+gamma_p)
        return out

    def gfcn(l,gamma_q,t,tp):
        out=2*torch.exp(l**2*gamma_q**2/4)*torch.exp(-gamma_q*(t-tp))-torch.exp(-(t-tp)**2/l**2)*wofz(1j*zfcn(l,t,tp,gamma_q))-torch.exp(-tp**2/l**2)*torch.exp(-gamma_q*t)*wofz(-1j*zfcn(l,0,tp,gamma_q))
        return out

    def zfcn(l,t,tp,gamma_q):
        out=(t-tp)/l-(l*gamma_q)/2
        return out
    
    device = X.device

    d_ode=freq**2 
    c_ode=2*damping*freq
    alpha=c_ode /2
    w=torch.sqrt(4*d_ode-c_ode**2)/2
    
    length_X=X.numel()
    length_Y=Y.numel()

    t = X.double().unsqueeze(-1).expand(*X.shape[:-1], X.shape[-1], Y.shape[-1])
    tp = Y.double().unsqueeze(-2).expand(*Y.shape[:-1], X.shape[-1], Y.shape[-1])
    gamma=alpha+1j*w
    gamma_t=alpha-1j*w

    Sigma = (torch.sqrt(torch.pi*l**2)/8/w**2*(
        hfcn(l,gamma_t,gamma,t,tp) \
        + hfcn(l,gamma,gamma_t,tp,t) \
        + hfcn(l,gamma,gamma_t,t,tp) \
        + hfcn(l,gamma_t,gamma,tp,t) \
        - hfcn(l,gamma_t,gamma_t,t,tp) \
        - hfcn(l,gamma_t,gamma_t,tp,t) \
        - hfcn(l,gamma,gamma,t,tp) \
        - hfcn(l,gamma,gamma,tp,t)
    )).real

    return Sigma

def kernel_rbf(x1, x2, length_scale):
    length_scale = torch.tensor(length_scale, dtype=x1.dtype, device=x1.device)
    diff = x1.unsqueeze(-1) - x2.unsqueeze(-2)
    cov_matrix = torch.exp(-0.5 * (diff / length_scale)**2)
    
    return cov_matrix