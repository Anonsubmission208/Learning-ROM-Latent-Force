import torch
from .kernels import kernel_ode, kernel_rbf

def gaussian_process_ode(train_time_points, prior_mean, prior_var, test_time_points, length_scale=1, nat_freq=None, damp_ratio=None, amp=None):
    device = train_time_points.device
    B, T = train_time_points.shape
    shared_train = torch.equal(train_time_points, train_time_points[:1].expand_as(train_time_points))
    shared_test  = torch.equal(test_time_points,  test_time_points[:1].expand_as(test_time_points))

    if shared_train:
        t1   = train_time_points[:1]
        K_tt = kernel_ode(t1, t1, length_scale, nat_freq, damp_ratio).to(device).expand(B, -1, -1)
    else:
        K_tt = kernel_ode(train_time_points, train_time_points, length_scale, nat_freq, damp_ratio).to(device)
    if amp is not None:
        K_tt = amp * K_tt

    prior_var = torch.clamp(prior_var, min=1e-12)
    L = torch.linalg.cholesky(K_tt + torch.diag_embed(prior_var))

    const = T * torch.log(torch.tensor([2*torch.pi], device=device))
    logdet = 2 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), 1)
    m = prior_mean.unsqueeze(-1)
    quad = (m.transpose(-1, -2) @ torch.cholesky_solve(m, L)).view(B)
    gp_likelihood = -0.5 * (const + quad + logdet)

    if torch.equal(train_time_points, test_time_points):
        Ainv_K  = torch.cholesky_solve(K_tt, L)
        postcov = K_tt - K_tt @ Ainv_K
        gp_var  = torch.diagonal(postcov, dim1=-2, dim2=-1) + 1e-6
        gp_mean = (K_tt @ torch.cholesky_solve(m, L)).squeeze(-1)
        return gp_mean, gp_var, gp_likelihood

    if shared_train and shared_test:
        s1 = test_time_points[:1]
        K_tstar    = kernel_ode(t1, s1, length_scale, nat_freq, damp_ratio).to(device).expand(B, -1, -1)
        K_starstar = kernel_ode(s1, s1, length_scale, nat_freq, damp_ratio).to(device).expand(B, -1, -1)
    else:
        K_tstar    = kernel_ode(train_time_points, test_time_points, length_scale, nat_freq, damp_ratio).to(device)
        K_starstar = kernel_ode(test_time_points,  test_time_points,  length_scale, nat_freq, damp_ratio).to(device)

    if amp is not None:
        K_tstar = amp * K_tstar
        K_starstar = amp * K_starstar

    invK_Kts = torch.cholesky_solve(K_tstar, L)
    gp_mean  = (K_tstar.transpose(1, 2) @ torch.cholesky_solve(m, L)).squeeze(-1)
    gp_var_f = K_starstar - K_tstar.transpose(1, 2) @ invK_Kts
    gp_var   = torch.diagonal(gp_var_f, dim1=1, dim2=2) + 1e-6
    return gp_mean, gp_var, gp_likelihood
    
def gaussian_process_rbf(train_time_points, prior_mean, prior_var, test_time_points, length_scale=1, amp=None):
    device = train_time_points.device
    B, T = train_time_points.shape
    shared_train = torch.equal(train_time_points, train_time_points[:1].expand_as(train_time_points))
    shared_test  = torch.equal(test_time_points,  test_time_points[:1].expand_as(test_time_points))

    if shared_train:
        t1   = train_time_points[:1]
        K_tt = kernel_rbf(t1, t1, length_scale).to(device).expand(B, -1, -1)
    else:
        K_tt = kernel_rbf(train_time_points, train_time_points, length_scale).to(device)
    if amp is not None:
        K_tt = amp * K_tt
        
    L = torch.linalg.cholesky(K_tt + torch.diag_embed(prior_var))

    const = T * torch.log(torch.tensor([2*torch.pi], device=device))
    logdet = 2 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), 1)
    m = prior_mean.unsqueeze(-1)
    quad = (m.transpose(-1, -2) @ torch.cholesky_solve(m, L)).view(B)
    gp_likelihood = -0.5 * (const + quad + logdet)

    if torch.equal(train_time_points, test_time_points):
        Ainv_K  = torch.cholesky_solve(K_tt, L)
        postcov = K_tt - K_tt @ Ainv_K
        gp_var  = torch.diagonal(postcov, dim1=-2, dim2=-1) + 1e-6
        gp_mean = (K_tt @ torch.cholesky_solve(m, L)).squeeze(-1)
        return gp_mean, gp_var, gp_likelihood

    if shared_train and shared_test:
        s1 = test_time_points[:1]
        K_tstar    = kernel_rbf(t1, s1, length_scale).to(device).expand(B, -1, -1)
        K_starstar = kernel_rbf(s1, s1, length_scale).to(device).expand(B, -1, -1)
    else:
        K_tstar    = kernel_rbf(train_time_points, test_time_points, length_scale).to(device)
        K_starstar = kernel_rbf(test_time_points,  test_time_points, length_scale).to(device)

    if amp is not None:
        K_tstar = amp * K_tstar
        K_starstar = amp * K_starstar

    invK_Kts = torch.cholesky_solve(K_tstar, L)
    gp_mean  = (K_tstar.transpose(1, 2) @ torch.cholesky_solve(m, L)).squeeze(-1)
    gp_var_f = K_starstar - K_tstar.transpose(1, 2) @ invK_Kts
    gp_var   = torch.diagonal(gp_var_f, dim1=1, dim2=2) + 1e-6
    return gp_mean, gp_var, gp_likelihood