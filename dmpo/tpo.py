import torch.nn.functional as F
from einops import einsum

def exists(val):
    return val is not None

def tpo_target(log_scores, u, eta = 1.0):
    return F.log_softmax(F.log_softmax(log_scores, dim = -1) + u / eta, dim = -1)

def tpo_forward_kl_loss(log_p, log_q):
    q = log_q.exp()
    return -einsum(q, log_p, '... k, ... k -> ...').mean()

def tpo_reverse_kl_loss(log_p, log_q):
    p = log_p.exp()
    return einsum(p, log_p - log_q, '... k, ... k -> ...').mean()

def tpo_js_loss(log_p, log_q, weight = 0.5, eps = 1e-10):
    p = log_p.exp()
    q = log_q.exp()

    m = q.lerp(p, weight)
    log_m = m.clamp(min = eps).log()

    kl_p_m = einsum(p, log_p - log_m, '... k, ... k -> ...').mean()
    kl_q_m = einsum(q, log_q - log_m, '... k, ... k -> ...').mean()

    return kl_q_m.lerp(kl_p_m, weight)
