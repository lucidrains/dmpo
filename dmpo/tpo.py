import torch.nn.functional as F
from einops import einsum

def tpo_target(log_scores, u, eta = 1.0):
    return F.log_softmax(F.log_softmax(log_scores, dim = -1) + u / eta, dim = -1)

def tpo_forward_kl_loss(log_p, log_q):
    q = log_q.exp()
    return -einsum(q, log_p, '... k, ... k -> ...').mean()

def tpo_reverse_kl_loss(log_p, log_q):
    p = log_p.exp()
    return einsum(p, log_p - log_q, '... k, ... k -> ...').mean()
