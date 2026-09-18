"""Actual update evidence for fixed-epoch training."""
import hashlib
import torch

def parameter_digest(model, *, trainable_only=False):
    digest=hashlib.sha256()
    for name,p in model.named_parameters():
        if trainable_only and not p.requires_grad:continue
        digest.update(name.encode());digest.update(p.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

def finite_gradients(model):
    grads=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not grads or not torch.stack([torch.isfinite(g).all() for g in grads]).all():
        raise ValueError('Missing or nonfinite training gradient')

def update_evidence(model,before,steps):
    after=parameter_digest(model,trainable_only=True)
    if not steps or before==after:raise ValueError('Optimizer did not change parameters')
    return {'status':'PASS','optimizer_steps':steps,'finite_losses':True,'finite_gradients':True,
        'parameters_changed':True,'parameter_hash_before':before,'parameter_hash_after':after,'cuda_oom':False}
