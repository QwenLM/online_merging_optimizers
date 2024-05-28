# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch optimization with Online Merging"""

import math
import warnings
from functools import partial
from typing import Callable, Iterable, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer
from torch.distributions.bernoulli import Bernoulli
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau


import logging
logger = logging.getLogger(__name__)


class ODAdamW(Optimizer):
    """
    Implements Online Merging Optimizer OnDARE AdamW algorithm

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.001):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-06):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
        no_deprecation_warning (`bool`, *optional*, defaults to `False`):
            A flag used to disable the deprecation warning (set to `True` to disable the warning).
        reserve_p (`float`):
            reserve percentage of parameters
        param_name_map (`dict`):
            mapping between params ids in param_groups to parma name
        use_merge (`bool`):
            whether using merging at each step to add reference delta parameters
        alpha (`float`):
            merging rate of the reference delta parameters
        rescale (`bool`):
            whether rescale the rest of parameters, default False
        clip_val (`float`):
            clipping for numerical stability
        online_step (`int`):
            online gap step for applying merging, default 1
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        no_deprecation_warning: bool = True,
        reserve_p = 1,
        param_name_map=None,
        use_merge=False,
        alpha=0,
        rescale=False,
        clip_val=1,
        online_step=1
    ):
        if not no_deprecation_warning:
            warnings.warn(
                "This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch"
                " implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this"
                " warning",
                FutureWarning,
            )
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "correct_bias": correct_bias}
        
        self.reserve_p = reserve_p
        self.param_name_map = param_name_map
        self.use_merge = use_merge
        self.alpha = alpha
        self.rescale = rescale
        self.clip_val = clip_val
        self.eps = eps
        self.online_step = online_step
            
        super().__init__(params, defaults)

    @torch.no_grad()
    def init_ref_param_diff(self, ref_model, base_model):
        '''Obtain delta parameters of the reference model and applied dropout'''
        param_diff = {}
        for name, param in base_model.named_parameters():
            param_diff[name] = param
        for name, param in ref_model.named_parameters():
            param_diff[name] = param - param_diff[name].to(param.device)
            F.dropout(param_diff[name], p=1-self.reserve_p, training=True, inplace=True)
        for gid, group in enumerate(self.param_groups):
            for pid, p in enumerate(group["params"]):
                self.state[p]['tau_ref'] = param_diff[self.param_name_map[(gid, pid)]].to(p.device)
                self.state[p]['tau_ref'] = self.state[p]['tau_ref'].clamp_(min=-self.clip_val, max=self.clip_val)
        del param_diff

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                if 'tau_ref' not in state and self.use_merge:
                    raise RuntimeError('Online Merging optimizer is not properly initialize with reference task vector')

                # State initialization
                if len(state) <= 1: # We pre-init a task vector of ref model
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    # cached delta for k-step online merging
                    if self.online_step != 1:
                        state['cached_delta'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if 'cached_delta' in state:
                    cached_delta = state['cached_delta']
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                '''ADAM Modification Begin'''

                delta = (exp_avg / denom) * -step_size

                # k-step merging processing
                if self.online_step != 1:
                    if state['step'] % self.online_step == 0:
                        # On the k-th step (merging step)
                        # rollback p on the online step
                        p.sub_(cached_delta)
                        # process current delta with cached delta
                        delta.add_(cached_delta)
                        # clean cached delta
                        cached_delta.zero_()
                        # Move to the sparsification and consensus
                        # in merging with rollbacked p and accumulated delta
                    else:
                        # Normal ADAM if not on the online merging step
                        # cache delta in each step
                        cached_delta.add_(delta)
                        p.add_(delta, alpha=1)
                        if group["weight_decay"] > 0.0:
                            p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))
                        return loss


                # Sparsification
                org_sum = delta.abs().sum()
                F.dropout(delta, p=1-self.reserve_p, training=True, inplace=True)
                new_sum = delta.abs().sum()

                # Consensus
                if self.use_merge:
                    # new delta = alpha * tau_ref + (1-alpha) * delta
                    p.add_(state['tau_ref'], alpha=self.alpha)
                    
                    if self.rescale and org_sum >= 1e-8:
                        delta.mul_(org_sum / (new_sum+self.eps))
                        delta.clamp_(min=-self.clip_val, max=self.clip_val)
                        
                    p.add_(delta, alpha=1-self.alpha)
                else:
                    # new delta = delta
                    if self.rescale and org_sum >= 1e-8:
                        delta.mul_(org_sum / (new_sum+self.eps))
                        delta.clamp_(min=-self.clip_val, max=self.clip_val)
                        
                    p.add_(delta, alpha=1)
                
                '''ADAM Modification End'''

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

        return loss


class OTAdamW(Optimizer):
    """
    Implements Online TIES Adam algorithm with weight decay fix as introduced in [Decoupled Weight Decay
    Regularization](https://arxiv.org/abs/1711.05101).

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.001):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-06):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
        no_deprecation_warning (`bool`, *optional*, defaults to `False`):
            A flag used to disable the deprecation warning (set to `True` to disable the warning).
        reserve_p (`float`):
            reserve percentage of parameters
        param_name_map (`dict`):
            mapping between params ids in param_groups to parma name
        use_merge (`bool`):
            whether using merging at each step to add reference delta parameters
        alpha (`float`):
            merging rate of the reference delta parameters
        rescale (`bool`):
            whether rescale the rest of parameters, default False
        online_step (`int`):
            online gap step for applying merging, default 1
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        no_deprecation_warning: bool = True,
        reserve_p = 1,
        param_name_map=None,
        use_merge=False,
        alpha=0,
        rescale=False,
        online_step=1
    ):
        if not no_deprecation_warning:
            warnings.warn(
                "This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch"
                " implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this"
                " warning",
                FutureWarning,
            )
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "correct_bias": correct_bias}
        
        self.reserve_p = reserve_p
        self.use_merge = use_merge
        self.param_name_map = param_name_map
        self.alpha = alpha
        self.rescale = rescale
        self.online_step = online_step
            
        super().__init__(params, defaults)

    @torch.no_grad()
    def init_ref_param_diff(self, ref_model, base_model):
        '''Obtain delta parameters of the reference model and applied topk dropout'''
        param_diff = {}
        for name, param in base_model.named_parameters():
            param_diff[name] = param
        for name, param in ref_model.named_parameters():
            param_diff[name] = param - param_diff[name].to(param.device)
            self.magnitude_mask_(param_diff[name], self.reserve_p, self.rescale)
        for gid, group in enumerate(self.param_groups):
            for pid, p in enumerate(group["params"]):
                self.state[p]['tau_ref'] = param_diff[self.param_name_map[(gid, pid)]].to(p.device)
        del param_diff
    
    @torch.no_grad()
    def magnitude_mask_(self, tensor: torch.Tensor, density: float, rescale=False) -> torch.Tensor:
        """Masks out the smallest values, retaining a proportion of `density`. True if not in largest top density% (drop logic)"""
        if density >= 1:
            return
    
        k = int(density * tensor.numel())
        if k < 1:
            return

        topk = torch.argsort(tensor.view(-1).abs(), descending=True)[:k]
        mask = torch.zeros_like(tensor)
        mask.view(-1)[topk] = 1

        org_sum = tensor.abs().sum()
        tensor.mul_(mask)
        new_sum = tensor.abs().sum()

        if rescale and org_sum >= 1e-8:
            # numeric stable
            tensor.mul_(org_sum / new_sum)

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]
                if 'tau_ref' not in state:
                    raise RuntimeError('Online Merging optimizer is not properly initialize with reference task vector')

                # State initialization
                if len(state) <= 1:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    if self.online_step != 1:
                        state['cached_delta'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if 'cached_delta'in state:
                    cached_delta = state['cached_delta']
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                delta = (exp_avg / denom) * -step_size

                if self.online_step != 1:
                    if state['step'] % self.online_step == 0:
                        # rollback p on the online step
                        p.sub_(cached_delta)
                        # process current delta with cached delta
                        delta.add_(cached_delta)
                        cached_delta.zero_()
                    else:
                        # Normal ADAM if not on the online merging step
                        cached_delta.add_(delta)
                        p.add_(delta, alpha=1)
                        if group["weight_decay"] > 0.0:
                            p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))
                        return loss

                # Sparsification of tau policy
                self.magnitude_mask_(delta, self.reserve_p, self.rescale)

                # Consensus
                deltas = torch.stack([state['tau_ref'], delta], dim=0)
                weights = torch.tensor(
                    [self.alpha, 1-self.alpha], dtype=deltas.dtype, device=deltas.device
                )
                while len(deltas.shape) > len(weights.shape):
                    weights.unsqueeze_(-1)
                majority_sign = ((deltas.sum(dim=0) >= 0) * 2 - 1)
                deltas.mul_(deltas.sign() == majority_sign)

                if self.use_merge:
                    p.add_((deltas * weights).sum(dim=0), alpha=1)
                else:
                    p.add_(deltas[1], alpha=1)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

        return loss