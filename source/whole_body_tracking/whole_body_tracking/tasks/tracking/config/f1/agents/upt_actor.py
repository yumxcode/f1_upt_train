"""UPT: MoE-lite actor (2 experts, mode-gated) for rsl_rl ActorCritic.

The actor trunk's final hidden layer is split into 2 experts whose outputs are
mixed by an externally supplied mode scalar (from the motion command's current
time step), NOT from the obs vector — this keeps the Linear input dims intact.

API: MoELinear(in_dim, out_dim); forward(x, mode) with mode in [0,1] (N,) or scalar.
UPTActorCritic wraps rsl_rl ActorCritic and patches the last actor layer; the
wrapped forward needs the env mode — rsl_rl's ActorCritic.forward doesn't accept
extra args, so we store the latest mode on the module (set_mode) which the
runner updates each step via a callback hook in the task code (see README).
"""
from __future__ import annotations

import torch
import torch.nn as nn

UPT_MOE = int(__import__('os').environ.get('UPT_MOE', '0'))
print(f"[UPT][PROBE] upt_actor loaded: UPT_MOE={UPT_MOE}")  # unconditional runtime probe


class MoELinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.expert0 = nn.Linear(in_dim, out_dim)
        self.expert1 = nn.Linear(in_dim, out_dim)
        self._mode = torch.zeros(1)  # updated per-step by set_mode

    def set_mode(self, mode):
        self._mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self._mode
        if m.device != x.device:
            m = m.to(x.device)
        # broadcast scalar mode; per-env mode tensor (N,1) also supported
        g = m.reshape(-1, 1) if m.dim() > 0 and m.numel() == x.shape[0] else m.reshape(1, 1).expand(x.shape[0], 1)
        return (1 - g) * self.expert0(x) + g * self.expert1(x)


class UPTActorCritic(nn.Module):
    """Build a standard rsl_rl ActorCritic then swap its actor final hidden Linear with MoELinear."""

    def __new__(cls, num_actor_obs, num_critic_obs, num_actions, **kwargs):
        from rsl_rl.modules import ActorCritic
        ac = ActorCritic(num_actor_obs, num_critic_obs, num_actions, **kwargs)
        if UPT_MOE:
            old = ac.actor[-2]
            assert isinstance(old, nn.Linear)
            moe = MoELinear(old.in_features, old.out_features)
            with torch.no_grad():
                moe.expert0.weight.copy_(old.weight); moe.expert0.bias.copy_(old.bias)
                moe.expert1.weight.copy_(old.weight); moe.expert1.bias.copy_(old.bias)
            ac.actor[-2] = moe
            extra = sum(p.numel() for p in moe.parameters()) - sum(p.numel() for p in old.parameters())
            print(f"[UPT][PROBE] actor final layer -> MoE (params +{extra}); set mode each step via actor.actor[-2].set_mode(cmd.mode_flag)")
        else:
            print("[UPT][PROBE] MoE disabled, plain ActorCritic")
        return ac
