"""UPT C3: MoE-lite actor — learned-router 2-expert final hidden layer.

Final hidden Linear of the rsl_rl ActorCritic is replaced by:
    y = (1 - g) * E0(x) + g * E1(x),   g = sigmoid(router(x))
Init: E0 = E1 = old layer, router bias 0 -> y == old layer at step 0 (no warm-up shock).
Router/experts all train end-to-end with PPO (no external mode input needed).

Checkpoint keys: actor.<i>.expert0.weight / expert1.weight / router.weight.
load_policy_moe() below rebuilds such a checkpoint for sim2sim.
"""
from __future__ import annotations

import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

UPT_MOE = int(os.environ.get('UPT_MOE', '0'))
print(f"[UPT][PROBE] upt_actor loaded: UPT_MOE={UPT_MOE}")  # unconditional runtime probe


class MoELinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.expert0 = nn.Linear(in_dim, out_dim)
        self.expert1 = nn.Linear(in_dim, out_dim)
        self.router = nn.Linear(in_dim, 1)
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)  # g=0.5 at init; with E0==E1 the layer equals the original

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.router(x))
        return (1 - g) * self.expert0(x) + g * self.expert1(x)


class UPTActorCritic(nn.Module):
    """Factory: builds a standard rsl_rl ActorCritic, then swaps actor's final hidden Linear for MoELinear."""

    def __new__(cls, num_actor_obs, num_critic_obs, num_actions, **kwargs):
        from rsl_rl.modules import ActorCritic
        ac = ActorCritic(num_actor_obs, num_critic_obs, num_actions, **kwargs)
        if UPT_MOE:
            linears = [i for i, m in enumerate(ac.actor) if isinstance(m, nn.Linear)]
            assert len(linears) >= 2, "actor needs >=2 Linear layers for MoE"
            idx = linears[-2]  # final hidden layer
            old = ac.actor[idx]
            moe = MoELinear(old.in_features, old.out_features)
            with torch.no_grad():
                moe.expert0.weight.copy_(old.weight); moe.expert0.bias.copy_(old.bias)
                moe.expert1.weight.copy_(old.weight); moe.expert1.bias.copy_(old.bias)
                # break the expert/router symmetry: identical experts + zero router receive identical gradients and
                # never specialize (observed in Run H TASK_20261126_159: expert0==expert1, router==0 after 6000 iters)
                moe.expert1.weight.add_(0.01 * torch.randn_like(moe.expert1.weight))
                moe.expert1.bias.add_(0.01 * torch.randn_like(moe.expert1.bias))
                moe.router.weight.copy_(0.1 / (old.in_features ** 0.5) * torch.randn_like(moe.router.weight))
            ac.actor[idx] = moe
            extra = sum(p.numel() for p in moe.parameters()) - sum(p.numel() for p in old.parameters())
            print(f"[UPT][PROBE] actor[{idx}] -> MoE (in={old.in_features}, out={old.out_features}, params +{extra})")
        else:
            print("[UPT][PROBE] MoE disabled, plain ActorCritic")
        return ac


def load_policy_moe(checkpoint_path: str, device: str = "cpu"):
    """Load an rsl_rl checkpoint whose actor may contain MoELinear layers (for sim2sim)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_dict = ckpt.get("model_dict") or ckpt.get("model_state_dict")
    actor_keys = sorted([k for k in model_dict if k.startswith("actor.")])
    max_idx = max(int(k.split(".")[1]) for k in actor_keys)
    modules = []
    obs_dim = None; action_dim = None; n_linear = 0
    for i in range(max_idx + 1):
        w_key = f"actor.{i}.weight"
        if f"actor.{i}.expert0.weight" in model_dict:
            w0 = model_dict[f"actor.{i}.expert0.weight"]; w1 = model_dict[f"actor.{i}.expert1.weight"]
            wr = model_dict[f"actor.{i}.router.weight"]
            moe = MoELinear(w0.shape[1], w0.shape[0])
            moe.expert0.weight.data = w0; moe.expert0.bias.data = model_dict[f"actor.{i}.expert0.bias"]
            moe.expert1.weight.data = w1; moe.expert1.bias.data = model_dict[f"actor.{i}.expert1.bias"]
            moe.router.weight.data = wr; moe.router.bias.data = model_dict[f"actor.{i}.router.bias"]
            modules.append(moe); n_linear += 1
            action_dim = w0.shape[0]
        elif w_key in model_dict:
            lin = torch.nn.Linear(model_dict[w_key].shape[1], model_dict[w_key].shape[0])
            lin.weight.data = model_dict[w_key]; lin.bias.data = model_dict[f"actor.{i}.bias"]
            modules.append(lin); n_linear += 1
            action_dim = model_dict[w_key].shape[0]
            if obs_dim is None:
                obs_dim = model_dict[w_key].shape[1]
    # insert ELU between consecutive linear-like modules
    out = []
    for j, m in enumerate(modules):
        out.append(m)
        if j < len(modules) - 1:
            out.append(torch.nn.ELU())
    actor = torch.nn.Sequential(*out).to(device); actor.eval()
    norm_state = ckpt.get("obs_normalizer") or ckpt.get("obs_norm_state_dict")
    normalizer = None
    if isinstance(norm_state, dict):
        mean_np = norm_state.get("mean", norm_state.get("_mean")); var_np = norm_state.get("var", norm_state.get("_var"))
        if mean_np is not None:
            normalizer = (torch.tensor(np.asarray(mean_np), dtype=torch.float32).to(device),
                          torch.sqrt(torch.tensor(np.asarray(var_np), dtype=torch.float32).to(device) + 1e-6))
    print(f"[policy] MoE-aware loader: obs_dim={obs_dim}, action_dim={action_dim}, layers={n_linear}")
    return actor, normalizer, obs_dim, action_dim
