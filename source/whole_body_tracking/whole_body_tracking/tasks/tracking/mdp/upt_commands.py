"""UPT: phase/mode utilities for MotionLoader & MotionCommand (append-only hooks)."""
from __future__ import annotations

import numpy as np
import torch

UPT_PHASE_COND = int(__import__('os').environ.get('UPT_PHASE_COND', '0'))
print(f"[UPT][PROBE] upt_commands loaded: UPT_PHASE_COND={UPT_PHASE_COND}")  # unconditional runtime probe


def load_phase_mode(motion_file: str, device: str = "cpu"):
    """Load optional phase/mode arrays from a UPT npz. Returns (phase, mode) tensors or (None, None)."""
    try:
        data = np.load(motion_file, allow_pickle=True)
    except Exception as e:  # noqa: BLE001
        print(f"[UPT][PROBE] phase/mode load failed: {e}")
        return None, None
    if "phase" in data and "mode" in data:
        phase = torch.tensor(np.asarray(data["phase"], dtype=np.float32), device=device)
        mode = torch.as_tensor(np.asarray(data["mode"]), dtype=torch.long, device=device)
        print(f"[UPT][PROBE] phase/mode loaded: T={phase.shape[0]} mode1_frac={float((mode == 1).float().mean()):.2f}")
        return phase, mode
    print("[UPT][PROBE] npz has no phase/mode; UPT disabled for this motion")
    return None, None


def attach_phase_mode(command):
    """Attach phase_sin_cos/mode_flag properties to a MotionCommand instance (call from its __init__ tail)."""
    motion = command.motion
    phase, mode = load_phase_mode(command.cfg.motion_file, device=command.device)
    command._upt_phase = phase
    command._upt_mode = mode
    if phase is not None:
        # precompute sin/cos
        command._upt_phase_sin_cos = torch.stack([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)
        # phi-dot (rad/s of normalized phase): central differences * fps
        fps = float(motion.fps) if not torch.is_tensor(motion.fps) else float(motion.fps.item() if motion.fps.numel() == 1 else motion.fps[0])
        dphi = torch.gradient(phase)[0] * fps
        command._upt_phase_dot = dphi.clamp(-10.0, 10.0)
        print(f"[UPT][PROBE] phase attached: fps={fps}")
    else:
        command._upt_phase_sin_cos = None
        command._upt_phase_dot = None
    return command


def phase_obs(command, env_ids=None):
    """[sin phi, cos phi, phi_dot, mode] at current time step; zeros if UPT data missing."""
    if command._upt_phase_sin_cos is None:
        dev = command.time_steps.device
        return torch.zeros((command.time_steps.shape[0], 4), device=dev)
    ts = command.time_steps.clamp(0, command._upt_phase_sin_cos.shape[0] - 1)
    sc = command._upt_phase_sin_cos[ts]
    pd = command._upt_phase_dot[ts].unsqueeze(-1)
    md = command._upt_mode[ts].float().unsqueeze(-1)
    return torch.cat([sc, pd, md], dim=-1)


def stand_bin_boost(command, boost: float):
    """C2 (stand/stop as first-class skills): sampling prior for time bins dominated by stand/stop frames (mode==0).
    BeyondMimic's adaptive sampler draws bins ~ failure_count + uniform floor, so 'easy' stand/stop bins are
    under-sampled and the policy drifts when asked to hold still (Run E/G': stand dev 0.15-0.18 rad vs gate 0.10).
    This adds `boost * uniform_floor` to every stand bin (computed once from the mode array)."""
    mask = getattr(command, "_upt_stand_bin_mask", None)
    if mask is None:
        mode = getattr(command, "_upt_mode", None)
        bc = int(command.bin_count)
        if mode is None:
            # mode array not attached yet (attach_phase_mode runs after the first resample) -> do not cache
            return torch.zeros(bc, device=command.device)
        if True:
            T = int(command.motion.time_step_total)
            ts = torch.arange(T, device=command.device)
            b = torch.clamp((ts * bc) // max(T, 1), 0, bc - 1)
            m0 = (mode[:T] == 0).float()
            cnt = torch.bincount(b, minlength=bc).clamp(min=1)
            m0c = torch.bincount(b, weights=m0, minlength=bc)
            mask = ((m0c / cnt) >= 0.5).float()
        command._upt_stand_bin_mask = mask
        print(f"[UPT][PROBE] stand-bin sampling prior: {int(mask.sum())}/{bc} bins flagged (mode==0 frac>=0.5), boost={boost}, mode_len={int(mode.shape[0])}")
    floor = command.cfg.adaptive_uniform_ratio / float(command.bin_count)
    return float(boost) * floor * mask
