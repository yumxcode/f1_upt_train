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
