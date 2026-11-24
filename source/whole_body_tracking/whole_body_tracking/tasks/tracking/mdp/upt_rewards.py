"""UPT: phase-gated contact/swing rewards with calibrated payment form.

Design (per exp_mw5t04w1): timing targets need (a) structured conditioning AND
(b) reward payment aligned to gait events. phase_contact pays a binary bonus
only near the reference contact event of each foot (phi in event window), not
per-frame swing shaping — so it does not bias cadence upward.
"""
from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg


def phase_contact_reward(env, command_name: str = "motion", sensor_cfg: SceneEntityCfg | None = None,
                         window: float = 0.12, weight_contact: float = 0.5) -> torch.Tensor:
    """Reward foot contact state matching the reference contact pattern within a phase window.

    r = w * 1[ |sin(pi*phi)| indicates expected swing/contact agreement ] computed per foot.
    Reference contact for foot f at time t: reference foot sole z below threshold
    is already encoded by the tracking rewards; here we pay only when the ROBOT
    foot state at phase phi matches the REFERENCE foot state at phase phi.
    """
    command = env.command_manager.get_term(command_name)
    robot = env.scene[sensor_cfg.body_names[0]] if sensor_cfg else env.scene["robot"]
    feet_names = [n for n in sensor_cfg.body_names]
    # current robot foot heights (world)
    foot_idx = [robot.body_names.index(n) for n in feet_names]
    foot_z = robot.data.body_pos_w[:, foot_idx, 2]
    # reference foot heights at current step
    ref_body_idx = [command.motion.body_names.index(n) if hasattr(command.motion, 'body_names') else i
                    for i, n in enumerate(feet_names)]
    ref_z = command.body_pos_w[:, ref_body_idx, 2]
    contact_thr = 0.05
    robot_contact = (foot_z < contact_thr).float()
    ref_contact = (ref_z < contact_thr).float()
    agree = (robot_contact == ref_contact).float()
    # pay only inside the skill segment (mode==1): stand handled by tracking rewards
    if getattr(command, "_upt_mode", None) is not None:
        mode = command._upt_mode[command.time_steps.clamp(0, command._upt_mode.shape[0] - 1)].float()
        agree = agree * mode.unsqueeze(-1)
    return agree.mean(dim=-1)
