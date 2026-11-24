"""This sub-module contains the functions that are specific to the locomotion environments."""

from isaaclab.envs.mdp import *  # noqa: F401, F403

from whole_body_tracking.tasks.tracking.mdp import *  # noqa: F401, F403

from .commands import *  # noqa: F401, F403
from .events import *  # noqa: F401, F403
from .observations import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403
from .terminations import *  # noqa: F401, F403

# --- UPT conditional exports (phase conditioning C1 / phase reward) ---
import os as _os
from whole_body_tracking.tasks.tracking.mdp.upt_commands import phase_obs as _upt_phase_obs

def upt_phase_obs_term(env, command_name: str = "motion"):
    cmd = env.command_manager.get_term(command_name)
    if int(_os.environ.get("UPT_PHASE_COND", "0")) and getattr(cmd, "_upt_phase_sin_cos", None) is not None:
        return _upt_phase_obs(cmd)
    import torch
    return torch.zeros((cmd.time_steps.shape[0], 4), device=cmd.time_steps.device)

print(f"[UPT][PROBE] mdp exports ready: UPT_PHASE_COND={_os.environ.get('UPT_PHASE_COND','0')}")
