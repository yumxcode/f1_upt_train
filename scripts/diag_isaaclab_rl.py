#!/usr/bin/env python3
"""Diagnostic: print isaaclab_rl RSL-RL cfg field names to locate the actor-critic class hook.
Run via gm-run (platform image). Prints unconditionally (probe style)."""
import os, sys, dataclasses
print("[DIAG] python", sys.version.split()[0])
try:
    import isaaclab_rl
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg
    print("[DIAG] isaaclab_rl", getattr(isaaclab_rl, "__version__", "?"))
    for cls in (RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg):
        fields = [f.name for f in dataclasses.fields(cls)] if dataclasses.is_dataclass(cls) else [a for a in dir(cls) if not a.startswith("_")]
        print(f"[DIAG] {cls.__name__} fields:", fields)
        print(f"[DIAG] {cls.__name__} has actor_critic_class:", "actor_critic_class" in fields, getattr(cls, "actor_critic_class", None))
    # how the runner consumes it
    import inspect
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunner
    src = inspect.getsource(RslRlOnPolicyRunner.__init__)
    for line in src.splitlines():
        if "actor_critic_class" in line or "policy." in line:
            print("[DIAG] runner:", line.strip())
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "source", "whole_body_tracking"))
    from whole_body_tracking.tasks.tracking.config.f1.agents.rsl_rl_ppo_cfg import F1FlatPPORunnerCfg
    c = F1FlatPPORunnerCfg()
    print("[DIAG] F1FlatPPORunnerCfg post-init: runner.actor_critic_class=", getattr(c, "actor_critic_class", "ABSENT"), "| policy.actor_critic_class=", getattr(c.policy, "actor_critic_class", "ABSENT"))
    os.environ["UPT_MOE"] = "1"
    c2 = F1FlatPPORunnerCfg()
    print("[DIAG] with UPT_MOE=1: policy.actor_critic_class=", getattr(c2.policy, "actor_critic_class", "ABSENT"), "| runner.actor_critic_class=", getattr(c2, "actor_critic_class", "ABSENT"))
except Exception as e:
    import traceback; traceback.print_exc()
    print("[DIAG] ERROR:", e)
print("[DIAG] DONE")
