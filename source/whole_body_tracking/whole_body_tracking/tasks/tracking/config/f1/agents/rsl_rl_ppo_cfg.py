import os
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class F1FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 500
    experiment_name = "f1_flat"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    def __post_init__(self):
        super().__post_init__()
        # UPT C3 (MoE-lite): learned-router 2-expert final actor layer. Default off.
        if os.environ.get("UPT_MOE", "0") == "1":
            # IsaacLab 2.1.0 RslRlPpoActorCriticCfg selects the network by STRING: policy.class_name = "ActorCritic",
            # resolved by rsl_rl's runner (resolve_callable: simple names searched inside rsl_rl packages;
            # older runners: getattr(rsl_rl.modules, name)). Register the class under rsl_rl.modules and
            # set the simple name -> works for both. (Two earlier hooks on actor_critic_class were silently ignored.)
            import rsl_rl.modules as _mods
            from whole_body_tracking.tasks.tracking.config.f1.agents.upt_actor import UPTActorCritic
            _mods.UPTActorCritic = UPTActorCritic
            if hasattr(self.policy, "class_name"):
                self.policy.class_name = "UPTActorCritic"
            else:
                self.policy.actor_critic_class = UPTActorCritic
            print(f"[UPT][PROBE] MoE-lite ON: policy.class_name={getattr(self.policy, 'class_name', None)} rsl_rl.modules.UPTActorCritic={getattr(_mods, 'UPTActorCritic', None) is not None}")
        else:
            print("[UPT][PROBE] MoE-lite OFF (UPT_MOE=0): policy.class_name unchanged")
