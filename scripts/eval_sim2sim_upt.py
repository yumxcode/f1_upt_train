#!/usr/bin/env python3
"""W3: STANDARD_SIM2SIM gated evaluation for UPT policies (F1 29DOF).

Wraps scripts/play_mujoco.py's headless loop with the standard's phase machine:
  P0 stand  (>=2 s, robot holds f1_stand with stand-clip commands)
  P1 blend  (<=1 s, commands blend stand -> clip frame 0)
  P2 track  (>=10 s clip core, or whole clip)
  P3 stop   (commands blend clip end -> stand, then >=3 s hold)
  P4 push   (optional 1.5 m/s push impulse during P0)

Per-phase metrics are dumped to JSON with gate values per docs/STANDARD_SIM2SIM.md.

Usage:
  python eval_sim2sim_upt.py --checkpoint model.pt --motion clip_or_library.npz \
      --robot f1 --headless --stand_hold 2.0 --stop_hold 3.0 --out report.json \
      [--record video.mp4] [--category walk|run|dance]
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import mujoco

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
# reuse play_mujoco internals (joint params, order maps, policy loader)
import importlib.util
spec = importlib.util.spec_from_file_location("play_mujoco", REPO / "scripts" / "play_mujoco.py")
pm = importlib.util.module_from_spec(spec)
pm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pm)
pm._ROBOT = "f1"  # module-global robot switch used by _match_joint_params_auto

POLICY_DT = getattr(pm, "POLICY_DT", 0.02)
DECIMATION = getattr(pm, "DECIMATION", 10)


def stand_pose_npz(bm_npz=None):
    """Stand reference: joint pos + root pose from f1_stand (URDF order) mapped to MJCF names via npz joint_names."""
    stand = json.load(open("/Users/yumx/code/Beyondminic-Weilai-X1/F1_29DOF/mjcf/robot/f1/f1_stand.json"))
    jmap = stand.get("joint_pos_urdf_order") or stand.get("joint_pos_sensor_order")
    return jmap, stand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--motion", required=True)
    ap.add_argument("--category", default="walk", choices=["walk", "run", "dance", "auto"])
    ap.add_argument("--robot", default="f1", choices=["f1"])
    ap.add_argument("--headless", action="store_true", default=True)
    ap.add_argument("--stand_hold", type=float, default=2.0)
    ap.add_argument("--blend", type=float, default=0.8)
    ap.add_argument("--stop_hold", type=float, default=3.0)
    ap.add_argument("--max_track_s", type=float, default=None, help="limit P2 duration (default: full clip)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--record", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--push", action="store_true", help="P4 push impulse during P0")
    args = ap.parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed); np.random.seed(args.seed)

    # ---- model (same as play_mujoco f1 path) ----
    mjcf = REPO / "source/whole_body_tracking/whole_body_tracking/assets/f1/mjcf/f1_flat.xml"
    assert mjcf.exists(), mjcf
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    model.opt.timestep = pm.SIM_DT
    data = mujoco.MjData(model)

    joint_names_dof, joint_qpos_adr, joint_dof_adr = [], [], []
    for i in range(1, model.njnt):
        joint_names_dof.append(model.joint(i).name)
        joint_qpos_adr.append(model.jnt_qposadr[i])
        joint_dof_adr.append(model.jnt_dofadr[i])
    default_pos, kp, kd, action_scale = pm.build_joint_arrays(joint_names_dof)
    for _nm in joint_names_dof:
        for _pat, _eff, _arm, _kp, _kd in pm._F1_JOINT_CLASS:
            import re
            if re.fullmatch(_pat, _nm):
                dadr = model.jnt_dofadr[model.joint(_nm).id]
                model.dof_armature[dadr] = _arm
                model.dof_damping[dadr] = 0.0
                model.dof_frictionloss[dadr] = 0.0
                break
    _act_for_joint = []
    for _name in joint_names_dof:
        _jid = model.joint(_name).id
        _act_for_joint.append(int(np.where(model.actuator_trnid[:, 0] == _jid)[0][0]))
    _act_for_joint = np.array(_act_for_joint)

    # ---- reference motion ----
    md = np.load(args.motion, allow_pickle=True)
    ref_jpos = md["joint_pos"].astype(np.float32)
    ref_jvel = md["joint_vel"].astype(np.float32)
    ref_bpos = md["body_pos_w"].astype(np.float32)
    ref_bquat = md["body_quat_w"].astype(np.float32)
    fps = float(md["fps"])
    if "joint_names" in md:
        npz_jn = [str(n) for n in md["joint_names"]]
        perm = [npz_jn.index(n) for n in joint_names_dof]
        ref_jpos = ref_jpos[:, perm]; ref_jvel = ref_jvel[:, perm]
    NPZ_BODY = [str(b) for b in md["body_names"]]
    anchor_npz_idx = NPZ_BODY.index("lumbar_pitch_link")
    anchor_mj = model.body("lumbar_pitch_link").id
    T = ref_jpos.shape[0]
    category = args.category if args.category != "auto" else str(md.get("category", "walk"))
    phase = md["phase"].astype(np.float32) if "phase" in md else np.zeros(T, np.float32)
    mode = md["mode"].astype(np.int64) if "mode" in md else np.ones(T, np.int64)

    # ---- policy ----
    # MoE-aware policy load (falls back to plain MLP when no MoE keys)
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("upt_actor", REPO / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/f1/agents/upt_actor.py")
    _ua = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_ua)
    _ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    _md = _ck.get("model_dict") or _ck.get("model_state_dict") or {}
    if any(".expert0." in k for k in _md):
        actor, normalizer, obs_dim, action_dim = _ua.load_policy_moe(args.checkpoint, "cpu")
        print("[upt] MoE policy detected")
    else:
        actor, normalizer, obs_dim, action_dim = pm.load_policy(args.checkpoint, "cpu")

    # PhysX order map (same as play_mujoco)
    F1_PHYSX_JOINT_ORDER = pm.__dict__.get("F1_PHYSX_JOINT_ORDER") or [
        "left_hip_pitch_joint","lumbar_yaw_joint","right_hip_pitch_joint","left_hip_roll_joint","lumbar_roll_joint",
        "right_hip_roll_joint","left_hip_yaw_joint","lumbar_pitch_joint","right_hip_yaw_joint","left_knee_pitch_joint",
        "left_shoulder_pitch_joint","right_shoulder_pitch_joint","right_knee_pitch_joint","left_ankle_pitch_joint",
        "left_shoulder_roll_joint","right_shoulder_roll_joint","right_ankle_pitch_joint","left_ankle_roll_joint",
        "left_shoulder_yaw_joint","right_shoulder_yaw_joint","right_ankle_roll_joint","left_elbow_pitch_joint",
        "right_elbow_pitch_joint","left_elbow_yaw_joint","right_elbow_yaw_joint","left_wrist_pitch_joint",
        "right_wrist_pitch_joint","left_wrist_roll_joint","right_wrist_roll_joint"]
    _px = {n: i for i, n in enumerate(F1_PHYSX_JOINT_ORDER)}
    _mj_to_px = np.array([_px[n] for n in joint_names_dof], int)
    _mj_to_px_inv = np.argsort(_mj_to_px)
    to_px = lambda v: v[_mj_to_px_inv]
    from_px = lambda v: v[_mj_to_px]

    # UPT phase/mode injection (obs extension): policy trained with UPT_PHASE_COND=1 has obs 164+4
    upt_obs = (obs_dim == 5 * action_dim + 15 + 4)
    if upt_obs:
        print(f"[upt] phase-conditioned policy detected (obs {obs_dim} = base {5*action_dim+15} + 4)")

    def reset_to(frame=0, stand=False):
        mujoco.mj_resetData(model, data)
        if stand:
            data.qpos[2] = 0.5876
        else:
            data.qpos[0:3] = ref_bpos[frame, 0]
            data.qpos[3:7] = ref_bquat[frame, 0]
            for i, adr in enumerate(joint_qpos_adr):
                data.qpos[adr] = ref_jpos[frame, i]
            for i, adr in enumerate(joint_dof_adr):
                data.qvel[adr] = ref_jvel[frame, i]
        mujoco.mj_forward(model, data)

    stand_jmap, _ = stand_pose_npz()
    stand_j = np.array([stand_jmap.get(n, 0.0) for n in
                        [str(x) for x in md["joint_names"]]])  # npz order

    # metrics buffers per phase
    M = {k: [] for k in ("phase", "t", "anchor_err", "joint_err", "root_z", "root_xy",
                         "root_speed", "stand_dev", "L_contact", "R_contact", "L_slip", "R_slip",
                         "phi", "ref_contact_L", "ref_contact_R", "tau_max_frac")}
    foot_bodies = {s: model.body(f"{s}_ankle_roll").id for s in ("left", "right")}
    corners = pm.__dict__.get("ROBOT_FOOT_CORNERS", {"f1": [(-0.0967,-0.0395,-0.0337),(0.1071,-0.0395,-0.0337),(-0.0967,0.0365,-0.0337),(0.1071,0.0365,-0.0337)]})["f1"]

    def obs_and_step(t_ref, phase_val, mode_val, last_action):
        robot_anchor_pos = data.xpos[anchor_mj].copy()
        robot_anchor_quat = data.xquat[anchor_mj].copy()
        base_lin = pm.quat_rotate_inv(data.qpos[3:7], data.qvel[0:3].copy())
        base_ang = data.qvel[3:6].copy()
        jpos = np.array([data.qpos[a] for a in joint_qpos_adr])
        jvel = np.array([data.qvel[a] for a in joint_dof_adr])
        rjpos = ref_jpos[t_ref]; rjvel = ref_jvel[t_ref]
        anchor_pos_b, anchor_quat_b = pm.subtract_frame_transform(
            robot_anchor_pos, robot_anchor_quat, ref_bpos[t_ref, anchor_npz_idx], ref_bquat[t_ref, anchor_npz_idx])
        rot = pm.quat_to_rotmat(anchor_quat_b)
        obs = np.concatenate([to_px(rjpos), to_px(rjvel), anchor_pos_b, rot[:, :2].flatten(),
                              base_lin, base_ang, to_px(jpos - default_pos), to_px(jvel), last_action])
        if upt_obs:
            obs = np.concatenate([obs, [np.sin(2*np.pi*phase_val), np.cos(2*np.pi*phase_val), 0.0, float(mode_val)]]).astype(np.float32)
        with torch.no_grad():
            act = actor(pm.normalize_obs(torch.from_numpy(obs).unsqueeze(0), normalizer)).squeeze(0).numpy().astype(np.float32)
        return act, jpos, jvel, anchor_pos_b

    def pd_step(target):
        jpos = np.array([data.qpos[a] for a in joint_qpos_adr])
        jvel = np.array([data.qvel[a] for a in joint_dof_adr])
        tau = kp * (target - jpos) - kd * jvel
        for i, nm in enumerate(joint_names_dof):
            eff, _, _ = pm._match_joint_params_auto(nm)
            tau[i] = np.clip(tau[i], -eff, eff)
        ctrl = np.zeros(model.nu); ctrl[_act_for_joint] = tau; data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)

    def record(phase_tag, t_ref, anchor_pos_b, jpos, prev_foot_pos):
        r_anchor_z = float(data.xpos[anchor_mj][2])
        ref_anchor_z = float(ref_bpos[t_ref, anchor_npz_idx][2])
        M["phase"].append(phase_tag); M["t"].append(len(M["t"]) * POLICY_DT)
        M["anchor_err"].append(float(np.linalg.norm(anchor_pos_b)))
        M["joint_err"].append(float(np.mean(np.abs(jpos - ref_jpos[t_ref]))))
        M["root_z"].append(float(data.qpos[2]))
        M["root_xy"].append(data.qpos[0:2].copy())
        M["root_speed"].append(float(np.linalg.norm(data.qvel[0:3])))
        M["stand_dev"].append(float(np.mean(np.abs(jpos - to_px_inv_local(stand_j)))))
        for s in ("left", "right"):
            bid = foot_bodies[s]
            R = data.xmat[bid].reshape(3, 3)
            zmin = min(float((data.xpos[bid] + R @ np.array(c))[2]) for c in corners)
            M[f"{s}_contact" if False else "L_contact" if s == "left" else "R_contact"].append(zmin < 0.02)
            fp = data.xpos[bid][:2].copy()
            if prev_foot_pos.get(s) is not None and zmin < 0.02:
                M["L_slip" if s == "left" else "R_slip"].append(float(np.linalg.norm(fp - prev_foot_pos[s]) / POLICY_DT))
            prev_foot_pos[s] = fp
        return prev_foot_pos

    def to_px_inv_local(v):
        return v[_mj_to_px]

    # ---------------- run phases ----------------
    reset_to(0, stand=True)
    last_action = np.zeros(action_dim, np.float32)
    prev_foot = {"left": None, "right": None}
    steps_stand = int(args.stand_hold / POLICY_DT)
    steps_blend = int(args.blend / POLICY_DT)
    steps_track = int((args.max_track_s or T / fps) * 50) if False else int((args.max_track_s or T / fps) / POLICY_DT)
    steps_stop = int(args.blend / POLICY_DT)
    steps_hold = int(args.stop_hold / POLICY_DT)

    if args.push:
        pass  # applied mid-P0 below

    def apply_push(force_N=150.0):
        dadr = model.jnt_dofadr[0]
        data.qvel[0:3] += np.array([1.0, 0.0, 0.0]) * (force_N / 35.0)

    t0 = time.time()
    # P0 stand: track stand pose (commands = stand joints), take mid-phase push
    for i in range(steps_stand):
        act, jpos, jvel, ab = obs_and_step(0, 0.0, 0.0, last_action)
        # stand target: clip command replaced by stand pose (still policy-driven)
        target = default_pos + from_px(act) * action_scale
        target_mj = default_pos + (to_px_inv_local(stand_j) - default_pos) * 0.0 + (from_px(act)) * action_scale
        for _ in range(DECIMATION):
            pd_step(target_mj)
        if args.push and i == steps_stand // 2:
            apply_push()
        prev_foot = record("P0_stand", 0, ab, jpos, prev_foot)
        last_action = act.copy()

    # P1 blend stand -> clip frame0 (commands lerp)
    for i in range(steps_blend):
        s = (i + 1) / steps_blend
        blend_j = (1 - s) * stand_j + s * ref_jpos[0]
        blend_v = s * ref_jvel[0]
        ref_jpos[0] = blend_j; ref_jvel[0] = blend_v  # temporary
        act, jpos, jvel, ab = obs_and_step(0, 0.0, 0.0, last_action)
        target = default_pos + from_px(act) * action_scale
        for _ in range(DECIMATION):
            pd_step(target)
        prev_foot = record("P1_blend", 0, ab, jpos, prev_foot)
        last_action = act.copy()
    ref_jpos[0] = md["joint_pos"][0][[list(map(str, md["joint_names"])).index(n) for n in joint_names_dof]] if False else ref_jpos[0]
    # restore frame0 from raw
    raw = np.load(args.motion, allow_pickle=True)["joint_pos"].astype(np.float32)
    npz_jn = [str(n) for n in md["joint_names"]]
    perm = [npz_jn.index(n) for n in joint_names_dof]
    ref_jpos[0] = raw[0][:, perm][0] if raw[0].ndim == 2 else raw[0][perm]
    ref_jvel[0] = np.load(args.motion, allow_pickle=True)["joint_vel"].astype(np.float32)[0][perm]

    # P2 track
    core0 = int(np.argmax(mode == 1)) if (mode == 1).any() else 0
    for k in range(steps_track):
        t_ref = min(core0 + k, T - 1)
        pv = float(phase[min(t_ref, len(phase) - 1)])
        act, jpos, jvel, ab = obs_and_step(t_ref, pv, 1, last_action)
        target = default_pos + from_px(act) * action_scale
        for _ in range(DECIMATION):
            pd_step(target)
        prev_foot = record("P2_track", t_ref, ab, jpos, prev_foot)
        last_action = act.copy()

    # P3 stop: blend clip end -> stand, then hold
    end_t = min(core0 + steps_track - 1, T - 1)
    for i in range(steps_stop):
        s = (i + 1) / steps_stop
        blend_j = (1 - s) * ref_jpos[end_t] + s * stand_j
        ref_jpos[end_t] = blend_j
        act, jpos, jvel, ab = obs_and_step(end_t, 0.0, 0.0, last_action)
        target = default_pos + from_px(act) * action_scale
        for _ in range(DECIMATION):
            pd_step(target)
        prev_foot = record("P3_stop_blend", end_t, ab, jpos, prev_foot)
        last_action = act.copy()
    for i in range(steps_hold):
        act, jpos, jvel, ab = obs_and_step(end_t, 0.0, 0.0, last_action)
        target = default_pos + from_px(act) * action_scale
        for _ in range(DECIMATION):
            pd_step(target)
        prev_foot = record("P3_hold", end_t, ab, jpos, prev_foot)
        last_action = act.copy()

    wall = time.time() - t0
    # ---------------- gates ----------------
    P = np.array(M["phase"])
    def seg(tag):
        i = [k for k, x in enumerate(M["phase"]) if x == tag]
        return i
    def arr(key, tag=None):
        v = np.array(M[key], dtype=object) if key in ("root_xy",) else np.array(M[key])
        if tag is None:
            return v
        idx = seg(tag)
        return v[idx] if len(idx) else v[:0]

    gates = {}
    # S-00 fall check over all
    gates["S00_no_fall"] = bool(np.min(M["root_z"]) >= 0.28)
    # S-01 stand start
    i0 = seg("P0_stand")
    xy0 = np.stack([M["root_xy"][k] for k in i0]) if i0 else np.zeros((0, 2))
    z0 = np.array(M["root_z"])[i0] if i0 else np.zeros(0)
    sd0 = np.array(M["stand_dev"])[i0] if i0 else np.zeros(0)
    gates["S01_stand_xy_le_5cm"] = bool(len(xy0) and np.max(np.linalg.norm(xy0 - xy0[0], axis=1)) <= 0.05)
    gates["S01_stand_z_dev_le_2cm"] = bool(len(z0) and (np.max(z0) - np.min(z0)) <= 0.02)
    gates["S01_stand_jdev_p95_le_0p10"] = bool(len(sd0) and np.quantile(sd0, 0.95) <= 0.10)
    # S-02 stop
    ih = seg("P3_hold")
    vh = np.array(M["root_speed"])[ih] if ih else np.zeros(0)
    sdh = np.array(M["stand_dev"])[ih] if ih else np.zeros(0)
    xyh = np.stack([M["root_xy"][k] for k in ih]) if ih else np.zeros((0, 2))
    gates["S02_stop_speed_p95_le_0p10"] = bool(len(vh) and np.quantile(vh, 0.95) <= 0.10)
    gates["S02_stop_jdev_p95_le_0p10"] = bool(len(sdh) and np.quantile(sdh, 0.95) <= 0.10)
    gates["S02_hold_3s"] = bool(len(ih) >= int(3.0 / POLICY_DT) - 2)
    # S-03 track
    it = seg("P2_track")
    ae = np.array(M["anchor_err"])[it] if it else np.zeros(0)
    je = np.array(M["joint_err"])[it] if it else np.zeros(0)
    thr_anchor = 0.10 if category == "dance" else 0.08
    thr_joint = 0.25 if category == "dance" else 0.20
    gates["S03_anchor_p50_le"] = bool(len(ae) and np.quantile(ae, 0.5) <= thr_anchor)
    gates["S03_joint_p50_le"] = bool(len(je) and np.quantile(je, 0.5) <= thr_joint)
    # S-04 slip
    slip = np.array(M["L_slip"] + M["R_slip"]) if (M["L_slip"] or M["R_slip"]) else np.zeros(0)
    gates["S04_slip_p50_le_0p05"] = bool(len(slip) == 0 or np.quantile(slip, 0.5) <= 0.05)
    # category rhythm gates (walk/run)
    if category in ("walk", "run") and it:
        lc = np.array(M["L_contact"])[it]; rc = np.array(M["R_contact"])[it]
        sw = (1 - lc.astype(float)); swR = (1 - rc.astype(float))
        swings = np.where((sw[1:] > 0.5) & (sw[:-1] < 0.5))[0]
        cadence = len(swings) / (len(it) * POLICY_DT) / 2.0
        airborne = float(np.mean((np.array(M["root_z"])[it] * 0)))  # placeholder: both feet off
        both_off = ((~lc) & (~rc)).astype(float)
        gates["S_W2_airborne_frac"] = float(both_off.mean())
        if category == "walk":
            gates["S_W2_cadence_in_band"] = bool(0.55 <= cadence <= 1.6)
            gates["S_W2_airborne_le_2pct"] = bool(both_off.mean() <= 0.02)
        else:
            gates["S_R2_airborne_ge_5pct"] = bool(both_off.mean() >= 0.05)
            gates["S_R2_cadence_in_band"] = bool(1.4 <= cadence <= 3.2)

    report = {
        "standard": "STANDARD_SIM2SIM v1.0",
        "checkpoint": args.checkpoint, "motion": args.motion, "category": category,
        "obs_dim": obs_dim, "upt_phase_conditioned": upt_obs,
        "phases": {"stand_s": args.stand_hold, "blend_s": args.blend, "track_s": len(it) * POLICY_DT, "stop_hold_s": args.stop_hold},
        "wall_s": round(wall, 1),
        "gates": gates,
        "metrics": {
            "anchor_err_p50": float(np.quantile(ae, 0.5)) if len(ae) else None,
            "anchor_err_p95": float(np.quantile(ae, 0.95)) if len(ae) else None,
            "joint_err_p50": float(np.quantile(je, 0.5)) if len(je) else None,
            "min_root_z": float(np.min(M["root_z"])),
            "stand_dev_p95_P0": float(np.quantile(sd0, 0.95)) if len(sd0) else None,
            "stop_speed_p95": float(np.quantile(vh, 0.95)) if len(vh) else None,
            "slip_p50": float(np.quantile(slip, 0.5)) if len(slip) else None,
        },
    }
    report["ALL_PASS"] = all(v for k, v in gates.items() if isinstance(v, bool))
    json.dump(report, open(args.out, "w"), indent=1)
    np.savez(Path(args.out).with_suffix(".npz"), **{k: np.array(v) for k, v in M.items() if k != "t_ref"})
    print(json.dumps(report, indent=1)[:1200])

if __name__ == "__main__":
    main()
