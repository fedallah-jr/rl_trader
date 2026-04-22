"""CleanRL-style single-file PPO.

`train_ppo` takes three configs (its own hparams, an env config, and a policy
config) so different architectures + envs compose freely. Synchronous vec envs,
GAE advantages, clipped-surrogate policy loss, entropy bonus, value loss,
optional LR anneal.
"""

from __future__ import annotations

import copy
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .. import maybe_compile_forward, resolve_device
from ..architectures.factored_attention import ActorCritic, PolicyConfig
from ..envs.multi_asset import EnvConfig
from ..eval.validation import evaluate_policy
from ._rollout import SyncVecEnv


# ---- config --------------------------------------------------------------

@dataclass
class PPOConfig:
    # rollout
    n_envs: int = 8
    n_steps: int = 128

    # optimisation
    total_timesteps: int = 1_000_000
    lr: float = 3e-4
    anneal_lr: bool = True

    # PPO
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 4
    minibatch_size: int = 256
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    normalize_advantage: bool = True

    # infra
    device: str = "auto"         # "auto" -> cuda if available else cpu
    torch_threads: int = 2
    # Compile the policy forward with torch.compile. Only applied on CUDA
    # (CPU compile warmup dwarfs any CPU-side speedup for short runs).
    torch_compile: bool = True
    seed: int = 0
    log_interval: int = 1
    ckpt_interval: int = 50
    ckpt_dir: str = "checkpoints/ppo"

    # validation during training (0 disables)
    eval_interval_steps: int = 50_000    # every ~50 iterations at defaults
    eval_episodes: int = 4               # fixed starts inside the val range
    eval_episode_length: int = 10_080    # 1 week of 1m bars per eval episode


# ---- GAE ----------------------------------------------------------------

def compute_gae(
    reward_buf:      torch.Tensor,   # [T, B]
    value_buf:       torch.Tensor,   # [T, B]
    term_buf:        torch.Tensor,   # [T, B] 1.0 on true terminal (V=0)
    trunc_buf:       torch.Tensor,   # [T, B] 1.0 on time-limit truncation
    trunc_value_buf: torch.Tensor,   # [T, B] V(final_obs) at truncation; 0 else
    bootstrap_T:     torch.Tensor,   # [B] V at rollout boundary (already accounts
                                     #     for trunc envs via caller's torch.where)
    gamma:      float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalised Advantage Estimation with proper term/trunc handling.

    Two masks, not one:
      * `non_term` = 1 − term        — governs the δ_t bootstrap. Truncation
        does NOT zero the bootstrap; V(s') still exists, just via final_obs.
      * `non_bound` = 1 − (term ∨ trunc)  — governs the GAE recursion carry.
        ANY episode boundary must stop advantage propagation, because step
        t+1 belongs to a freshly-reset (different) episode whose advantage
        is independent of step t.

    Conflating the two (the common CleanRL-style single `done` flag) silently
    mislabels every truncation as terminal, biasing V(s) near episode ends.
    """
    T, B = reward_buf.shape
    advantages = torch.zeros_like(reward_buf)
    last_gae = torch.zeros(B, device=reward_buf.device, dtype=reward_buf.dtype)
    for t in reversed(range(T)):
        non_term  = 1.0 - term_buf[t]
        non_bound = 1.0 - torch.maximum(term_buf[t], trunc_buf[t])
        if t == T - 1:
            nv = bootstrap_T
        else:
            nv = torch.where(
                trunc_buf[t].bool(),
                trunc_value_buf[t],
                value_buf[t + 1],
            )
        delta = reward_buf[t] + gamma * nv * non_term - value_buf[t]
        last_gae = delta + gamma * gae_lambda * non_bound * last_gae
        advantages[t] = last_gae
    returns = advantages + value_buf
    return advantages, returns


# ---- training -----------------------------------------------------------

def train_ppo(
    cfg: PPOConfig,
    env_cfg: EnvConfig,
    policy_cfg: PolicyConfig,
    val_env_cfg: EnvConfig | None = None,
) -> ActorCritic:
    """Train PPO. If `val_env_cfg` is given, evaluate periodically on that env
    and keep the best-performing weights (by mean net-PnL across fixed val
    rollouts). Best weights are saved to `<ckpt_dir>/best.pt` both on-improvement
    and at the end of training.
    """
    device = resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if device.type == "cpu" and cfg.torch_threads:
        torch.set_num_threads(cfg.torch_threads)
        os.environ.setdefault("OMP_NUM_THREADS", str(cfg.torch_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(cfg.torch_threads))

    envs = SyncVecEnv(cfg.n_envs, env_cfg, base_seed=cfg.seed)
    obs_np = envs.reset()

    # best-model tracking (only if we have a val env)
    do_val = val_env_cfg is not None and cfg.eval_interval_steps > 0
    last_eval_step: int = 0

    net = ActorCritic(policy_cfg).to(device)
    opt = optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5)
    compiled = maybe_compile_forward(net, device, enable=cfg.torch_compile)
    print(
        f"policy params: {net.num_params():,}   device: {device}"
        f"   torch.compile: {'on' if compiled else 'off'}"
    )

    # ---- val-eval helper (closure over cfg / val_env_cfg / best) ---------
    best = {"metric": float("-inf"), "state": None, "iter": -1, "step": -1,
            "metrics": None}

    def _run_val(it_idx: int, gstep: int) -> None:
        m = evaluate_policy(
            net, val_env_cfg,
            n_episodes=cfg.eval_episodes,
            episode_length=cfg.eval_episode_length,
            device=device,
        )
        print(
            f"  VAL @ step {gstep:>9,}  "
            f"mean_net_pnl {m['mean_net_pnl']:+.4f}  "
            f"std {m['std_net_pnl']:.4f}  "
            f"min {m['min_net_pnl']:+.4f}  max {m['max_net_pnl']:+.4f}  "
            f"(episodes={m['n_episodes']} × {m['episode_length']} steps)"
        )
        if m["mean_net_pnl"] > best["metric"]:
            best["metric"]  = m["mean_net_pnl"]
            best["state"]   = copy.deepcopy(net.state_dict())
            best["iter"]    = it_idx
            best["step"]    = gstep
            best["metrics"] = m
            bk = Path(cfg.ckpt_dir) / "best.pt"
            bk.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "iter": it_idx + 1,
                    "global_step": gstep,
                    "model": best["state"],
                    "ppo_cfg":     asdict(cfg),
                    "env_cfg":     asdict(env_cfg),
                    "val_env_cfg": asdict(val_env_cfg) if val_env_cfg is not None else None,
                    "policy_cfg":  asdict(policy_cfg),
                    "val_metrics": m,
                    "val_metric_key": "mean_net_pnl",
                },
                bk,
            )
            print(f"  → new best, saved {bk}")

    # rollout buffers
    B, T = cfg.n_envs, cfg.n_steps
    pc = policy_cfg
    market_buf       = torch.zeros(T, B, pc.window, pc.n_symbols, pc.n_features, device=device)
    account_buf      = torch.zeros(T, B, pc.n_symbols, pc.n_account, device=device)
    globals_buf      = torch.zeros(T, B, pc.n_globals, device=device)
    actions_buf      = torch.zeros(T, B, pc.n_symbols, dtype=torch.int64, device=device)
    logp_buf         = torch.zeros(T, B, device=device)
    reward_buf       = torch.zeros(T, B, device=device)
    value_buf        = torch.zeros(T, B, device=device)
    # Split term/trunc + V(final_obs) for correct truncation bootstrap in GAE.
    term_buf         = torch.zeros(T, B, device=device)
    trunc_buf        = torch.zeros(T, B, device=device)
    trunc_value_buf  = torch.zeros(T, B, device=device)

    n_iters = max(1, cfg.total_timesteps // (B * T))
    global_step = 0
    t0 = time.time()

    for it in range(n_iters):
        if cfg.anneal_lr:
            frac = 1.0 - it / n_iters
            for g in opt.param_groups:
                g["lr"] = cfg.lr * frac

        # ---- rollout --------------------------------------------------
        for step in range(T):
            global_step += B

            market_t  = torch.as_tensor(obs_np["market"],  device=device)
            account_t = torch.as_tensor(obs_np["account"], device=device)
            globals_t = torch.as_tensor(obs_np["globals"], device=device)

            market_buf[step]  = market_t
            account_buf[step] = account_t
            globals_buf[step] = globals_t

            with torch.no_grad():
                action, log_prob, value = net.act(market_t, account_t, globals_t)
            actions_buf[step] = action
            logp_buf[step]    = log_prob
            value_buf[step]   = value

            obs_np, r, term_np, trunc_np, final_obs_list = envs.step(action.cpu().numpy())
            reward_buf[step] = torch.as_tensor(r,        device=device)
            term_buf[step]   = torch.as_tensor(term_np,  device=device)
            trunc_buf[step]  = torch.as_tensor(trunc_np, device=device)

            # For envs that truncated (not terminated) this step, compute
            # V(final_obs) in one batched forward and stash. This is the
            # correct bootstrap target for GAE on truncations; without it
            # the learner treats time-limits as terminal (zero future), which
            # is the common CleanRL-style bug.
            trunc_idx_np = np.where(trunc_np > 0.0)[0]
            if trunc_idx_np.size:
                fm = np.stack([final_obs_list[i]["market"]  for i in trunc_idx_np])
                fa = np.stack([final_obs_list[i]["account"] for i in trunc_idx_np])
                fg = np.stack([final_obs_list[i]["globals"] for i in trunc_idx_np])
                with torch.no_grad():
                    _, v_final = net.forward(
                        torch.as_tensor(fm, device=device),
                        torch.as_tensor(fa, device=device),
                        torch.as_tensor(fg, device=device),
                    )
                trunc_idx = torch.as_tensor(trunc_idx_np, device=device, dtype=torch.long)
                trunc_value_buf[step].index_copy_(0, trunc_idx, v_final)

        # ---- rollout-boundary bootstrap -------------------------------
        # `obs_np` now holds the state we'll start iter it+1 from: the
        # continuation obs for envs that didn't end, or the post-reset obs
        # for envs that did. V(obs_np) is correct for the former; for envs
        # truncated at T-1 we must override with V(final_obs) stashed above.
        with torch.no_grad():
            market_t  = torch.as_tensor(obs_np["market"],  device=device)
            account_t = torch.as_tensor(obs_np["account"], device=device)
            globals_t = torch.as_tensor(obs_np["globals"], device=device)
            next_value = net.forward(market_t, account_t, globals_t)[1]
        bootstrap_T = torch.where(
            trunc_buf[T - 1].bool(),
            trunc_value_buf[T - 1],
            next_value,
        )

        advantages, returns = compute_gae(
            reward_buf, value_buf,
            term_buf, trunc_buf, trunc_value_buf,
            bootstrap_T,
            cfg.gamma, cfg.gae_lambda,
        )

        # ---- flatten & update ----------------------------------------
        bs = B * T
        b_market  = market_buf.reshape(bs, pc.window, pc.n_symbols, pc.n_features)
        b_account = account_buf.reshape(bs, pc.n_symbols, pc.n_account)
        b_globals = globals_buf.reshape(bs, pc.n_globals)
        b_actions = actions_buf.reshape(bs, pc.n_symbols)
        b_logp    = logp_buf.reshape(bs)
        b_adv     = advantages.reshape(bs)
        b_ret     = returns.reshape(bs)
        b_val     = value_buf.reshape(bs)

        inds = np.arange(bs)
        pg_losses, v_losses, ent_vals, clip_fracs, kls = [], [], [], [], []
        for _ in range(cfg.update_epochs):
            np.random.shuffle(inds)
            for start in range(0, bs, cfg.minibatch_size):
                mb = torch.as_tensor(inds[start:start + cfg.minibatch_size], device=device)

                new_lp, ent, new_v = net.evaluate(
                    b_market[mb], b_account[mb], b_globals[mb], b_actions[mb]
                )

                log_ratio = new_lp - b_logp[mb]
                ratio = log_ratio.exp()

                adv = b_adv[mb]
                if cfg.normalize_advantage and adv.numel() > 1:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                pg1 = -adv * ratio
                pg2 = -adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()

                v_loss = 0.5 * (new_v - b_ret[mb]).pow(2).mean()
                ent_loss = ent.mean()

                loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent_loss

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                opt.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()

                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                ent_vals.append(ent_loss.item())
                clip_fracs.append(clip_frac)
                kls.append(approx_kl)

        # ---- logging -------------------------------------------------
        if it % cfg.log_interval == 0:
            ep_rew = (np.mean(envs.recent_ep_rewards) if envs.recent_ep_rewards else float("nan"))
            ep_len = (np.mean(envs.recent_ep_lengths) if envs.recent_ep_lengths else float("nan"))
            ep_pnl = (np.mean(envs.recent_ep_pnls)    if envs.recent_ep_pnls    else float("nan"))
            term_rate = (np.mean(envs.recent_ep_term) if envs.recent_ep_term    else float("nan"))
            ev = 1.0 - (b_ret - b_val).var().item() / (b_ret.var().item() + 1e-8)
            sps = global_step / max(1e-6, time.time() - t0)
            print(
                f"it {it:>4}/{n_iters}  step {global_step:>9,}  sps {sps:>5.0f}  "
                f"ep_rew {ep_rew:>+7.4f}  ep_pnl {ep_pnl:>+7.4f}  "
                f"ep_len {ep_len:>5.0f}  term% {term_rate:>.2f}  "
                f"pg {np.mean(pg_losses):>+.4f}  v {np.mean(v_losses):>.5f}  "
                f"ent {np.mean(ent_vals):>.3f}  kl {np.mean(kls):>.4f}  "
                f"clip {np.mean(clip_fracs):>.3f}  ev {ev:>+.3f}  "
                f"lr {opt.param_groups[0]['lr']:.1e}"
            )

        # ---- checkpoint ----------------------------------------------
        if cfg.ckpt_interval > 0 and (it + 1) % cfg.ckpt_interval == 0:
            p = Path(cfg.ckpt_dir)
            p.mkdir(parents=True, exist_ok=True)
            ck = p / f"iter{it+1:05d}.pt"
            torch.save(
                {
                    "iter": it + 1,
                    "global_step": global_step,
                    "model": net.state_dict(),
                    "opt": opt.state_dict(),
                    "ppo_cfg":    asdict(cfg),
                    "env_cfg":    asdict(env_cfg),
                    "policy_cfg": asdict(policy_cfg),
                },
                ck,
            )
            print(f"  saved {ck}")

        # ---- periodic val eval --------------------------------------
        if do_val and (global_step - last_eval_step) >= cfg.eval_interval_steps:
            _run_val(it_idx=it, gstep=global_step)
            last_eval_step = global_step

    # final val eval at end-of-training (so we don't miss a last-iter improvement)
    if do_val and global_step != last_eval_step:
        _run_val(it_idx=n_iters - 1, gstep=global_step)

    # restore best weights into the returned net
    if do_val and best["state"] is not None:
        print(
            f"best val mean_net_pnl {best['metric']:+.4f} "
            f"at iter {best['iter']+1} / step {best['step']:,}"
        )
        net.load_state_dict(best["state"])

    return net
