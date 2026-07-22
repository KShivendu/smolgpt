"""Group Relative Policy Optimization (GRPO). Ports `src/grpo.rs`.

For each prompt we sample a *group* of completions, score them, and compute a
group-relative advantage (reward minus the group mean, over the group std). Two
modes:

- ``lite``: REINFORCE with the group-relative advantage. No clipping, no KL.
- ``full``: PPO-style importance-ratio clipping + a KL-to-reference penalty
  (k3 estimator) + K mini-epochs per group.

Saves to a ``-grpo`` / ``-grpo-full`` checkpoint, never overwriting the base.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from .eval import Problem
from .tokenizer import Tokenizer


@dataclass
class GrpoConfig:
    mode: str = "lite"  # "lite" | "full"
    group_size: int = 8
    rounds: int = 30
    temperature: float = 1.0
    lr: float = 1e-3
    max_answer_len: int = 8
    clip_eps: float = 0.2  # full only
    kl_coef: float = 0.02  # full only
    mini_epochs: int = 2  # full only


@torch.no_grad()
def _rollout(model, prompt_ids, cfg, stop_id, device, generator):
    """Sample one completion; return (completion_ids, old_logprobs)."""
    model.eval()
    block_size = model.block_size
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    comp: list[int] = []
    old_lp: list[float] = []
    for _ in range(cfg.max_answer_len):
        logits = model(idx[:, -block_size:])[0, -1, :] / cfg.temperature
        logp = F.log_softmax(logits, dim=-1)
        probs = logp.exp()
        next_id = int(torch.multinomial(probs, 1, generator=generator))
        comp.append(next_id)
        old_lp.append(float(logp[next_id]))
        if stop_id is not None and next_id == stop_id:
            break
        idx = torch.cat([idx, torch.tensor([[next_id]], device=device)], dim=1)
    return comp, old_lp


def _seq_logprobs(model, prompt_ids: list[int], comp_ids: list[int], device):
    """Per-completion-token log-probs under the current model (with grad)."""
    block_size = model.block_size
    full = prompt_ids + comp_ids
    full = full[-(block_size + 1) :] if len(full) > block_size + 1 else full
    x = torch.tensor([full[:-1]], dtype=torch.long, device=device)
    targets = torch.tensor(full[1:], dtype=torch.long, device=device)
    logits = model(x)[0]  # (T, vocab)
    logp = F.log_softmax(logits, dim=-1)
    chosen = logp[torch.arange(len(targets)), targets]
    # Keep only the completion-token positions (the tail).
    return chosen[-len(comp_ids) :]


def _reward(tokenizer, comp_ids, problem: Problem) -> float:
    got = tokenizer.decode(comp_ids).split("\n", 1)[0].strip()
    return 1.0 if got == problem.answer else 0.0


def grpo_train(
    model,
    tokenizer: Tokenizer,
    problems: list[Problem],
    model_path: str | Path,
    cfg: GrpoConfig,
    device: str = "cpu",
    generator: torch.Generator | None = None,
) -> None:
    stop_ids = tokenizer.encode("\n")
    stop_id = stop_ids[0] if stop_ids else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    ref_model = copy.deepcopy(model) if cfg.mode == "full" else None
    if ref_model is not None:
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)

    for r in range(1, cfg.rounds + 1):
        round_reward = 0.0
        round_groups = 0
        for problem in problems:
            prompt_ids = tokenizer.encode(problem.prompt)
            # 1) Sample a group of completions and score them.
            group = []
            rewards = []
            for _ in range(cfg.group_size):
                comp, old_lp = _rollout(
                    model, prompt_ids, cfg, stop_id, device, generator
                )
                group.append((comp, old_lp))
                rewards.append(_reward(tokenizer, comp, problem))
            round_reward += sum(rewards)
            round_groups += 1

            # 2) Group-relative advantage (skip degenerate all-equal groups).
            rt = torch.tensor(rewards)
            if rt.std() < 1e-8:
                continue
            adv = (rt - rt.mean()) / (rt.std() + 1e-8)

            # 3) Policy update.
            epochs = cfg.mini_epochs if cfg.mode == "full" else 1
            for _ in range(epochs):
                optimizer.zero_grad(set_to_none=True)
                total_loss = torch.zeros((), device=device)
                n_tokens = 0
                for (comp, old_lp), a in zip(group, adv.tolist()):
                    if not comp:
                        continue
                    new_lp = _seq_logprobs(model, prompt_ids, comp, device)
                    if cfg.mode == "lite":
                        # REINFORCE: maximize advantage-weighted log-prob.
                        total_loss = total_loss + (-a * new_lp.sum())
                    else:
                        old = torch.tensor(old_lp[: len(new_lp)], device=device)
                        ratio = (new_lp - old).exp()
                        unclipped = ratio * a
                        clipped = torch.clamp(
                            ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps
                        ) * a
                        ppo = -torch.min(unclipped, clipped).sum()
                        # k3 KL estimator to the frozen reference.
                        with torch.no_grad():
                            ref_lp = _seq_logprobs(
                                ref_model, prompt_ids, comp, device
                            )
                        diff = ref_lp - new_lp
                        kl = (diff.exp() - diff - 1).sum()
                        total_loss = total_loss + ppo + cfg.kl_coef * kl
                    n_tokens += len(new_lp)
                if n_tokens == 0:
                    continue
                (total_loss / n_tokens).backward()
                optimizer.step()

        mean_r = round_reward / (round_groups * cfg.group_size) if round_groups else 0.0
        print(f"GRPO[{cfg.mode}] round {r}/{cfg.rounds}: mean reward = {mean_r:.3f}")
        torch.save(model.state_dict(), model_path)
    print(f"GRPO model saved to {model_path}")
