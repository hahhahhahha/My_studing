"""Pure, separately testable interventions for the paired training pilot."""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import numpy as np


def gae_with_bootstrap(arrays, gamma, gae_lambda, bootstrap_values):
    """Bootstrap a continuing/time-limit rollout; true terminals still mask V."""
    rewards, values = arrays["reward"], arrays["value"]
    dones, users, slots = arrays["done"], arrays["user_id"], arrays["slot"]
    advantages = np.zeros_like(rewards, dtype=np.float32)
    returns = np.zeros_like(rewards, dtype=np.float32)
    for user in np.unique(users):
        idxs = np.flatnonzero(users == user)
        idxs = idxs[np.argsort(slots[idxs], kind="stable")]
        gae = 0.0
        for pos in reversed(range(len(idxs))):
            idx = idxs[pos]
            nxt = values[idxs[pos + 1]] if pos + 1 < len(idxs) else bootstrap_values[int(user)]
            nonterminal = 1.0 - dones[idx]
            delta = rewards[idx] + gamma * nxt * nonterminal - values[idx]
            gae = delta + gamma * gae_lambda * nonterminal * gae
            advantages[idx], returns[idx] = gae, gae + values[idx]
    return advantages, returns


def mask_action_features(obs, n_edges, system_dim):
    """Zero only six current-portfolio columns, preserving network/init shape."""
    obs = np.asarray(obs, dtype=np.float32).copy()
    start = 5 * int(n_edges) + int(system_dim)
    if obs.ndim != 2 or start + 6 > obs.shape[1]:
        raise ValueError("Unexpected critic observation layout")
    obs[:, start:start + 6] = 0.0
    return obs


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_digest(root):
    digest = hashlib.sha256()
    for path in sorted((Path(root) / "mappo_gp_sf_dsr").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def instrument_train_source(source):
    """Insert a rollout-boundary hook; fail closed on any upstream code drift."""
    replacements = {
        "            last_update = policy.update(buffer)":
        "            last_update = repair_update(policy, buffer, env, args, rewarder, previous_train_portfolios, train_lcc_obs_state)",
        "        last_update = policy.update(buffer)":
        "        last_update = repair_update(policy, buffer, env, args, rewarder, previous_train_portfolios, train_lcc_obs_state)",
    }
    for old, new in replacements.items():
        # Include the leading newline to distinguish the two indentation levels.
        if source.count("\n" + old + "\n") != 1:
            raise RuntimeError("Training update anchor changed: " + old.strip())
        source = source.replace("\n" + old + "\n", "\n" + new + "\n", 1)
    ast.parse(source)
    return source
