from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MAPPOTransition:
    slot: int
    user_id: int
    actor_obs: np.ndarray
    critic_obs: np.ndarray
    latent_action: np.ndarray
    log_prob: float
    value: float
    reward: float
    done: bool
    aux_target: np.ndarray | None = None
    aux_weight: float = 0.0


class MAPPOBuffer:
    def __init__(self, max_size: int = 20_000) -> None:
        self.max_size = max_size
        self.data: list[MAPPOTransition] = []

    def __len__(self) -> int:
        return len(self.data)

    def clear(self) -> None:
        self.data.clear()

    def add(self, transition: MAPPOTransition) -> None:
        self.data.append(transition)
        if len(self.data) > self.max_size:
            self.data = self.data[-self.max_size :]

    def add_batch(
        self,
        slot: int,
        actor_obs: np.ndarray,
        critic_obs: np.ndarray,
        latent_actions: np.ndarray,
        log_probs: np.ndarray,
        values: np.ndarray,
        rewards: np.ndarray,
        aux_targets: np.ndarray | None = None,
        aux_weights: np.ndarray | None = None,
        dones: np.ndarray | None = None,
    ) -> None:
        n_users = actor_obs.shape[0]
        if dones is None:
            dones = np.zeros(n_users, dtype=np.float32)
        if aux_weights is None:
            aux_weights = np.zeros(n_users, dtype=np.float32)
        for user_id in range(n_users):
            self.add(
                MAPPOTransition(
                    slot=slot,
                    user_id=user_id,
                    actor_obs=actor_obs[user_id].astype(np.float32),
                    critic_obs=critic_obs[user_id].astype(np.float32),
                    latent_action=latent_actions[user_id].astype(np.float32),
                    log_prob=float(log_probs[user_id]),
                    value=float(values[user_id]),
                    reward=float(rewards[user_id]),
                    aux_target=(
                        aux_targets[user_id].astype(np.float32)
                        if aux_targets is not None
                        else None
                    ),
                    aux_weight=float(aux_weights[user_id]),
                    done=bool(dones[user_id]),
                )
            )

    def arrays(self) -> dict[str, np.ndarray]:
        latent_actions = np.stack([tr.latent_action for tr in self.data]).astype(np.float32)
        aux_dim = max(1, latent_actions.shape[1] - 1)
        aux_targets = np.stack(
            [
                tr.aux_target.astype(np.float32)
                if tr.aux_target is not None
                else np.zeros(aux_dim, dtype=np.float32)
                for tr in self.data
            ]
        ).astype(np.float32)
        return {
            "slot": np.asarray([tr.slot for tr in self.data], dtype=np.int64),
            "user_id": np.asarray([tr.user_id for tr in self.data], dtype=np.int64),
            "actor_obs": np.stack([tr.actor_obs for tr in self.data]).astype(np.float32),
            "critic_obs": np.stack([tr.critic_obs for tr in self.data]).astype(np.float32),
            "latent_action": latent_actions,
            "log_prob": np.asarray([tr.log_prob for tr in self.data], dtype=np.float32),
            "value": np.asarray([tr.value for tr in self.data], dtype=np.float32),
            "reward": np.asarray([tr.reward for tr in self.data], dtype=np.float32),
            "aux_target": aux_targets,
            "aux_weight": np.asarray([tr.aux_weight for tr in self.data], dtype=np.float32),
            "done": np.asarray([tr.done for tr in self.data], dtype=np.float32),
        }
