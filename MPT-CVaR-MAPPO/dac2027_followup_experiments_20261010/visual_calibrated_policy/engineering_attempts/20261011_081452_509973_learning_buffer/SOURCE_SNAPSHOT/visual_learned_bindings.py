"""Frozen historical model bindings, lazy construction after real CUDA gate."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'results_qla_training_repair_pilot_20260930/source_snapshot'
EXPECTED = {'models.py': '770d4791b4a1ddc86e89469d49dc1aeb84f69493b5c43b8b327509bc8a528b85',
            'config.py': 'ecb80a3c7455b2ce72cefbce72336ea02d67dcf17250adce87507bb32a62fbfd',
            'paper_policy_baselines.py': '08481b36d74964f8118f341f0d665ddec13ba986993c0d45ab5b5e4311fa25e4'}


def source_bindings():
    binding = {}
    for name, expected in EXPECTED.items():
        path = SOURCE / 'mappo_gp_sf_dsr' / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise RuntimeError('Historical model source changed; no silent replacement')
        binding[str(path)] = digest
    return binding


@dataclass(frozen=True)
class VisualPolicyDimensions:
    n_users: int
    n_edges: int
    actor_obs_dim: int
    critic_obs_dim: int
    max_supported_users: int
    include_feedback_age: bool = True
    feedback_age_norm: float = .5


def build_cuda_model(method, dimensions, learner_kwargs, seed):
    """Prepared entry only. Never exercised by CPU engineering checks.

    A fresh independent complete entry review and final protocol are still
    required before callers may use this factory for formal policy work.
    """
    from windows_runtime import initialize
    initialize()
    source_bindings()
    if method not in ('QLA-MAPPO', 'PPO-MV-A'):
        raise ValueError('Unknown learning method; no substitute or dummy model')
    from qla_environment import guard, witness
    spec = guard()
    expected_python = Path(spec['isolated_environment']) / 'Scripts/python.exe'
    if Path(sys.executable).resolve() != expected_python.resolve():
        raise RuntimeError('Main model process must use the declared isolated environment')
    witness()  # original resource gate + separate-process seeded CUDA kernel
    if dimensions.actor_obs_dim != 3 * dimensions.n_edges + 7:
        raise ValueError('Matched structured actor layout required')
    loaded = sys.modules.get('mappo_gp_sf_dsr')
    if loaded is not None and Path(loaded.__file__).resolve().parent != (SOURCE / 'mappo_gp_sf_dsr').resolve():
        raise RuntimeError('Different source package already loaded; use a fresh isolated process')
    sys.path.insert(0, str(SOURCE))
    import torch
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.manual_seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no CPU fallback')
    if method == 'QLA-MAPPO':
        config = importlib.import_module('mappo_gp_sf_dsr.config')
        models = importlib.import_module('mappo_gp_sf_dsr.models')
        kwargs = dict(learner_kwargs)
        if kwargs.get('edgewise_price_actor', True) is not True or kwargs.get('lcc_price_logit_bias', .5) != .5:
            raise ValueError('Full visual QLA requires unchanged edge-wise structure and bias0.5')
        kwargs.update(edgewise_price_actor=True, lcc_price_logit_bias=.5)
        return models.MAPPOPolicy(dimensions, config.MAPPOConfig(**kwargs), seed=seed, device='cuda')
    if method == 'PPO-MV-A':
        baseline = importlib.import_module('mappo_gp_sf_dsr.paper_policy_baselines')
        return baseline.SharedGaussianPPO(dimensions.actor_obs_dim, dimensions.n_edges).to('cuda')
    raise ValueError('Unknown learning method; no substitute or dummy model')
