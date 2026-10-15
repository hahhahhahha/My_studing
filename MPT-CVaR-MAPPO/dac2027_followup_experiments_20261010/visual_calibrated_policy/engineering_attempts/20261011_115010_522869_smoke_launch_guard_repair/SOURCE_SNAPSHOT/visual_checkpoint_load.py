"""Actual, hash-bound own checkpoint loading; no model or load in checks."""
from __future__ import annotations

import copy
import importlib
from pathlib import Path

from visual_episode_collector import require_actual_cuda_controller
from visual_learned_bindings import source_bindings
from visual_neural_controller import NeuralVisualController
from visual_training_coordinator import actual_parameter_sha256
from visual_training_transactions import validation_checkpoint_reference, require_sha


def load_own_snapshot_in_verified_cuda_process(reference_controller, checkpoint_path,
        descriptor, manifest_sha256, expected_parameter_sha256, initialization_seed):
    """Source-only until actual CUDA smoke runs.

    Clone construction uses the ALREADY verified same-process actual class,
    not a second GPU-free test that mistakes our own allocation for competing
    work. No optimizer/H/MV/RNG training state is restored; not a resume API.
    Caller must have independently checked launch resources before creating
    reference_controller. Only hash-verified own local model files are loaded.
    """
    require_actual_cuda_controller(reference_controller)
    source_bindings()
    if (descriptor['method'] != reference_controller.method or type(initialization_seed) is not int
            or initialization_seed < 0):
        raise ValueError('Matching actual method and declared fresh initialization seed required')
    checked = validation_checkpoint_reference(checkpoint_path, descriptor, manifest_sha256, 'validation')
    model_path = Path(checkpoint_path).resolve()/'MODEL_STATE.pt'
    torch = reference_controller._torch
    if reference_controller.method == 'QLA-MAPPO':
        module = importlib.import_module('mappo_gp_sf_dsr.models')
        model = module.MAPPOPolicy(copy.deepcopy(reference_controller.model.sys_cfg),
            copy.deepcopy(reference_controller.model.cfg), seed=initialization_seed, device='cuda')
        # Original loader reads only this verified self-generated local file.
        model.load_checkpoint(model_path, actor_only=False)
    else:
        module = importlib.import_module('mappo_gp_sf_dsr.paper_policy_baselines')
        old = reference_controller.model
        actor_dim = old.actor[0].in_features
        model = module.SharedGaussianPPO(actor_dim, old.log_std.numel(), old.log_std_min, old.log_std_max).to('cuda')
        payload = torch.load(model_path, map_location='cuda', weights_only=True)
        if payload.get('method') != 'PPO-MV-A' or set(payload) != {'method', 'state_dict'}:
            raise ValueError('Exact own PPO checkpoint schema required')
        model.load_state_dict(payload['state_dict'], strict=True)
    loaded = NeuralVisualController(reference_controller.method, model, reference_controller.settings)
    if actual_parameter_sha256(loaded) != require_sha(expected_parameter_sha256):
        raise RuntimeError('Loaded parameters do not match the externally bound saved model')
    if loaded._started_episode_tokens:
        raise RuntimeError('Loaded validation controller must have no optimizer transaction history')
    validation_checkpoint_reference(checkpoint_path, descriptor, manifest_sha256, 'validation')
    return loaded, dict(**checked, actual_loaded_parameter_sha256=expected_parameter_sha256,
        optimizer_training_state_restored=False, H_MV_or_training_RNG_state_restored=False,
        fresh_constructor_may_advance_or_reseed_global_RNG=True, actual_roundtrip_performed=True)
