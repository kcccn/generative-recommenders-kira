#!/usr/bin/env python3
"""
Smoke test DlrmHSTU inference directly with KuaiRand-style KJT inputs.

This script is smoke-test oriented:
- config is embedded in code (no external config files)
- model weights are randomly initialized
- inputs are generated with deterministic random seed
- output metrics are printed for side-by-side comparison
"""

import os
import random
import sys
from typing import Dict, List, Tuple

os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
sys.path.insert(0, REPO_ROOT)

SEED = 20260313
DATASET_NAME = "kuairand-1k"


def _set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_kuairand_config_dict() -> Dict[str, object]:
    return {
        "max_seq_len": 64,
        "max_num_candidates": 20,
        "max_num_candidates_inference": 20,
        "hstu_num_heads": 2,
        "hstu_attn_linear_dim": 64,
        "hstu_attn_qk_dim": 32,
        "hstu_attn_num_layers": 2,
        "hstu_embedding_table_dim": 64,
        "hstu_preprocessor_hidden_dim": 64,
        "hstu_transducer_embedding_dim": 64,
        "hstu_group_norm": False,
        "hstu_input_dropout_ratio": 0.0,
        "hstu_linear_dropout_rate": 0.0,
        "contextual_feature_to_max_length": {
            "user_id": 1,
            "user_active_degree": 1,
            "follow_user_num_range": 1,
            "fans_user_num_range": 1,
            "friend_user_num_range": 1,
            "register_days_range": 1,
        },
        "candidates_weight_feature_name": "item_action_weight",
        "candidates_watchtime_feature_name": "item_target_watchtime",
        "candidates_querytime_feature_name": "item_query_time",
        "user_embedding_feature_names": [
            "video_id",
            "user_id",
            "user_active_degree",
            "follow_user_num_range",
            "fans_user_num_range",
            "friend_user_num_range",
            "register_days_range",
        ],
        "item_embedding_feature_names": ["item_video_id"],
        "uih_post_id_feature_name": "video_id",
        "uih_action_time_feature_name": "action_timestamp",
        "uih_weight_feature_name": "action_weight",
        "hstu_uih_feature_names": [
            "user_id",
            "user_active_degree",
            "follow_user_num_range",
            "fans_user_num_range",
            "friend_user_num_range",
            "register_days_range",
            "video_id",
            "action_timestamp",
            "action_weight",
            "watch_time",
        ],
        "hstu_candidate_feature_names": [
            "item_video_id",
            "item_action_weight",
            "item_target_watchtime",
            "item_query_time",
        ],
        "merge_uih_candidate_feature_mapping": [
            ("video_id", "item_video_id"),
            ("action_timestamp", "item_query_time"),
            ("action_weight", "item_action_weight"),
            ("watch_time", "item_target_watchtime"),
        ],
        "action_weights": [1, 2, 4, 8, 16, 32, 64, 128],
        # Small smoke-test embedding tables (format preserved, sizes reduced)
        "embedding_tables": {
            "video_id": {
                "name": "video_id",
                "embedding_dim": 64,
                "num_embeddings": 10000,
                "feature_names": ["video_id", "item_video_id"],
            },
            "user_id": {
                "name": "user_id",
                "embedding_dim": 64,
                "num_embeddings": 10000,
                "feature_names": ["user_id"],
            },
            "user_active_degree": {
                "name": "user_active_degree",
                "embedding_dim": 64,
                "num_embeddings": 8,
                "feature_names": ["user_active_degree"],
            },
            "follow_user_num_range": {
                "name": "follow_user_num_range",
                "embedding_dim": 64,
                "num_embeddings": 9,
                "feature_names": ["follow_user_num_range"],
            },
            "fans_user_num_range": {
                "name": "fans_user_num_range",
                "embedding_dim": 64,
                "num_embeddings": 9,
                "feature_names": ["fans_user_num_range"],
            },
            "friend_user_num_range": {
                "name": "friend_user_num_range",
                "embedding_dim": 64,
                "num_embeddings": 8,
                "feature_names": ["friend_user_num_range"],
            },
            "register_days_range": {
                "name": "register_days_range",
                "embedding_dim": 64,
                "num_embeddings": 8,
                "feature_names": ["register_days_range"],
            },
        },
    }


def _feature_upper_bounds(embedding_tables: Dict[str, Dict[str, object]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for table in embedding_tables.values():
        num_embeddings = int(table["num_embeddings"])
        for name in table["feature_names"]:
            out[str(name)] = num_embeddings
    return out


def _sample_values_for_key(key: str, length: int, upper_bounds: Dict[str, int], device):
    import torch

    if key in {"action_timestamp", "item_query_time"}:
        return torch.randint(1_700_000_000, 1_700_100_000, (length,), device=device)
    if key in {"action_weight", "item_action_weight"}:
        return torch.randint(1, 129, (length,), device=device)
    if key in {"watch_time", "item_target_watchtime"}:
        return torch.randint(1, 1000, (length,), device=device)

    upper = upper_bounds.get(key, 2048)
    if upper <= 1:
        return torch.zeros((length,), dtype=torch.long, device=device)
    return torch.randint(0, upper, (length,), device=device)


def _build_kjt_inputs(config: Dict[str, object], device) -> Tuple[object, object, int, int]:
    import torch
    from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

    contextual = set(config["contextual_feature_to_max_length"].keys())
    uih_keys = list(config["hstu_uih_feature_names"])
    cand_keys = list(config["hstu_candidate_feature_names"])
    upper_bounds = _feature_upper_bounds(config["embedding_tables"])

    max_seq_len = int(config["max_seq_len"])
    max_candidates = int(config["max_num_candidates"])
    max_uih_len = max_seq_len - max_candidates - len(contextual)
    low_uih = max(2, int(max_uih_len * 0.8))
    uih_seq_len = int(torch.randint(low_uih, max_uih_len + 1, (1,), device=device).item())
    num_candidates = int(torch.randint(2, max_candidates + 1, (1,), device=device).item())

    uih_lengths: List[int] = []
    uih_values_parts = []
    for key in uih_keys:
        length = 1 if key in contextual else uih_seq_len
        uih_lengths.append(length)
        uih_values_parts.append(_sample_values_for_key(key, length, upper_bounds, device))

    cand_lengths: List[int] = []
    cand_values_parts = []
    for key in cand_keys:
        cand_lengths.append(num_candidates)
        cand_values_parts.append(_sample_values_for_key(key, num_candidates, upper_bounds, device))

    uih_kjt = KeyedJaggedTensor.from_lengths_sync(
        keys=uih_keys,
        values=torch.cat(uih_values_parts, dim=0).long(),
        lengths=torch.tensor(uih_lengths, device=device).long(),
    )
    cand_kjt = KeyedJaggedTensor.from_lengths_sync(
        keys=cand_keys,
        values=torch.cat(cand_values_parts, dim=0).long(),
        lengths=torch.tensor(cand_lengths, device=device).long(),
    )

    return uih_kjt, cand_kjt, uih_seq_len, num_candidates


def _print_summary(preds, uih_kjt, cand_kjt, uih_seq_len: int, num_candidates: int) -> None:
    import torch

    flat = preds.detach().float().reshape(-1).cpu()
    preview_n = min(8, flat.numel())

    print("\n=== Comparison Metrics ===")
    print(f"dataset: {DATASET_NAME}")
    print(f"seed: {SEED}")
    print(f"uih_keys: {list(uih_kjt.keys())}")
    print(f"candidate_keys: {list(cand_kjt.keys())}")
    print(f"uih_seq_len: {uih_seq_len}")
    print(f"num_candidates: {num_candidates}")
    print(f"uih_lengths: {uih_kjt.lengths().cpu().tolist()}")
    print(f"candidate_lengths: {cand_kjt.lengths().cpu().tolist()}")
    print(f"pred_shape: {tuple(preds.shape)}")
    print(f"pred_dtype: {preds.dtype}")
    print(f"pred_device: {preds.device}")
    print(f"pred_preview_{preview_n}: {flat[:preview_n].tolist()}")
    print(f"pred_sum: {float(flat.sum().item()):.6f}")
    print(f"pred_mean: {float(flat.mean().item()):.6f}")
    std_val = float(torch.std(flat, unbiased=False).item()) if flat.numel() > 0 else 0.0
    print(f"pred_std: {std_val:.6f}")


def _build_model(config: Dict[str, object], device):
    import torch
    from generative_recommenders.modules.dlrm_hstu import DlrmHSTU, DlrmHSTUConfig
    from generative_recommenders.modules.multitask_module import (
        MultitaskTaskType,
        TaskConfig,
    )
    from torchrec.modules.embedding_configs import DataType, EmbeddingConfig
    from torchrec.modules.embedding_modules import EmbeddingBagCollection, EmbeddingCollection

    task_names = [
        "is_click",
        "is_like",
        "is_follow",
        "is_comment",
        "is_forward",
        "is_hate",
        "long_view",
        "is_profile_enter",
    ]
    multitask_configs = [
        TaskConfig(
            task_name=name,
            task_weight=(1 << idx),
            task_type=MultitaskTaskType.BINARY_CLASSIFICATION,
        )
        for idx, name in enumerate(task_names)
    ]

    hstu_cfg = DlrmHSTUConfig(
        max_seq_len=int(config["max_seq_len"]),
        max_num_candidates=int(config["max_num_candidates"]),
        max_num_candidates_inference=int(config["max_num_candidates_inference"]),
        hstu_num_heads=int(config["hstu_num_heads"]),
        hstu_attn_linear_dim=int(config["hstu_attn_linear_dim"]),
        hstu_attn_qk_dim=int(config["hstu_attn_qk_dim"]),
        hstu_attn_num_layers=int(config["hstu_attn_num_layers"]),
        hstu_embedding_table_dim=int(config["hstu_embedding_table_dim"]),
        hstu_preprocessor_hidden_dim=int(config["hstu_preprocessor_hidden_dim"]),
        hstu_transducer_embedding_dim=int(config["hstu_transducer_embedding_dim"]),
        hstu_group_norm=bool(config["hstu_group_norm"]),
        hstu_input_dropout_ratio=float(config["hstu_input_dropout_ratio"]),
        hstu_linear_dropout_rate=float(config["hstu_linear_dropout_rate"]),
        contextual_feature_to_max_length=dict(config["contextual_feature_to_max_length"]),
        candidates_weight_feature_name=str(config["candidates_weight_feature_name"]),
        candidates_watchtime_feature_name=str(config["candidates_watchtime_feature_name"]),
        candidates_querytime_feature_name=str(config["candidates_querytime_feature_name"]),
        user_embedding_feature_names=list(config["user_embedding_feature_names"]),
        item_embedding_feature_names=list(config["item_embedding_feature_names"]),
        uih_post_id_feature_name=str(config["uih_post_id_feature_name"]),
        uih_action_time_feature_name=str(config["uih_action_time_feature_name"]),
        uih_weight_feature_name=str(config["uih_weight_feature_name"]),
        hstu_uih_feature_names=list(config["hstu_uih_feature_names"]),
        hstu_candidate_feature_names=list(config["hstu_candidate_feature_names"]),
        merge_uih_candidate_feature_mapping=list(config["merge_uih_candidate_feature_mapping"]),
        action_weights=list(config["action_weights"]),
        multitask_configs=multitask_configs,
    )

    table_cfg = {}
    for table_name, table in config["embedding_tables"].items():
        table_cfg[table_name] = EmbeddingConfig(
            num_embeddings=int(table["num_embeddings"]),
            embedding_dim=int(table["embedding_dim"]),
            name=str(table["name"]),
            data_type=DataType.FP16,
            feature_names=list(table["feature_names"]),
        )

    model = DlrmHSTU(
        hstu_configs=hstu_cfg,
        embedding_tables=table_cfg,
        is_inference=True,
        is_dense=False,
    )
    model.eval()
    model.recursive_setattr("_use_triton_cc", False)

    for _, module in model.named_modules():
        if isinstance(module, EmbeddingBagCollection) or isinstance(module, EmbeddingCollection):
            module.to_empty(device=device)

    for module in model.modules():
        reset_fn = getattr(module, "reset_parameters", None)
        if callable(reset_fn):
            try:
                reset_fn()
            except Exception:
                pass

    model = model.to(device)
    return model


def main() -> int:
    try:
        import numpy as np  # noqa: F401
        import torch
        from torchrec.sparse.jagged_tensor import KeyedJaggedTensor  # noqa: F401
    except Exception as exc:
        print("Failed to import required dependencies.")
        print("Please ensure torch, torchrec and generative_recommenders dependencies are installed.")
        print(f"Import error: {exc}")
        return 2

    _set_seed(SEED)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    config = _build_kuairand_config_dict()

    try:
        model = _build_model(config, device)
        uih_kjt, cand_kjt, uih_seq_len, num_candidates = _build_kjt_inputs(config, device)

        with torch.no_grad():
            _, _, _, preds, _, _ = model(uih_kjt, cand_kjt)
            if preds is None:
                raise RuntimeError("Model returned None predictions.")

        _print_summary(preds, uih_kjt, cand_kjt, uih_seq_len, num_candidates)
        return 0
    except Exception as exc:
        print("Inference failed.")
        print("If this is related to fbgemm/torchrec ops, check torchrec + fbgemm installation.")
        print(f"Runtime error: {exc}")
        return 3


if __name__ == "__main__":
    sys.exit(main())
