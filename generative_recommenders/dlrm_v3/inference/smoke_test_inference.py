#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""
Single-file smoke demo for DLRMv3 inference on movielens-1m.

The prompt schema and feature wiring are fixed in this script to keep
native Kira smoke runs deterministic and easy to compare across checkpoints.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

import generative_recommenders.dlrm_v3.inference.model_family as kira_model_family_module
from generative_recommenders.dlrm_v3.configs import (
    get_embedding_table_config,
    get_hstu_configs,
)
from generative_recommenders.dlrm_v3.datasets.dataset import Samples
from generative_recommenders.dlrm_v3.inference.inference_modules import set_is_inference
from generative_recommenders.dlrm_v3.inference.model_family import HSTUModelFamily

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("kira_smoke")

ENV_DEBUG_IO = "KIRA_DEBUG_IO"
ENV_DEBUG_JSONL = "KIRA_DEBUG_JSONL_PATH"
ENV_DEBUG_PREVIEW_N = "KIRA_DEBUG_PREVIEW_N"
_TRUTHY = {"1", "true", "yes", "on", "y"}


def _is_debug_enabled() -> bool:
    return os.environ.get(ENV_DEBUG_IO, "0").strip().lower() in _TRUTHY


def _get_preview_n(default: int = 8) -> int:
    raw = os.environ.get(ENV_DEBUG_PREVIEW_N)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(value, 0)


def _debug_json_path() -> Optional[str]:
    path = os.environ.get(ENV_DEBUG_JSONL, "").strip()
    return path or None


def _emit_debug_event(event: str, payload: Dict[str, Any]) -> None:
    if not _is_debug_enabled():
        return
    record = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "payload": payload,
    }
    logger.info("KIRA_IO %s", json.dumps(record, ensure_ascii=False))

    path = _debug_json_path()
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, indent=2))
            f.write("\n")
    except Exception as exc:
        logger.warning("KIRA_IO failed writing JSON log %s: %s", path, exc)


def _tensor_summary(
    tensor: torch.Tensor,
    include_l2: bool = False,
) -> Dict[str, Any]:
    detached = tensor.detach()
    preview_n = _get_preview_n()
    summary: Dict[str, Any] = {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "numel": int(detached.numel()),
    }

    if detached.numel() == 0:
        summary["preview"] = []
        return summary

    summary["preview"] = detached.reshape(-1)[:preview_n].cpu().tolist()

    if torch.is_floating_point(detached):
        values = detached.float().cpu()
        summary["mean"] = float(values.mean().item())
        summary["std"] = float(values.std(unbiased=False).item())
        summary["min"] = float(values.min().item())
        summary["max"] = float(values.max().item())
        summary["checksum_fp32_sum"] = float(values.sum().item())
        if include_l2:
            summary["l2_norm"] = float(torch.linalg.vector_norm(values).item())
    else:
        values = detached.to(torch.int64).cpu()
        summary["min"] = int(values.min().item())
        summary["max"] = int(values.max().item())
        summary["checksum_int64_sum"] = int(values.sum().item())

    return summary


def _dtype_count_summary(named_tensors: Dict[str, torch.Tensor]) -> Dict[str, int]:
    counts = Counter(str(t.dtype) for t in named_tensors.values())
    return dict(sorted(counts.items()))


def _prefix_param_summary(
    named_tensors: Dict[str, torch.Tensor],
) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = {}
    for name, tensor in named_tensors.items():
        prefix = name.split(".", 1)[0]
        if prefix not in summary:
            summary[prefix] = {"tensors": 0, "params": 0}
        summary[prefix]["tensors"] += 1
        summary[prefix]["params"] += int(tensor.numel())
    return dict(sorted(summary.items()))


def _kjt_summary(kjt: KeyedJaggedTensor) -> Dict[str, Any]:
    return {
        "keys": list(kjt.keys()),
        "stride": int(kjt.stride()),
        "lengths": _tensor_summary(kjt.lengths()),
        "values": _tensor_summary(kjt.values()),
    }


def _configure_debug_env(
    debug_io: bool,
    debug_jsonl_path: Optional[str],
    debug_preview_n: int,
) -> None:
    os.environ[ENV_DEBUG_IO] = "1" if debug_io else "0"
    os.environ[ENV_DEBUG_PREVIEW_N] = str(max(0, int(debug_preview_n)))
    if debug_jsonl_path:
        os.environ[ENV_DEBUG_JSONL] = os.path.abspath(debug_jsonl_path)
    else:
        os.environ.pop(ENV_DEBUG_JSONL, None)


def _emit_model_loaded_event(
    model_path: str,
    config_json_path: Optional[str],
    hstu_config: Any,
    table_config: Dict[str, Any],
    model_family: HSTUModelFamily,
) -> None:
    if not _is_debug_enabled():
        return

    sparse_state: Dict[str, torch.Tensor] = {}
    dense_state: Dict[str, torch.Tensor] = {}

    sparse_module = getattr(model_family.sparse, "module", None)
    if sparse_module is not None and hasattr(sparse_module, "_hstu_model"):
        sparse_state = {
            k: v for k, v in sparse_module._hstu_model.state_dict().items()  # pyre-ignore[16]
            if isinstance(v, torch.Tensor)
        }

    dense_module = getattr(model_family.dense, "model", None)
    if dense_module is not None:
        dense_state = {
            k: v for k, v in dense_module.state_dict().items()
            if isinstance(v, torch.Tensor)
        }

    _emit_debug_event(
        event="kira_smoke.model_loaded",
        payload={
            "model_path": model_path,
            "config_json": config_json_path,
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "hstu_config": {
                "max_seq_len": int(hstu_config.max_seq_len),
                "max_num_candidates": int(hstu_config.max_num_candidates),
                "max_num_candidates_inference": int(
                    hstu_config.max_num_candidates_inference
                ),
                "hstu_num_heads": int(hstu_config.hstu_num_heads),
                "hstu_attn_linear_dim": int(hstu_config.hstu_attn_linear_dim),
                "hstu_attn_qk_dim": int(hstu_config.hstu_attn_qk_dim),
                "hstu_attn_num_layers": int(hstu_config.hstu_attn_num_layers),
                "hstu_embedding_table_dim": int(hstu_config.hstu_embedding_table_dim),
                "hstu_preprocessor_hidden_dim": int(
                    hstu_config.hstu_preprocessor_hidden_dim
                ),
                "hstu_transducer_embedding_dim": int(
                    hstu_config.hstu_transducer_embedding_dim
                ),
                "hstu_uih_feature_names": list(hstu_config.hstu_uih_feature_names),
                "hstu_candidate_feature_names": list(
                    hstu_config.hstu_candidate_feature_names
                ),
                "merge_uih_candidate_feature_mapping": list(
                    hstu_config.merge_uih_candidate_feature_mapping
                ),
            },
            "embedding_tables": {
                name: {
                    "embedding_dim": int(cfg.embedding_dim),
                    "num_embeddings": int(cfg.num_embeddings),
                }
                for name, cfg in table_config.items()
            },
            "sparse_state": {
                "tensor_count": len(sparse_state),
                "param_count": int(sum(t.numel() for t in sparse_state.values())),
                "dtype_counts": _dtype_count_summary(sparse_state),
                "prefix_summary": _prefix_param_summary(sparse_state),
            },
            "dense_state": {
                "tensor_count": len(dense_state),
                "param_count": int(sum(t.numel() for t in dense_state.values())),
                "dtype_counts": _dtype_count_summary(dense_state),
                "prefix_summary": _prefix_param_summary(dense_state),
            },
        },
    )


def _install_preprocess_hook(model_family: HSTUModelFamily) -> None:
    if not _is_debug_enabled():
        return

    sparse_module = getattr(model_family.sparse, "module", None)
    if sparse_module is None or not hasattr(sparse_module, "_hstu_model"):
        return

    hstu_model = sparse_module._hstu_model  # pyre-ignore[16]
    original_preprocess = hstu_model.preprocess

    if getattr(original_preprocess, "_kira_debug_wrapped", False):
        return

    def wrapped_preprocess(*args: Any, **kwargs: Any):
        out = original_preprocess(*args, **kwargs)
        (
            seq_embeddings,
            payload_features,
            max_uih_len,
            uih_seq_lengths,
            max_num_candidates,
            num_candidates,
        ) = out

        _emit_debug_event(
            event="kira_smoke.preprocess_output",
            payload={
                "max_uih_len": int(max_uih_len),
                "max_num_candidates": int(max_num_candidates),
                "uih_seq_lengths": _tensor_summary(uih_seq_lengths),
                "num_candidates": _tensor_summary(num_candidates),
                "payload_features": {
                    k: _tensor_summary(v)
                    for k, v in payload_features.items()
                },
                "seq_embeddings": {
                    k: {
                        "lengths": _tensor_summary(v.lengths),
                        "embedding": _tensor_summary(v.embedding, include_l2=True),
                    }
                    for k, v in seq_embeddings.items()
                },
            },
        )
        return out

    wrapped_preprocess._kira_debug_wrapped = True  # type: ignore[attr-defined]
    hstu_model.preprocess = wrapped_preprocess


def _install_sparse_move_hook() -> None:
    if not _is_debug_enabled():
        return

    original_move = kira_model_family_module.move_sparse_output_to_device
    if getattr(original_move, "_kira_debug_wrapped", False):
        return

    def wrapped_move_sparse_output_to_device(
        seq_embeddings: Dict[str, Any],
        payload_features: Dict[str, torch.Tensor],
        uih_seq_lengths: torch.Tensor,
        num_candidates: torch.Tensor,
        device: torch.device,
    ):
        payload_before = {
            k: {"device": str(v.device), "dtype": str(v.dtype)}
            for k, v in payload_features.items()
        }

        result = original_move(
            seq_embeddings=seq_embeddings,
            payload_features=payload_features,
            uih_seq_lengths=uih_seq_lengths,
            num_candidates=num_candidates,
            device=device,
        )

        moved_seq_embeddings, moved_payload, moved_uih_lengths, moved_num_candidates = (
            result
        )

        seq_move: Dict[str, Any] = {}
        for key in moved_seq_embeddings.keys():
            src_embedding = seq_embeddings[key].embedding
            dst_embedding = moved_seq_embeddings[key].embedding
            move_info: Dict[str, Any] = {
                "lengths_device": (
                    f"{seq_embeddings[key].lengths.device}->"
                    f"{moved_seq_embeddings[key].lengths.device}"
                ),
                "embedding_device": f"{src_embedding.device}->{dst_embedding.device}",
                "embedding_dtype": f"{src_embedding.dtype}->{dst_embedding.dtype}",
            }
            if src_embedding.numel() > 0 and torch.is_floating_point(src_embedding):
                src_fp32 = src_embedding.detach().float().cpu()
                dst_fp32 = dst_embedding.detach().float().cpu()
                abs_diff = (src_fp32 - dst_fp32).abs()
                move_info["cast_max_abs_diff"] = float(abs_diff.max().item())
                move_info["cast_mean_abs_diff"] = float(abs_diff.mean().item())
            move_info["embedding_after_cast"] = _tensor_summary(
                dst_embedding,
                include_l2=True,
            )
            seq_move[key] = move_info

        payload_move = {
            k: {
                "device": (
                    f"{payload_before.get(k, {}).get('device', 'unknown')}->"
                    f"{moved_payload[k].device}"
                ),
                "dtype": (
                    f"{payload_before.get(k, {}).get('dtype', 'unknown')}->"
                    f"{moved_payload[k].dtype}"
                ),
            }
            for k in moved_payload.keys()
        }

        _emit_debug_event(
            event="kira_smoke.sparse_output_moved",
            payload={
                "target_device": str(device),
                "seq_embeddings": seq_move,
                "payload_features": payload_move,
                "uih_seq_lengths": _tensor_summary(moved_uih_lengths),
                "num_candidates": _tensor_summary(moved_num_candidates),
            },
        )

        return result

    wrapped_move_sparse_output_to_device._kira_debug_wrapped = True  # type: ignore[attr-defined]
    kira_model_family_module.move_sparse_output_to_device = (
        wrapped_move_sparse_output_to_device
    )


def _install_dense_hooks(model_family: HSTUModelFamily) -> None:
    if not _is_debug_enabled():
        return

    dense_model = getattr(model_family.dense, "model", None)
    if dense_model is None:
        logger.warning(
            "KIRA_IO dense hooks skipped: dense.model is None "
            "(likely distributed dense workers)"
        )
        return

    original_item_forward = dense_model._item_forward  # pyre-ignore[16]
    if getattr(original_item_forward, "_kira_dense_debug_wrapped", False):
        return

    item_feature_names = list(dense_model._hstu_configs.item_embedding_feature_names)  # pyre-ignore[16]
    item_mlp_params = {
        name: p.detach()
        for name, p in dense_model._item_embedding_mlp.named_parameters()  # pyre-ignore[16]
    }
    _emit_debug_event(
        event="kira_smoke.item_forward_mlp_weights_once",
        payload={
            "item_embedding_mlp": {
                name: _tensor_summary(param)
                for name, param in item_mlp_params.items()
            }
        },
    )

    def wrapped_item_forward(*args: Any, **kwargs: Any):
        seq_embeddings = kwargs.get("seq_embeddings")
        if seq_embeddings is None and args:
            seq_embeddings = args[0]

        if isinstance(seq_embeddings, dict):
            _emit_debug_event(
                event="kira_smoke.item_forward_input_features",
                payload={
                    "item_embedding_feature_names": item_feature_names,
                    "item_feature_embeddings": {
                        name: _tensor_summary(
                            seq_embeddings[name].embedding,
                            include_l2=True,
                        )
                        for name in item_feature_names
                    },
                },
            )

            all_embeddings = torch.cat(
                [seq_embeddings[name].embedding for name in item_feature_names],
                dim=-1,
            )
            _emit_debug_event(
                event="kira_smoke.item_forward_concat_output",
                payload={
                    "all_embeddings": _tensor_summary(
                        all_embeddings,
                        include_l2=True,
                    ),
                },
            )

        layer_outputs: Dict[str, Dict[str, Any]] = {}
        hooks = []
        item_mlp = dense_model._item_embedding_mlp  # pyre-ignore[16]

        def _make_hook(layer_name: str):
            def _hook(_module: torch.nn.Module, _inputs: Tuple[Any, ...], output: Any):
                if isinstance(output, torch.Tensor):
                    layer_outputs[layer_name] = _tensor_summary(
                        output,
                        include_l2=True,
                    )
            return _hook

        for idx, layer in enumerate(item_mlp):
            layer_name = f"{idx}_{layer.__class__.__name__}"
            hooks.append(layer.register_forward_hook(_make_hook(layer_name)))

        try:
            item_embeddings = original_item_forward(*args, **kwargs)
        finally:
            for hook in hooks:
                hook.remove()

        _emit_debug_event(
            event="kira_smoke.item_forward_mlp_layer_outputs",
            payload={
                "layers": layer_outputs,
            },
        )

        _emit_debug_event(
            event="kira_smoke.item_forward_output",
            payload={
                "candidates_item_embeddings": _tensor_summary(
                    item_embeddings,
                    include_l2=True,
                ),
            },
        )
        return item_embeddings

    wrapped_item_forward._kira_dense_debug_wrapped = True  # type: ignore[attr-defined]
    dense_model._item_forward = wrapped_item_forward  # pyre-ignore[16]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Native smoke demo for DLRMv3 movielens-1m inference",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to native checkpoint directory (contains non_sparse.ckpt and sparse/).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top candidates to print (default: 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for reproducibility (default: 123).",
    )
    parser.add_argument(
        "--config-json",
        type=str,
        default=None,
        help=(
            "Optional config.json path with embedding_tables[*].num_embeddings. "
            "If omitted, the script tries <model-path>/config.json."
        ),
    )
    parser.add_argument(
        "--max-num-embeddings",
        type=int,
        default=None,
        help=(
            "Optional upper bound for each embedding table's num_embeddings. "
            "Use this to match truncated checkpoints and avoid shape mismatch."
        ),
    )
    parser.add_argument(
        "--debug-io",
        action="store_true",
        default=False,
        help="Enable Kira structured debug events for KJT/preprocess/sparse move.",
    )
    parser.add_argument(
        "--debug-jsonl-path",
        type=str,
        default=None,
        help="Optional path to write structured Kira debug JSON logs.",
    )
    parser.add_argument(
        "--debug-preview-n",
        type=int,
        default=8,
        help="Preview length for vector/tensor debug summaries.",
    )
    return parser.parse_args()


def _validate_model_path(model_path: str) -> None:
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    dense_path = os.path.join(model_path, "non_sparse.ckpt")
    sparse_path = os.path.join(model_path, "sparse")
    if not os.path.isfile(dense_path):
        raise FileNotFoundError(f"Missing dense checkpoint: {dense_path}")
    if not os.path.isdir(sparse_path):
        raise FileNotFoundError(f"Missing sparse checkpoint directory: {sparse_path}")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_config_json_path(
    model_path: str,
    config_json_arg: Optional[str],
) -> Optional[str]:
    if config_json_arg:
        return config_json_arg

    inferred = os.path.join(model_path, "config.json")
    if os.path.isfile(inferred):
        return inferred
    return None


def _apply_num_embeddings_from_config(
    table_config: Dict[str, Any],
    config_json_path: str,
) -> None:
    with open(config_json_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    embedding_tables = cfg.get("embedding_tables")
    if not isinstance(embedding_tables, list):
        print(
            f"config.json has no embedding_tables list, skip sync: {config_json_path}"
        )
        return

    print(f"Syncing num_embeddings from config: {config_json_path}")
    cfg_rows_by_name: Dict[str, int] = {}
    for table in embedding_tables:
        if isinstance(table, dict) and "name" in table and "num_embeddings" in table:
            cfg_rows_by_name[str(table["name"])] = int(table["num_embeddings"])

    for table_name, emb_cfg in table_config.items():
        if table_name not in cfg_rows_by_name:
            continue
        old_rows = int(emb_cfg.num_embeddings)
        new_rows = int(cfg_rows_by_name[table_name])
        emb_cfg.num_embeddings = new_rows
        print(f"  table={table_name}: {old_rows} -> {new_rows} (from config)")


def _truncate_table_config_num_embeddings(
    table_config: Dict[str, Any],
    max_num_embeddings: Optional[int],
) -> None:
    if max_num_embeddings is None:
        return
    if max_num_embeddings <= 0:
        raise ValueError(
            f"--max-num-embeddings must be > 0, got {max_num_embeddings}"
        )

    print(f"Applying num_embeddings cap: {max_num_embeddings}")
    for table_name, emb_cfg in table_config.items():
        old_rows = int(emb_cfg.num_embeddings)
        new_rows = min(old_rows, max_num_embeddings)
        emb_cfg.num_embeddings = new_rows
        print(f"  table={table_name}: {old_rows} -> {new_rows}")


def _build_hardcoded_prompt() -> Tuple[List[Dict[str, int]], List[Dict[str, int]], Dict[str, int]]:
    user_history = [
        {
            "movie_id": 1,
            "movie_rating": 5,
            "action_timestamp": 978300760,
            "item_weights": 1,
            "dummy_watch_time": 0,
        },
        {
            "movie_id": 260,
            "movie_rating": 4,
            "action_timestamp": 978301398,
            "item_weights": 1,
            "dummy_watch_time": 0,
        },
        {
            "movie_id": 1196,
            "movie_rating": 5,
            "action_timestamp": 978302174,
            "item_weights": 2,
            "dummy_watch_time": 0,
        },
        {
            "movie_id": 1210,
            "movie_rating": 4,
            "action_timestamp": 978302983,
            "item_weights": 1,
            "dummy_watch_time": 0,
        },
        {
            "movie_id": 2028,
            "movie_rating": 5,
            "action_timestamp": 978303390,
            "item_weights": 4,
            "dummy_watch_time": 0,
        },
    ]

    candidates = [
        {
            "item_movie_id": 3114,
            "item_query_time": 978310000,
            "item_dummy_watchtime": 0,
            "item_movie_rating": 0,
            "item_action_weights": 0,
        },
        {
            "item_movie_id": 2571,
            "item_query_time": 978310000,
            "item_dummy_watchtime": 0,
            "item_movie_rating": 0,
            "item_action_weights": 0,
        },
        {
            "item_movie_id": 1580,
            "item_query_time": 978310000,
            "item_dummy_watchtime": 0,
            "item_movie_rating": 0,
            "item_action_weights": 0,
        },
        {
            "item_movie_id": 1270,
            "item_query_time": 978310000,
            "item_dummy_watchtime": 0,
            "item_movie_rating": 0,
            "item_action_weights": 0,
        },
        {
            "item_movie_id": 593,
            "item_query_time": 978310000,
            "item_dummy_watchtime": 0,
            "item_movie_rating": 0,
            "item_action_weights": 0,
        },
    ]

    context = {
        "user_id": 42,
        "sex": 1,
        "age_group": 25,
        "occupation": 7,
        "zip_code": 10001,
    }

    return user_history, candidates, context


def _build_native_samples(
    hstu_config: Any,
    user_history: List[Dict[str, int]],
    candidates: List[Dict[str, int]],
    context: Dict[str, int],
) -> Samples:
    seq_len = len(user_history)
    num_candidates = len(candidates)

    uih_keys = list(hstu_config.hstu_uih_feature_names)
    contextual_keys = set(hstu_config.contextual_feature_to_max_length.keys())
    ctx_keys = [key for key in uih_keys if key in contextual_keys]

    uih_values: List[int] = []
    uih_lengths: List[int] = []
    for key in uih_keys:
        if key in contextual_keys:
            uih_values.append(int(context.get(key, 0)))
            uih_lengths.append(1)
        else:
            vals = [int(row.get(key, 0)) for row in user_history]
            uih_values.extend(vals)
            uih_lengths.append(seq_len)

    cand_keys = list(hstu_config.hstu_candidate_feature_names)
    cand_values: List[int] = []
    cand_lengths: List[int] = []
    for key in cand_keys:
        vals = [int(row.get(key, 0)) for row in candidates]
        cand_values.extend(vals)
        cand_lengths.append(num_candidates)

    if _is_debug_enabled():
        _emit_debug_event(
            event="kira_smoke.prompt_features_built",
            payload={
                "uih_keys": uih_keys,
                "cand_keys": cand_keys,
                "ctx_keys": ctx_keys,
                "history_len": seq_len,
                "candidate_len": num_candidates,
                "uih_lengths_per_key": uih_lengths,
                "cand_lengths_per_key": cand_lengths,
                "uih_values_preview": uih_values[:_get_preview_n()],
                "cand_values_preview": cand_values[:_get_preview_n()],
            },
        )

    uih_features_kjt = KeyedJaggedTensor(
        keys=uih_keys,
        lengths=torch.tensor(uih_lengths, dtype=torch.long),
        values=torch.tensor(uih_values, dtype=torch.long),
    )
    candidates_features_kjt = KeyedJaggedTensor(
        keys=cand_keys,
        lengths=torch.tensor(cand_lengths, dtype=torch.long),
        values=torch.tensor(cand_values, dtype=torch.long),
    )

    if _is_debug_enabled():
        _emit_debug_event(
            event="kira_smoke.kjt_built",
            payload={
                "uih_features_kjt": _kjt_summary(uih_features_kjt),
                "candidates_features_kjt": _kjt_summary(candidates_features_kjt),
            },
        )

    return Samples(
        uih_features_kjt=uih_features_kjt,
        candidates_features_kjt=candidates_features_kjt,
    )


def main() -> None:
    args = _parse_args()
    _validate_model_path(args.model_path)

    _configure_debug_env(
        debug_io=args.debug_io,
        debug_jsonl_path=args.debug_jsonl_path,
        debug_preview_n=args.debug_preview_n,
    )

    os.environ["WORLD_SIZE"] = "1"
    _set_seed(args.seed)
    set_is_inference(is_inference=True)

    hstu_config = get_hstu_configs("movielens-1m")
    hstu_config.max_num_candidates = hstu_config.max_num_candidates_inference
    table_config = get_embedding_table_config("movielens-1m")

    config_json_path = _resolve_config_json_path(
        model_path=args.model_path,
        config_json_arg=args.config_json,
    )
    if config_json_path is not None:
        _apply_num_embeddings_from_config(
            table_config=table_config,
            config_json_path=config_json_path,
        )

    _truncate_table_config_num_embeddings(
        table_config=table_config,
        max_num_embeddings=args.max_num_embeddings,
    )

    model_family = HSTUModelFamily(
        hstu_config=hstu_config,
        table_config=table_config,
        output_trace=False,
        sparse_quant=False,
        compute_eval=False,
    )

    user_history, candidate_rows, context = _build_hardcoded_prompt()
    samples = _build_native_samples(
        hstu_config=hstu_config,
        user_history=user_history,
        candidates=candidate_rows,
        context=context,
    )

    print("=" * 70)
    print(" Native DLRMv3 Smoke Inference (movielens-1m)")
    print("=" * 70)
    print(f"Model path: {args.model_path}")
    print(f"Config json: {config_json_path}")
    print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE')}")
    print(f"Seed: {args.seed}")
    print(f"max_num_embeddings: {args.max_num_embeddings}")
    print(f"History len: {len(user_history)}")
    print(f"Candidates: {len(candidate_rows)}")
    if args.debug_io:
        print("Debug IO: ENABLED")
        print(f"Preview n: {_get_preview_n()}")
        if args.debug_jsonl_path:
            print(f"Debug JSON path: {os.path.abspath(args.debug_jsonl_path)}")

    _install_sparse_move_hook()

    model_family.load(args.model_path)
    _emit_model_loaded_event(
        model_path=args.model_path,
        config_json_path=config_json_path,
        hstu_config=hstu_config,
        table_config=table_config,
        model_family=model_family,
    )
    _install_preprocess_hook(model_family)
    _install_dense_hooks(model_family)

    try:
        pred_output = model_family.predict(samples)
        if pred_output is None:
            raise RuntimeError("predict returned None unexpectedly")
        mt_target_preds, _, _, dt_sparse, dt_dense = pred_output
    finally:
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            model_family.predict(None)

    scores = mt_target_preds.view(-1).detach().cpu().float().tolist()
    item_ids = [int(row["item_movie_id"]) for row in candidate_rows]

    pairs = list(zip(item_ids, scores))
    pairs.sort(key=lambda x: x[1], reverse=True)

    top_k = max(1, min(args.top_k, len(pairs)))
    print(f"\nTop-{top_k} recommendations:")
    for rank, (item_id, score) in enumerate(pairs[:top_k], start=1):
        print(f"  #{rank}: movie_{item_id} score={score:.6f}")

    print("\nTimings:")
    print(f"  sparse: {dt_sparse:.6f}s")
    print(f"  dense:  {dt_dense:.6f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
