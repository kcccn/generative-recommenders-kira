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
import generative_recommenders.modules.preprocessors as kira_preprocessors_module
from generative_recommenders.dlrm_v3.configs import (
    get_embedding_table_config,
    get_hstu_configs,
)
from generative_recommenders.dlrm_v3.datasets.dataset import Samples
from generative_recommenders.dlrm_v3.inference.inference_modules import set_is_inference
from generative_recommenders.dlrm_v3.inference.model_family import HSTUModelFamily
from generative_recommenders.ops.jagged_tensors import concat_2D_jagged

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

    hstu_transducer = getattr(dense_model, "_hstu_transducer", None)  # pyre-ignore[16]
    if hstu_transducer is not None:
        output_postprocessor = getattr(hstu_transducer, "_output_postprocessor", None)
        if output_postprocessor is not None:
            original_output_postprocessor_forward = output_postprocessor.forward
            if not getattr(
                original_output_postprocessor_forward,
                "_kira_user_debug_wrapped",
                False,
            ):
                def wrapped_output_postprocessor_forward(*args: Any, **kwargs: Any):
                    candidate_embeddings_in = kwargs.get(
                        "seq_embeddings",
                        args[0] if len(args) > 0 else None,
                    )
                    candidate_timestamps = kwargs.get(
                        "seq_timestamps",
                        args[1] if len(args) > 1 else None,
                    )
                    out = original_output_postprocessor_forward(*args, **kwargs)
                    _emit_debug_event(
                        event="kira_smoke.user_transducer_output_postprocessor_io",
                        payload={
                            "candidate_embeddings_input": (
                                _tensor_summary(candidate_embeddings_in, include_l2=True)
                                if isinstance(candidate_embeddings_in, torch.Tensor)
                                else None
                            ),
                            "candidate_timestamps": (
                                _tensor_summary(candidate_timestamps)
                                if isinstance(candidate_timestamps, torch.Tensor)
                                else None
                            ),
                            "candidate_embeddings_output": _tensor_summary(
                                out,
                                include_l2=True,
                            ),
                        },
                    )
                    return out

                wrapped_output_postprocessor_forward._kira_user_debug_wrapped = True  # type: ignore[attr-defined]
                output_postprocessor.forward = wrapped_output_postprocessor_forward

        input_preprocessor = getattr(hstu_transducer, "_input_preprocessor", None)
        if input_preprocessor is not None:
            original_input_preprocessor_forward = input_preprocessor.forward
            if not getattr(
                original_input_preprocessor_forward,
                "_kira_user_debug_wrapped",
                False,
            ):
                def wrapped_input_preprocessor_forward(*args: Any, **kwargs: Any):
                    seq_lengths = kwargs.get(
                        "seq_lengths",
                        args[4] if len(args) > 4 else None,
                    )
                    seq_timestamps = kwargs.get(
                        "seq_timestamps",
                        args[5] if len(args) > 5 else None,
                    )
                    seq_embeddings_in = kwargs.get(
                        "seq_embeddings",
                        args[6] if len(args) > 6 else None,
                    )
                    num_targets = kwargs.get(
                        "num_targets",
                        args[7] if len(args) > 7 else None,
                    )
                    seq_payloads = kwargs.get(
                        "seq_payloads",
                        args[8] if len(args) > 8 else None,
                    )

                    stage_tensors: Dict[str, torch.Tensor] = {}
                    hooks = []

                    def _capture(name: str):
                        def _hook(
                            _module: torch.nn.Module,
                            _inputs: Tuple[Any, ...],
                            output: Any,
                        ) -> None:
                            if isinstance(output, torch.Tensor):
                                stage_tensors[name] = output
                        return _hook

                    hooks.append(
                        input_preprocessor._content_embedding_mlp.register_forward_hook(  # pyre-ignore[16]
                            _capture("content_mlp_output")
                        )
                    )
                    hooks.append(
                        input_preprocessor._additional_embedding_mlp.register_forward_hook(  # pyre-ignore[16]
                            _capture("additional_mlp_output")
                        )
                    )
                    if getattr(input_preprocessor, "_action_weights", None) is not None:
                        hooks.append(
                            input_preprocessor._action_encoder.register_forward_hook(  # pyre-ignore[16]
                                _capture("action_embeddings")
                            )
                        )
                        hooks.append(
                            input_preprocessor._action_embedding_mlp.register_forward_hook(  # pyre-ignore[16]
                                _capture("action_mlp_output")
                            )
                        )

                    out = original_input_preprocessor_forward(*args, **kwargs)
                    for hook in hooks:
                        hook.remove()
                    (
                        out_max_seq_len,
                        out_total_uih_len,
                        out_total_targets,
                        out_seq_lengths,
                        out_seq_offsets,
                        out_seq_timestamps,
                        out_seq_embeddings,
                        out_num_targets,
                        out_seq_payloads,
                    ) = out

                    if (
                        isinstance(seq_embeddings_in, torch.Tensor)
                        and "content_mlp_output" in stage_tensors
                    ):
                        _emit_debug_event(
                            event="kira_smoke.input_preprocessor_stage_content_mlp_output",
                            payload={
                                "seq_embeddings_input": _tensor_summary(
                                    seq_embeddings_in,
                                    include_l2=True,
                                ),
                                "content_mlp_output": _tensor_summary(
                                    stage_tensors["content_mlp_output"],
                                    include_l2=True,
                                ),
                            },
                        )

                    if isinstance(seq_payloads, dict):
                        additional_features = list(
                            getattr(input_preprocessor, "_additional_embedding_features", [])
                        )
                        if additional_features and "additional_mlp_output" in stage_tensors:
                            additional_embeddings = torch.cat(
                                [seq_payloads[feature] for feature in additional_features],
                                dim=1,
                            )
                            seq_after_additional = (
                                stage_tensors["content_mlp_output"]
                                + stage_tensors["additional_mlp_output"]
                                if "content_mlp_output" in stage_tensors
                                else stage_tensors["additional_mlp_output"]
                            )
                            _emit_debug_event(
                                event="kira_smoke.input_preprocessor_stage_additional_mlp_output",
                                payload={
                                    "additional_embedding_features": additional_features,
                                    "additional_embeddings_input": _tensor_summary(
                                        additional_embeddings,
                                        include_l2=True,
                                    ),
                                    "additional_mlp_output": _tensor_summary(
                                        stage_tensors["additional_mlp_output"],
                                        include_l2=True,
                                    ),
                                    "seq_embeddings_after_additional": _tensor_summary(
                                        seq_after_additional,
                                        include_l2=True,
                                    ),
                                },
                            )

                    if "action_mlp_output" in stage_tensors:
                        if "additional_mlp_output" in stage_tensors and "content_mlp_output" in stage_tensors:
                            seq_before_action = (
                                stage_tensors["content_mlp_output"]
                                + stage_tensors["additional_mlp_output"]
                            )
                        else:
                            seq_before_action = stage_tensors.get(
                                "content_mlp_output",
                                stage_tensors["action_mlp_output"],
                            )
                        seq_after_action = seq_before_action + stage_tensors["action_mlp_output"]
                        _emit_debug_event(
                            event="kira_smoke.input_preprocessor_stage_action_mlp_output",
                            payload={
                                "action_embeddings": (
                                    _tensor_summary(
                                        stage_tensors["action_embeddings"],
                                        include_l2=True,
                                    )
                                    if "action_embeddings" in stage_tensors
                                    else None
                                ),
                                "action_mlp_output": _tensor_summary(
                                    stage_tensors["action_mlp_output"],
                                    include_l2=True,
                                ),
                                "seq_embeddings_after_action": _tensor_summary(
                                    seq_after_action,
                                    include_l2=True,
                                ),
                            },
                        )

                    if (
                        isinstance(seq_lengths, torch.Tensor)
                        and isinstance(seq_payloads, dict)
                        and isinstance(seq_embeddings_in, torch.Tensor)
                        and int(getattr(input_preprocessor, "_max_contextual_seq_len", 0)) > 0
                    ):
                        contextual_feature_to_max_length = getattr(
                            input_preprocessor,
                            "_contextual_feature_to_max_length",
                        )
                        contextual_feature_to_min_uih_length = getattr(
                            input_preprocessor,
                            "_contextual_feature_to_min_uih_length",
                        )
                        contextual_feature_order = list(
                            contextual_feature_to_max_length.keys()
                        )
                        helper_contextual_input_embeddings = (
                            kira_preprocessors_module.get_contextual_input_embeddings(
                                seq_lengths=seq_lengths,
                                seq_payloads=seq_payloads,
                                contextual_feature_to_max_length=contextual_feature_to_max_length,
                                contextual_feature_to_min_uih_length=contextual_feature_to_min_uih_length,
                                dtype=seq_embeddings_in.dtype,
                            )
                        )
                        contextual_inputs_by_feature: Dict[str, Dict[str, Any]] = {}
                        contextual_padded_values: List[torch.Tensor] = []
                        for key, max_len in contextual_feature_to_max_length.items():
                            padded = torch.flatten(
                                kira_preprocessors_module.jagged_to_padded_dense(
                                    values=seq_payloads[key].to(seq_embeddings_in.dtype),
                                    offsets=[seq_payloads[key + "_offsets"]],
                                    max_lengths=[max_len],
                                    padding_value=0.0,
                                ),
                                1,
                                2,
                            )
                            min_uih_length = contextual_feature_to_min_uih_length.get(
                                key, 0
                            )
                            if min_uih_length > 0:
                                padded = padded * (
                                    seq_lengths.view(-1, 1) >= min_uih_length
                                )
                            contextual_padded_values.append(padded)
                            contextual_inputs_by_feature[key] = {
                                "max_len": int(max_len),
                                "min_uih_length": int(min_uih_length),
                                "raw_values": _tensor_summary(
                                    seq_payloads[key],
                                    include_l2=True,
                                ),
                                "offsets": _tensor_summary(seq_payloads[key + "_offsets"]),
                                "padded_values": _tensor_summary(
                                    padded,
                                    include_l2=True,
                                ),
                            }
                        manual_contextual_input_embeddings = torch.cat(
                            contextual_padded_values,
                            dim=1,
                        )
                        abs_diff = (
                            manual_contextual_input_embeddings.detach().float().cpu()
                            - helper_contextual_input_embeddings.detach().float().cpu()
                        ).abs()
                        contextual_input_embeddings = helper_contextual_input_embeddings
                        _emit_debug_event(
                            event="kira_smoke.input_preprocessor_stage_contextual_input_components",
                            payload={
                                "contextual_feature_order": contextual_feature_order,
                                "contextual_inputs_by_feature": contextual_inputs_by_feature,
                                "contextual_input_embeddings": _tensor_summary(
                                    contextual_input_embeddings,
                                    include_l2=True,
                                ),
                            },
                        )
                        _emit_debug_event(
                            event="kira_smoke.input_preprocessor_stage_contextual_input_equivalence",
                            payload={
                                "manual_contextual_input_embeddings": _tensor_summary(
                                    manual_contextual_input_embeddings,
                                    include_l2=True,
                                ),
                                "helper_contextual_input_embeddings": _tensor_summary(
                                    helper_contextual_input_embeddings,
                                    include_l2=True,
                                ),
                                "manual_vs_helper_max_abs_diff": float(abs_diff.max().item()),
                                "manual_vs_helper_mean_abs_diff": float(abs_diff.mean().item()),
                            },
                        )
                        if not getattr(
                            input_preprocessor,
                            "_kira_contextual_weights_logged",
                            False,
                        ):
                            slot_feature_names: List[str] = []
                            for key, max_len in contextual_feature_to_max_length.items():
                                for idx in range(max_len):
                                    slot_feature_names.append(f"{key}[{idx}]")
                            _emit_debug_event(
                                event="kira_smoke.input_preprocessor_stage_contextual_weights_once",
                                payload={
                                    "contextual_feature_order": contextual_feature_order,
                                    "slot_feature_names": slot_feature_names,
                                    "batched_contextual_linear_weights": _tensor_summary(
                                        input_preprocessor._batched_contextual_linear_weights,  # pyre-ignore[16]
                                        include_l2=True,
                                    ),
                                    "batched_contextual_linear_bias": _tensor_summary(
                                        input_preprocessor._batched_contextual_linear_bias,  # pyre-ignore[16]
                                        include_l2=True,
                                    ),
                                    "slot_weights": {
                                        slot_feature_names[i]: _tensor_summary(
                                            input_preprocessor._batched_contextual_linear_weights[i],  # pyre-ignore[16]
                                            include_l2=True,
                                        )
                                        for i in range(len(slot_feature_names))
                                    },
                                    "slot_bias": {
                                        slot_feature_names[i]: _tensor_summary(
                                            input_preprocessor._batched_contextual_linear_bias[i],  # pyre-ignore[16]
                                            include_l2=True,
                                        )
                                        for i in range(len(slot_feature_names))
                                    },
                                },
                            )
                            input_preprocessor._kira_contextual_weights_logged = True  # pyre-ignore[16]
                        contextual_embeddings = torch.baddbmm(
                            input_preprocessor._batched_contextual_linear_bias.view(  # pyre-ignore[16]
                                -1,
                                1,
                                input_preprocessor._output_embedding_dim,  # pyre-ignore[16]
                            ).to(contextual_input_embeddings.dtype),
                            contextual_input_embeddings.view(
                                -1,
                                input_preprocessor._max_contextual_seq_len,  # pyre-ignore[16]
                                input_preprocessor._input_embedding_dim,  # pyre-ignore[16]
                            ).transpose(0, 1),
                            input_preprocessor._batched_contextual_linear_weights.to(  # pyre-ignore[16]
                                contextual_input_embeddings.dtype
                            ),
                        ).transpose(0, 1)
                        _emit_debug_event(
                            event="kira_smoke.input_preprocessor_stage_contextual_linear_output",
                            payload={
                                "contextual_input_embeddings": _tensor_summary(
                                    contextual_input_embeddings,
                                    include_l2=True,
                                ),
                                "contextual_embeddings": _tensor_summary(
                                    contextual_embeddings,
                                    include_l2=True,
                                ),
                            },
                        )
                        _emit_debug_event(
                            event="kira_smoke.input_preprocessor_stage_contextual_concat_output",
                            payload={
                                "seq_embeddings_after_contextual_concat": _tensor_summary(
                                    out_seq_embeddings,
                                    include_l2=True,
                                ),
                                "seq_timestamps_after_contextual_concat": _tensor_summary(
                                    out_seq_timestamps,
                                ),
                            },
                        )

                    _emit_debug_event(
                        event="kira_smoke.user_transducer_after_input_preprocessor",
                        payload={
                            "max_seq_len": int(out_max_seq_len),
                            "total_uih_len": int(out_total_uih_len),
                            "total_targets": int(out_total_targets),
                            "seq_lengths": _tensor_summary(out_seq_lengths),
                            "seq_offsets": _tensor_summary(out_seq_offsets),
                            "seq_timestamps": _tensor_summary(out_seq_timestamps),
                            "seq_embeddings": _tensor_summary(
                                out_seq_embeddings,
                                include_l2=True,
                            ),
                            "num_targets": _tensor_summary(out_num_targets),
                            "seq_payloads": {
                                k: _tensor_summary(v)
                                for k, v in out_seq_payloads.items()
                            },
                        },
                    )
                    return out

                wrapped_input_preprocessor_forward._kira_user_debug_wrapped = True  # type: ignore[attr-defined]
                input_preprocessor.forward = wrapped_input_preprocessor_forward

        positional_encoder = getattr(hstu_transducer, "_positional_encoder", None)
        if positional_encoder is not None:
            original_positional_encoder_forward = positional_encoder.forward
            if not getattr(
                original_positional_encoder_forward,
                "_kira_user_debug_wrapped",
                False,
            ):
                def wrapped_positional_encoder_forward(*args: Any, **kwargs: Any):
                    seq_lengths = kwargs.get(
                        "seq_lengths",
                        args[1] if len(args) > 1 else None,
                    )
                    seq_offsets = kwargs.get(
                        "seq_offsets",
                        args[2] if len(args) > 2 else None,
                    )
                    seq_timestamps = kwargs.get(
                        "seq_timestamps",
                        args[3] if len(args) > 3 else None,
                    )
                    seq_embeddings_in = kwargs.get(
                        "seq_embeddings",
                        args[4] if len(args) > 4 else None,
                    )
                    num_targets = kwargs.get(
                        "num_targets",
                        args[5] if len(args) > 5 else None,
                    )
                    max_seq_len = kwargs.get(
                        "max_seq_len",
                        args[0] if len(args) > 0 else None,
                    )
                    out = original_positional_encoder_forward(*args, **kwargs)
                    _emit_debug_event(
                        event="kira_smoke.user_transducer_after_positional_encoder",
                        payload={
                            "max_seq_len": (
                                int(max_seq_len)
                                if isinstance(max_seq_len, int)
                                else None
                            ),
                            "seq_lengths": (
                                _tensor_summary(seq_lengths)
                                if isinstance(seq_lengths, torch.Tensor)
                                else None
                            ),
                            "seq_offsets": (
                                _tensor_summary(seq_offsets)
                                if isinstance(seq_offsets, torch.Tensor)
                                else None
                            ),
                            "seq_timestamps": (
                                _tensor_summary(seq_timestamps)
                                if isinstance(seq_timestamps, torch.Tensor)
                                else None
                            ),
                            "seq_embeddings_input": (
                                _tensor_summary(seq_embeddings_in, include_l2=True)
                                if isinstance(seq_embeddings_in, torch.Tensor)
                                else None
                            ),
                            "seq_embeddings_output": _tensor_summary(
                                out,
                                include_l2=True,
                            ),
                            "num_targets": (
                                _tensor_summary(num_targets)
                                if isinstance(num_targets, torch.Tensor)
                                else None
                            ),
                        },
                    )
                    return out

                wrapped_positional_encoder_forward._kira_user_debug_wrapped = True  # type: ignore[attr-defined]
                positional_encoder.forward = wrapped_positional_encoder_forward

        original_transducer_preprocess = hstu_transducer._preprocess
        if not getattr(original_transducer_preprocess, "_kira_user_debug_wrapped", False):
            def wrapped_transducer_preprocess(*args: Any, **kwargs: Any):
                out = original_transducer_preprocess(*args, **kwargs)
                (
                    out_max_seq_len,
                    out_total_uih_len,
                    out_total_targets,
                    out_seq_lengths,
                    out_seq_offsets,
                    out_seq_timestamps,
                    out_seq_embeddings,
                    out_num_targets,
                    out_seq_payloads,
                ) = out
                _emit_debug_event(
                    event="kira_smoke.user_transducer_preprocess_output",
                    payload={
                        "max_seq_len": int(out_max_seq_len),
                        "total_uih_len": int(out_total_uih_len),
                        "total_targets": int(out_total_targets),
                        "seq_lengths": _tensor_summary(out_seq_lengths),
                        "seq_offsets": _tensor_summary(out_seq_offsets),
                        "seq_timestamps": _tensor_summary(out_seq_timestamps),
                        "seq_embeddings": _tensor_summary(
                            out_seq_embeddings,
                            include_l2=True,
                        ),
                        "num_targets": _tensor_summary(out_num_targets),
                        "seq_payloads": {
                            k: _tensor_summary(v)
                            for k, v in out_seq_payloads.items()
                        },
                    },
                )
                return out

            wrapped_transducer_preprocess._kira_user_debug_wrapped = True  # type: ignore[attr-defined]
            hstu_transducer._preprocess = wrapped_transducer_preprocess

        original_transducer_hstu_compute = hstu_transducer._hstu_compute
        if not getattr(original_transducer_hstu_compute, "_kira_user_debug_wrapped", False):
            def wrapped_transducer_hstu_compute(*args: Any, **kwargs: Any):
                out = original_transducer_hstu_compute(*args, **kwargs)
                _emit_debug_event(
                    event="kira_smoke.user_transducer_hstu_output",
                    payload={
                        "encoded_seq_embeddings": _tensor_summary(
                            out,
                            include_l2=True,
                        ),
                    },
                )
                return out

            wrapped_transducer_hstu_compute._kira_user_debug_wrapped = True  # type: ignore[attr-defined]
            hstu_transducer._hstu_compute = wrapped_transducer_hstu_compute

        original_transducer_postprocess = hstu_transducer._postprocess
        if not getattr(original_transducer_postprocess, "_kira_user_debug_wrapped", False):
            def wrapped_transducer_postprocess(*args: Any, **kwargs: Any):
                out = original_transducer_postprocess(*args, **kwargs)
                full_embeddings, candidate_embeddings = out
                _emit_debug_event(
                    event="kira_smoke.user_transducer_postprocess_output",
                    payload={
                        "candidate_embeddings": _tensor_summary(
                            candidate_embeddings,
                            include_l2=True,
                        ),
                        "full_seq_embeddings": (
                            _tensor_summary(full_embeddings, include_l2=True)
                            if isinstance(full_embeddings, torch.Tensor)
                            else None
                        ),
                    },
                )
                return out

            wrapped_transducer_postprocess._kira_user_debug_wrapped = True  # type: ignore[attr-defined]
            hstu_transducer._postprocess = wrapped_transducer_postprocess

        stu_module = getattr(hstu_transducer, "_stu_module", None)
        stu_layers = getattr(stu_module, "_stu_layers", None)
        if isinstance(stu_layers, torch.nn.ModuleList):
            for layer_idx, layer in enumerate(stu_layers):
                original_layer_forward = layer.forward
                if getattr(original_layer_forward, "_kira_user_debug_wrapped", False):
                    continue

                def _make_wrapped_layer_forward(
                    index: int,
                    original_forward: Any,
                ):
                    def wrapped_layer_forward(*args: Any, **kwargs: Any):
                        x = kwargs.get("x", args[0] if len(args) > 0 else None)
                        x_lengths = kwargs.get(
                            "x_lengths",
                            args[1] if len(args) > 1 else None,
                        )
                        x_offsets = kwargs.get(
                            "x_offsets",
                            args[2] if len(args) > 2 else None,
                        )
                        num_targets = kwargs.get(
                            "num_targets",
                            args[4] if len(args) > 4 else None,
                        )
                        max_seq_len = kwargs.get(
                            "max_seq_len",
                            args[3] if len(args) > 3 else None,
                        )
                        _emit_debug_event(
                            event="kira_smoke.user_transducer_stu_layer_input",
                            payload={
                                "layer_index": int(index),
                                "max_seq_len": (
                                    int(max_seq_len)
                                    if isinstance(max_seq_len, int)
                                    else None
                                ),
                                "x": (
                                    _tensor_summary(x, include_l2=True)
                                    if isinstance(x, torch.Tensor)
                                    else None
                                ),
                                "x_lengths": (
                                    _tensor_summary(x_lengths)
                                    if isinstance(x_lengths, torch.Tensor)
                                    else None
                                ),
                                "x_offsets": (
                                    _tensor_summary(x_offsets)
                                    if isinstance(x_offsets, torch.Tensor)
                                    else None
                                ),
                                "num_targets": (
                                    _tensor_summary(num_targets)
                                    if isinstance(num_targets, torch.Tensor)
                                    else None
                                ),
                            },
                        )

                        out = original_forward(*args, **kwargs)

                        _emit_debug_event(
                            event="kira_smoke.user_transducer_stu_layer_output",
                            payload={
                                "layer_index": int(index),
                                "x": _tensor_summary(out, include_l2=True),
                            },
                        )
                        return out

                    wrapped_layer_forward._kira_user_debug_wrapped = True  # type: ignore[attr-defined]
                    return wrapped_layer_forward

                layer.forward = _make_wrapped_layer_forward(  # pyre-ignore[8]
                    index=layer_idx,
                    original_forward=original_layer_forward,
                )

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
    user_transducer_params = {
        name: p.detach()
        for name, p in dense_model._hstu_transducer.named_parameters()  # pyre-ignore[16]
    }
    _emit_debug_event(
        event="kira_smoke.user_forward_weights_once",
        payload={
            "hstu_transducer": {
                name: _tensor_summary(param)
                for name, param in user_transducer_params.items()
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

    original_user_forward = dense_model._user_forward  # pyre-ignore[16]

    def wrapped_user_forward(*args: Any, **kwargs: Any):
        max_uih_len = kwargs.get("max_uih_len", args[0] if len(args) > 0 else None)
        max_candidates = kwargs.get("max_candidates", args[1] if len(args) > 1 else None)
        seq_embeddings = kwargs.get("seq_embeddings", args[2] if len(args) > 2 else None)
        payload_features = kwargs.get("payload_features", args[3] if len(args) > 3 else None)
        num_candidates = kwargs.get("num_candidates", args[4] if len(args) > 4 else None)

        if (
            isinstance(seq_embeddings, dict)
            and isinstance(payload_features, dict)
            and isinstance(num_candidates, torch.Tensor)
            and max_uih_len is not None
            and max_candidates is not None
        ):
            post_id_feature = dense_model._hstu_configs.uih_post_id_feature_name  # pyre-ignore[16]
            action_time_feature = dense_model._hstu_configs.uih_action_time_feature_name  # pyre-ignore[16]
            query_time_feature = dense_model._hstu_configs.candidates_querytime_feature_name  # pyre-ignore[16]

            source_lengths = seq_embeddings[post_id_feature].lengths
            source_timestamps = concat_2D_jagged(
                max_seq_len=int(max_uih_len) + int(max_candidates),
                max_len_left=int(max_uih_len),
                offsets_left=payload_features["uih_offsets"],
                values_left=payload_features[action_time_feature].unsqueeze(-1),
                max_len_right=int(max_candidates),
                offsets_right=payload_features["candidate_offsets"],
                values_right=payload_features[query_time_feature].unsqueeze(-1),
                kernel=dense_model.hammer_kernel(),  # pyre-ignore[16]
            ).squeeze(-1)
            embedding = seq_embeddings[post_id_feature].embedding
            seq_payload = dense_model._construct_payload(  # pyre-ignore[16]
                payload_features=payload_features,
                seq_embeddings=seq_embeddings,
            )

            _emit_debug_event(
                event="kira_smoke.user_forward_input_features",
                payload={
                    "source_lengths": _tensor_summary(source_lengths),
                    "source_timestamps": _tensor_summary(source_timestamps),
                    "embedding": _tensor_summary(embedding, include_l2=True),
                    "num_candidates": _tensor_summary(num_candidates),
                    "total_targets": int(num_candidates.sum().item()),
                    "seq_payloads": {
                        k: _tensor_summary(v)
                        for k, v in seq_payload.items()
                    },
                },
            )

        user_embeddings = original_user_forward(*args, **kwargs)
        _emit_debug_event(
            event="kira_smoke.user_forward_output",
            payload={
                "candidates_user_embeddings": _tensor_summary(
                    user_embeddings,
                    include_l2=True,
                ),
            },
        )
        return user_embeddings

    wrapped_user_forward._kira_dense_debug_wrapped = True  # type: ignore[attr-defined]
    dense_model._user_forward = wrapped_user_forward  # pyre-ignore[16]


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
        default=20,
        help="Number of top candidates to print (default: 20).",
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
        {
            "item_movie_id": 1,
            "item_query_time": 978310000,
            "item_dummy_watchtime": 0,
            "item_movie_rating": 0,
            "item_action_weights": 0,
        },
        {
            "item_movie_id": 260,
            "item_query_time": 978310000,
            "item_dummy_watchtime": 0,
            "item_movie_rating": 0,
            "item_action_weights": 0,
        },
        {
            "item_movie_id": 1196,
            "item_query_time": 978310000,
            "item_dummy_watchtime": 0,
            "item_movie_rating": 0,
            "item_action_weights": 0,
        },
        {
            "item_movie_id": 1210,
            "item_query_time": 978310000,
            "item_dummy_watchtime": 0,
            "item_movie_rating": 0,
            "item_action_weights": 0,
        },
        {
            "item_movie_id": 2028,
            "item_query_time": 978310000,
            "item_dummy_watchtime": 0,
            "item_movie_rating": 0,
            "item_action_weights": 0,
        },
        {
            "item_movie_id": 2762,
            "item_query_time": 978310000,
            "item_dummy_watchtime": 0,
            "item_movie_rating": 0,
            "item_action_weights": 0,
        },
        {
            "item_movie_id": 2997,
            "item_query_time": 978310000,
            "item_dummy_watchtime": 0,
            "item_movie_rating": 0,
            "item_action_weights": 0,
        },
        {
            "item_movie_id": 1097,
            "item_query_time": 978310000,
            "item_dummy_watchtime": 0,
            "item_movie_rating": 0,
            "item_action_weights": 0,
        },
        {
            "item_movie_id": 3578,
            "item_query_time": 978310000,
            "item_dummy_watchtime": 0,
            "item_movie_rating": 0,
            "item_action_weights": 0,
        },
        {
            "item_movie_id": 480,
            "item_query_time": 978310000,
            "item_dummy_watchtime": 0,
            "item_movie_rating": 0,
            "item_action_weights": 0,
        },
        {
            "item_movie_id": 589,
            "item_query_time": 978310000,
            "item_dummy_watchtime": 0,
            "item_movie_rating": 0,
            "item_action_weights": 0,
        },
        {
            "item_movie_id": 47,
            "item_query_time": 978310000,
            "item_dummy_watchtime": 0,
            "item_movie_rating": 0,
            "item_action_weights": 0,
        },
        {
            "item_movie_id": 50,
            "item_query_time": 978310000,
            "item_dummy_watchtime": 0,
            "item_movie_rating": 0,
            "item_action_weights": 0,
        },
        {
            "item_movie_id": 527,
            "item_query_time": 978310000,
            "item_dummy_watchtime": 0,
            "item_movie_rating": 0,
            "item_action_weights": 0,
        },
        {
            "item_movie_id": 608,
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
    hstu_config.max_num_candidates = max(
        hstu_config.max_num_candidates,
        hstu_config.max_num_candidates_inference,
        20,
    )
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
