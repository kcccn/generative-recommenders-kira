#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""
Generate GT output JSON with Kira native inference for movielens-1m.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


DATASET_NAME = "movielens-1m"
DEFAULT_SEED = 20260318


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate movielens-1m GT JSON with Kira native inference.",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to native checkpoint directory (contains non_sparse.ckpt and sparse/).",
    )
    parser.add_argument(
        "--input-json",
        required=True,
        help="Input benchmark JSON path.",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Output GT JSON path.",
    )
    parser.add_argument(
        "--dataset",
        default=DATASET_NAME,
        help=f"Dataset name (only {DATASET_NAME} is supported).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--config-json",
        type=str,
        default=None,
        help="Optional config.json path with embedding_tables[*].num_embeddings.",
    )
    parser.add_argument(
        "--max-num-embeddings",
        type=int,
        default=None,
        help="Optional upper bound for each table num_embeddings.",
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


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _validate_input_payload(payload: dict[str, Any], dataset: str) -> list[dict[str, Any]]:
    if dataset != DATASET_NAME:
        raise ValueError(f"Unsupported --dataset={dataset}, expected {DATASET_NAME}")

    meta_dataset = payload.get("meta", {}).get("dataset")
    if meta_dataset is not None and meta_dataset != DATASET_NAME:
        raise ValueError(
            f"Unsupported input dataset={meta_dataset}, expected {DATASET_NAME}"
        )

    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError("input JSON must contain a list field: cases")

    seen_case_ids: set[str] = set()
    for case in cases:
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("Every case must contain non-empty string case_id")
        if case_id in seen_case_ids:
            raise ValueError(f"Duplicate case_id found: {case_id}")
        seen_case_ids.add(case_id)

        if not isinstance(case.get("user_history"), list):
            raise ValueError(f"case={case_id} missing user_history list")
        if not isinstance(case.get("candidates"), list):
            raise ValueError(f"case={case_id} missing candidates list")
        if len(case["candidates"]) == 0:
            raise ValueError(f"case={case_id} has empty candidates")
        if not isinstance(case.get("context"), dict):
            raise ValueError(f"case={case_id} missing context dict")
    return cases


def _resolve_config_json_path(model_path: str, config_json_arg: str | None) -> str | None:
    if config_json_arg:
        return config_json_arg
    inferred = os.path.join(model_path, "config.json")
    if os.path.isfile(inferred):
        return inferred
    return None


def _apply_num_embeddings_from_config(table_config: dict[str, Any], config_json_path: str) -> None:
    with open(config_json_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    embedding_tables = cfg.get("embedding_tables")
    if not isinstance(embedding_tables, list):
        return

    cfg_rows_by_name: dict[str, int] = {}
    for table in embedding_tables:
        if isinstance(table, dict) and "name" in table and "num_embeddings" in table:
            cfg_rows_by_name[str(table["name"])] = int(table["num_embeddings"])

    for table_name, emb_cfg in table_config.items():
        if table_name in cfg_rows_by_name:
            emb_cfg.num_embeddings = int(cfg_rows_by_name[table_name])


def _truncate_table_config_num_embeddings(
    table_config: dict[str, Any],
    max_num_embeddings: int | None,
) -> None:
    if max_num_embeddings is None:
        return
    if max_num_embeddings <= 0:
        raise ValueError(
            f"--max-num-embeddings must be > 0, got {max_num_embeddings}"
        )
    for emb_cfg in table_config.values():
        emb_cfg.num_embeddings = min(int(emb_cfg.num_embeddings), max_num_embeddings)


def _build_native_samples(
    hstu_config: Any,
    user_history: list[dict[str, int]],
    candidates: list[dict[str, int]],
    context: dict[str, int],
) -> Any:
    from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

    from generative_recommenders.dlrm_v3.datasets.dataset import Samples

    seq_len = len(user_history)
    num_candidates = len(candidates)

    uih_keys = list(hstu_config.hstu_uih_feature_names)
    contextual_keys = set(hstu_config.contextual_feature_to_max_length.keys())

    uih_values: list[int] = []
    uih_lengths: list[int] = []
    for key in uih_keys:
        if key in contextual_keys:
            uih_values.append(int(context.get(key, 0)))
            uih_lengths.append(1)
        else:
            values = [int(row.get(key, 0)) for row in user_history]
            uih_values.extend(values)
            uih_lengths.append(seq_len)

    cand_keys = list(hstu_config.hstu_candidate_feature_names)
    cand_values: list[int] = []
    cand_lengths: list[int] = []
    for key in cand_keys:
        values = [int(row.get(key, 0)) for row in candidates]
        cand_values.extend(values)
        cand_lengths.append(num_candidates)

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
    return Samples(
        uih_features_kjt=uih_features_kjt,
        candidates_features_kjt=candidates_features_kjt,
    )


def _extract_scores(mt_target_preds: torch.Tensor, expected_num_candidates: int) -> list[float]:
    scores = mt_target_preds.detach().cpu().float().reshape(-1)
    if scores.numel() != expected_num_candidates:
        raise ValueError(
            "Prediction length mismatch: "
            f"numel={scores.numel()} expected={expected_num_candidates}"
        )
    return [float(x) for x in scores.tolist()]


def _build_output_payload(
    input_payload: dict[str, Any],
    result_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    in_meta = input_payload.get("meta", {})
    return {
        "meta": {
            "dataset": in_meta.get("dataset", DATASET_NAME),
            "num_cases": len(result_cases),
        },
        "cases": result_cases,
    }


def main() -> int:
    args = _parse_args()
    _validate_model_path(args.model_path)

    from generative_recommenders.dlrm_v3.configs import (
        get_embedding_table_config,
        get_hstu_configs,
    )
    from generative_recommenders.dlrm_v3.inference.inference_modules import (
        set_is_inference,
    )
    from generative_recommenders.dlrm_v3.inference.model_family import HSTUModelFamily

    input_payload = _load_json(args.input_json)
    input_cases = _validate_input_payload(input_payload, dataset=args.dataset)

    os.environ["WORLD_SIZE"] = "1"
    _set_seed(args.seed)
    set_is_inference(is_inference=True)

    hstu_config = get_hstu_configs(args.dataset)
    hstu_config.max_num_candidates = hstu_config.max_num_candidates_inference
    table_config = get_embedding_table_config(args.dataset)

    config_json_path = _resolve_config_json_path(args.model_path, args.config_json)
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
    model_family.load(args.model_path)

    result_cases: list[dict[str, Any]] = []
    total_cases = len(input_cases)
    for idx, case in enumerate(input_cases, start=1):
        candidates = case["candidates"]
        samples = _build_native_samples(
            hstu_config=hstu_config,
            user_history=case["user_history"],
            candidates=candidates,
            context=case["context"],
        )
        pred_output = model_family.predict(samples)
        if pred_output is None:
            raise RuntimeError("predict returned None unexpectedly")
        mt_target_preds, _, _, _, _ = pred_output
        scores = _extract_scores(mt_target_preds, expected_num_candidates=len(candidates))

        result_cases.append(
            {
                "case_id": case["case_id"],
                "scores": [
                    {
                        "candidate_index": cand_idx,
                        "item_movie_id": int(cand["item_movie_id"]),
                        "score": float(score),
                    }
                    for cand_idx, (cand, score) in enumerate(zip(candidates, scores))
                ],
            }
        )
        if idx % 20 == 0 or idx == total_cases:
            print(f"progress: {idx}/{total_cases}")

    output_payload = _build_output_payload(input_payload, result_cases)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote GT output: {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[acc_output] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
