#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""
Single-file smoke demo for DLRMv3 inference on movielens-1m.

The input prompt semantics are hardcoded to align with vllm-GR's
examples/load_converted_model.py demo, while using native
generative-recommenders inference APIs only.
"""

import argparse
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

from generative_recommenders.dlrm_v3.configs import (
    get_embedding_table_config,
    get_hstu_configs,
)
from generative_recommenders.dlrm_v3.datasets.dataset import Samples
from generative_recommenders.dlrm_v3.inference.inference_modules import set_is_inference
from generative_recommenders.dlrm_v3.inference.model_family import HSTUModelFamily


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


def main() -> None:
    args = _parse_args()
    _validate_model_path(args.model_path)

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

    model_family.load(args.model_path)
    try:
        pred_output = model_family.predict(samples)
        if pred_output is None:
            raise RuntimeError("predict returned None unexpectedly")
        mt_target_preds, _, _, dt_sparse, dt_dense = pred_output
    finally:
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
