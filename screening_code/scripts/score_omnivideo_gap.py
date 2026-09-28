#!/usr/bin/env python3
"""Resumable first-token MCQ scoring against an existing vLLM service."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scripts.audit_omnivideo_score_media import (
    audited_candidates,
    validate_observed_media,
)
from scripts.prepare_omnivideo_exact_gold_candidates import (
    CONTRACT,
    _read,
    _sha256,
    binding,
)


def post(endpoint, payload, timeout=300):
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
            request, timeout=timeout
        ) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise ValueError(
            f"HTTP {e.code}: {e.read().decode(errors='replace')[:4000]}"
        ) from e


def option_probabilities(response, token_ids):
    entries = response["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    scores = {}
    for letter, token in token_ids.items():
        values = [
            e["logprob"] for e in entries if e["token"] in (letter, f"token_id:{token}")
        ]
        if len(values) != 1 or not math.isfinite(values[0]) or values[0] < -1e20:
            raise ValueError(
                f"missing/invalid exact logprob for {letter}; do not approximate"
            )
        scores[letter] = values[0]
    offset = max(scores.values())
    weights = {k: math.exp(v - offset) for k, v in scores.items()}
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}, scores


def complete_option_logprobs(endpoint, payload, response, token_ids):
    """vLLM returns raw (pre-mask) logprobs: request missing letters explicitly.

    Never replace missing probabilities with zero or a top-k lower bound.
    Backends returning post-mask degenerate logprobs are rejected.
    """
    import copy

    response = copy.deepcopy(response)
    entries = response["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    calls = 0
    for letter, token in token_ids.items():
        if any(e["token"] in (letter, f"token_id:{token}") for e in entries):
            continue
        forced = dict(payload, allowed_token_ids=[token])
        extra = post(endpoint, forced)
        calls += 1
        generated = extra["choices"][0]["logprobs"]["content"][0]
        if (
            generated["token"] not in (letter, f"token_id:{token}")
            or not generated["logprob"] < -1e-7
        ):
            raise ValueError("backend does not expose raw selected-token logprobs")
        entries.append({"token": f"token_id:{token}", "logprob": generated["logprob"]})
    return response, calls


def model_identity(model_dir, model_name, mode, template):
    names = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "chat_template.json",
        "model.safetensors.index.json",
    ]
    assets = {n: _sha256(model_dir / n) for n in names if (model_dir / n).is_file()}
    if "tokenizer.json" not in assets or "config.json" not in assets:
        raise ValueError("local model/tokenizer files required")
    return dict(
        model=model_name,
        model_path=str(model_dir.resolve()),
        asset_sha256=assets,
        av_mode=mode,
        prompt_sha256=hashlib.sha256(template.encode()).hexdigest(),
        reconstruction_code_sha256={
            "scorer": _sha256(Path(__file__)),
            "media": _sha256(
                Path(__file__).resolve().parents[1] / "src/omni_opsd/selection/media.py"
            ),
            "joint_server_adapter": _sha256(
                Path(__file__).resolve().parents[1]
                / "compat/qwen25_joint/joint_patch.py"
            )
            if mode == "joint"
            else None,
        },
        weights_verification="local metadata hashes; server weights not remotely hashed",
    )


PROMPT = "Question: {question}\n\nOptions:\n{choices}\n\nAnswer with exactly one uppercase option letter (A, B, C, or D)."


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--canonical", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--model", default="Qwen2.5-Omni-7B")
    p.add_argument("--endpoint", default="http://gpu02:8091/v1/chat/completions")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--av-mode", choices=["split", "joint"], default="joint")
    p.add_argument(
        "--cache-root", type=Path, default=Path("data/cache/omnivideo_5k/processed")
    )
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--seed", type=int, default=20260906)
    p.add_argument("--views", default="av,gold", help="av,gold,text")
    a = p.parse_args()
    models_url = a.endpoint.rsplit("/v1/", 1)[0] + "/v1/models"
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
        models_url, timeout=10
    ) as r:
        served = json.load(r)
    matches = [m for m in served["data"] if m["id"] == a.model]
    if (
        len(matches) != 1
        or Path(matches[0].get("root", "")).resolve() != a.model_dir.resolve()
    ):
        raise ValueError(
            "served model root does not match local tokenizer/processor assets"
        )
    from transformers import AutoProcessor

    from omni_opsd.selection.media import prepare

    processor = AutoProcessor.from_pretrained(
        a.model_dir, local_files_only=True, use_fast=False
    )
    tokenizer = processor.tokenizer
    token_ids = {}
    for k in "ABCD":
        ids = tokenizer.encode(k, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError("ABCD must each encode as one token")
        token_ids[k] = ids[0]
    # Joint mode requires server --trust-request-chat-template. Remove the standalone
    # audio placeholder because the processor inserts audio inside each video token.
    template = json.loads((a.model_dir / "chat_template.json").read_text())[
        "chat_template"
    ]
    joint_template = template.replace("<|audio_bos|><|AUDIO|><|audio_eos|>", "")
    identity = model_identity(
        a.model_dir,
        a.model,
        a.av_mode,
        PROMPT + (joint_template if a.av_mode == "joint" else template),
    )
    rows, excluded = audited_candidates(a.canonical, a.audit)
    if a.limit:
        rows = rows[: a.limit]
    views = a.views.split(",")
    if not views or set(views) - {"av", "gold", "text"}:
        raise ValueError("unsupported view")
    audit = json.loads(a.audit.read_text())
    by_video = {x["video_id"]: x for x in audit["items"]}
    a.output_dir.mkdir(parents=True, exist_ok=True)
    lock = (a.output_dir / "score.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    import importlib.metadata

    run = dict(
        canonical_sha256=_sha256(a.canonical),
        audit_sha256=_sha256(a.audit),
        model_identity=identity,
        sampling_contract=CONTRACT,
        seed=a.seed,
        client_versions={
            k: importlib.metadata.version(k)
            for k in ("transformers", "av", "numpy", "Pillow")
        },
        server_max_model_len=matches[0].get("max_model_len"),
        endpoint=a.endpoint,
        selected_ids=[r["sample_id"] for r in rows],
        views=views,
        historical_match=False,
        protocol_note="split corresponds to C0/C3; joint uses repository paired-item V1 adapter (server deployment verified separately); historical prompt unavailable",
    )
    run_path = a.output_dir / "run.json"
    if run_path.exists() and json.loads(run_path.read_text()) != run:
        raise ValueError("resume configuration differs; use a new output directory")
    run_path.write_text(json.dumps(run, indent=2) + "\n")
    done = {}
    for view in views:
        path = a.output_dir / f"{view}.jsonl"
        done[view] = {}
        for row in _read(path) if path.exists() else []:
            if row["sample_id"] in done[view]:
                raise ValueError("duplicate completed score")
            done[view][row["sample_id"]] = row
    sources = {r["sample_id"]: r for r in rows}
    from scripts.prepare_omnivideo_exact_gold_candidates import _validate_exact_contract
    from select_omnivideo_gap_5000 import _validate_probabilities

    for view in views:
        for sid, result in done[view].items():
            if sid not in sources or result["model_identity"] != identity:
                raise ValueError("resumed score source/model mismatch")
            _validate_exact_contract(result, sources[sid], view)
            _validate_probabilities(result, view)
            if view != "text":
                validate_observed_media(result, sources[sid], view)
    # Rehash only candidate source files, once per video per invocation.
    for vid in {r["video_id"] for r in rows}:
        row = next(r for r in rows if r["video_id"] == vid)
        if _sha256(row["video_path"]) != by_video[vid]["media_sha256"]:
            raise ValueError("source media changed since audit")

    def score_one(job):
        row, view = job
        start = time.monotonic()
        try:
            content, observed = [], None
            if view != "text":
                spans = (
                    [[0.0, row["duration"]]] if view == "av" else row["evidence_spans"]
                )
                content, observed = prepare(
                    row, spans, by_video[row["video_id"]], processor, a.cache_root
                )
            prompt = PROMPT.format(
                question=row["question"],
                choices="\n".join(f"{k}. {v}" for k, v in zip("ABCD", row["choices"])),
            )
            content.append({"type": "text", "text": prompt})
            payload = dict(
                model=a.model,
                messages=[{"role": "user", "content": content}],
                temperature=1.0,
                max_tokens=1,
                seed=a.seed,
                logprobs=True,
                top_logprobs=20,
                allowed_token_ids=list(token_ids.values()),
                return_tokens_as_token_ids=True,
                mm_processor_kwargs=dict(
                    min_pixels=3136,
                    max_pixels=28672,
                    size={"shortest_edge": 3136, "longest_edge": 28672},
                    fps=2.0,
                    do_sample_frames=False,
                    use_audio_in_video=a.av_mode == "joint" and view != "text",
                ),
            )
            if a.av_mode == "joint" and view != "text":
                payload["chat_template"] = joint_template
            # Temperature 1 retains the soft distribution. Prediction is our argmax,
            # independent of the sampled output token. Restriction preserves ABCD ratios.
            for attempt in range(a.retries + 1):
                try:
                    response = post(a.endpoint, payload)
                    break
                except Exception:
                    if attempt == a.retries:
                        raise
                    time.sleep(min(2**attempt, 4))
            response, extra_calls = complete_option_logprobs(
                a.endpoint, payload, response, token_ids
            )
            probabilities, logprobs = option_probabilities(response, token_ids)
            predicted = max(probabilities, key=probabilities.get)
            result = dict(
                sample_id=row["sample_id"],
                answer=row["answer"],
                source_binding=binding(row),
                model_identity=identity,
                sampling_contract=CONTRACT,
                scores={
                    view: dict(
                        probabilities=probabilities,
                        predicted=predicted,
                        logprobs=logprobs,
                        observed_media=observed,
                    )
                },
                answer_probability={view: probabilities[row["answer"]]},
                prompt_tokens=response.get("usage", {}).get("prompt_tokens"),
                response_model=response.get("model"),
                elapsed_sec=time.monotonic() - start,
            )
            result["extra_option_requests"] = extra_calls
            if view != "text":
                validate_observed_media(result, row, view)
            return view, result, None
        except Exception as e:
            return (
                view,
                None,
                dict(
                    sample_id=row["sample_id"],
                    view=view,
                    observed_media=observed,
                    error=f"{type(e).__name__}: {e}",
                ),
            )

    jobs = [(r, v) for r in rows for v in views if r["sample_id"] not in done[v]]
    failures = 0
    with ThreadPoolExecutor(a.workers) as pool:
        for i, (view, result, error) in enumerate(pool.map(score_one, jobs)):
            path = a.output_dir / ("failures.jsonl" if error else f"{view}.jsonl")
            with path.open("a") as f:
                f.write(
                    json.dumps(error or result, ensure_ascii=False, allow_nan=False)
                    + "\n"
                )
                f.flush()
                os.fsync(f.fileno())
            failures += int(error is not None)
            print(
                json.dumps(
                    dict(
                        progress=f"{i + 1}/{len(jobs)}",
                        view=view,
                        error=error,
                        sample_id=(error or result)["sample_id"],
                    )
                ),
                flush=True,
            )
    if failures:
        raise SystemExit(
            f"{failures} failed requests; rerun same command to retry; no success marker"
        )
    (a.output_dir / "SCORES_SUCCESS").write_text(
        "All requested IDs and views complete.\n"
    )


if __name__ == "__main__":
    main()
