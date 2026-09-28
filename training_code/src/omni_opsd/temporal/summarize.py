"""Aggregate experiment JSONL outputs and emit CSV/plot-ready artifacts."""

from __future__ import annotations

import csv
from pathlib import Path
import json
import math
from typing import Any

from .bootstrap import paired_bootstrap_table, read_prediction_rows
from .evidence import evidence_duration_bucket, evidence_ratio_bucket, video_duration_bucket

try:
    import yaml
except Exception:  # pragma: no cover - plan-only environments may omit PyYAML
    yaml = None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _prediction_files(directory: Path) -> list[Path]:
    merged = directory / "predictions.jsonl"
    if merged.is_file():
        return [merged]
    return sorted(directory.glob("predictions.shard*.jsonl"))


def _run_config(directory: Path) -> dict[str, Any]:
    candidates = [directory / "config.yaml"] + sorted(directory.glob("config.shard*.yaml"))
    for path in candidates:
        if not path.is_file() or yaml is None:
            continue
        try:
            value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        if isinstance(value, dict):
            return value
    return {}


def _numeric(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, (list, tuple)):
        values = [float(item) for item in value if isinstance(item, (int, float))]
        return sum(values) / len(values) if values else None
    return None


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = []
    for row in rows:
        value = row.get(key) if key in row else row.get("resource", {}).get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
    return sum(values) / len(values) if values else None


def _accuracy(rows: list[dict[str, Any]]) -> float | None:
    valid = [row for row in rows if row.get("correct") is not None]
    return sum(bool(row.get("correct")) for row in valid) / len(valid) if valid else None


def _run_pipeline(config: dict[str, Any]) -> str:
    experiment = config.get("experiment", {}) if isinstance(config.get("experiment", {}), dict) else {}
    return str(config.get("pipeline_family", experiment.get("pipeline_family", "native_qwen")))


def _bin_tokens_per_frame(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "unknown"
    return f"{float(value):.0f}"


def _aggregate_rows(rows: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(name) for name in group_keys)
        groups.setdefault(key, []).append(row)
    result: list[dict[str, Any]] = []
    for key, group_rows in sorted(groups.items(), key=lambda item: repr(item[0])):
        valid = [row for row in group_rows if row.get("correct") is not None]
        output = {name: value for name, value in zip(group_keys, key)}
        output.update({
            "samples": len(group_rows),
            "evaluated": len(valid),
            "correct": sum(bool(row.get("correct")) for row in valid),
            "accuracy": _accuracy(group_rows),
            "mean_visual_tokens": _mean(group_rows, "visual_tokens"),
            "mean_video_tokens": _mean(group_rows, "video_tokens"),
            "mean_image_tokens": _mean(group_rows, "image_tokens"),
            "mean_tokens_per_frame": _mean(group_rows, "tokens_per_frame"),
            "mean_latency_sec": _mean(group_rows, "latency_sec"),
            "mean_peak_gpu_memory_mb": _mean(group_rows, "peak_gpu_memory_mb"),
        })
        result.append(output)
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(
    root: str | Path,
    *,
    reference: str | Path | None = None,
    candidates: list[str | Path] | None = None,
    bootstrap_seed: int = 0,
    bootstrap_resamples: int = 10_000,
    include_cross_pipeline: bool = False,
) -> Path:
    root = Path(root).resolve()
    directories = sorted({path.parent for path in root.rglob("predictions*.jsonl") if path.is_file()})
    summary_rows: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    for directory in directories:
        files = _prediction_files(directory)
        if not files:
            continue
        rows = [row for file in files for row in _read_jsonl(file)]
        if not rows:
            continue
        name = directory.name
        config = _run_config(directory)
        mode = str(config.get("experiment", {}).get("mode", "unknown"))
        pipeline_family = _run_pipeline(config)
        row = {
            "experiment": name,
            "mode": mode,
            "pipeline_family": pipeline_family,
            "samples": len(rows),
            "evaluated": sum(row.get("correct") is not None for row in rows),
            "accuracy": _accuracy(rows),
            "failures": sum(bool(row.get("failure_reason")) for row in rows),
            "mean_frames": _mean(rows, "num_video_frames"),
            "mean_input_tokens": _mean(rows, "input_tokens"),
            "mean_visual_tokens": _mean(rows, "visual_tokens_exact_before_compression"),
            "mean_visual_tokens_per_frame": _mean(rows, "visual_tokens_per_frame"),
            "mean_visual_tokens_per_second": _mean(rows, "visual_tokens_per_second"),
            "mean_latency_sec": _mean(rows, "latency_sec"),
            "max_peak_gpu_memory_mb": max((row.get("resource", {}).get("peak_gpu_memory_mb", 0) or 0 for row in rows), default=0),
        }
        summary_rows.append(row)
        for category_key in ("question_category", "question_type"):
            groups: dict[str, list[dict[str, Any]]] = {}
            for item in rows:
                group = str(item.get(category_key, "unknown"))
                groups.setdefault(group, []).append(item)
            for group, group_rows in groups.items():
                category_rows.append({
                    "experiment": name,
                    "pipeline_family": pipeline_family,
                    "category_key": category_key,
                    "question_type": group,
                    "samples": len(group_rows),
                    "evaluated": sum(item.get("correct") is not None for item in group_rows),
                    "accuracy": _accuracy(group_rows),
                    "mean_visual_tokens": _mean(group_rows, "visual_tokens_exact_before_compression"),
                    "mean_latency_sec": _mean(group_rows, "latency_sec"),
                })
        for item in rows:
            resource = item.get("resource", {})
            config_video = config.get("video", {}) if isinstance(config.get("video", {}), dict) else {}
            config_audio = config.get("audio", {}) if isinstance(config.get("audio", {}), dict) else {}
            config_subtitle = config.get("subtitle", {}) if isinstance(config.get("subtitle", {}), dict) else {}
            config_evidence = config.get("evidence", {}) if isinstance(config.get("evidence", {}), dict) else {}
            fps = _numeric(item.get("effective_fps"))
            if fps is None:
                fps = _numeric(resource.get("effective_fps"))
            visual_tokens = _numeric(resource.get("visual_tokens_exact_before_compression"))
            raw_rows.append({
                "experiment": name,
                "mode": mode,
                "pipeline_family": pipeline_family,
                "sample_id": item.get("sample_id"),
                "correct": item.get("correct"),
                "question_category": item.get("question_category", "other"),
                "question_category_source": item.get("question_category_source", "unknown"),
                "fps": _numeric(item.get("fps")),
                "effective_fps": fps,
                "effective_fps_values": resource.get("effective_fps"),
                "resolution": item.get("resolution"),
                "configured_resolution": config_video.get("resolution_policy"),
                "num_hr_images": resource.get("num_hr_images"),
                "visual_tokens": visual_tokens,
                "video_tokens": _numeric(resource.get("visual_tokens_video_exact_before_compression")),
                "image_tokens": _numeric(resource.get("visual_tokens_image_exact_before_compression")),
                "tokens_per_frame": _numeric(resource.get("visual_tokens_per_frame")),
                "tokens_per_second": _numeric(resource.get("visual_tokens_per_second")),
                "frames": resource.get("num_video_frames"),
                "latency_sec": resource.get("latency_sec"),
                "peak_gpu_memory_mb": resource.get("peak_gpu_memory_mb"),
                "use_audio": item.get("use_audio", resource.get("audio_enabled")),
                "subtitle": config_subtitle.get("enabled", False),
                "halo_before": config_evidence.get("halo_before"),
                "halo_after": config_evidence.get("halo_after"),
                "iso_budget_proxy_error_percent": resource.get("iso_budget_proxy_error_percent"),
                "iso_budget": (config.get("iso_budget", {}) or {}).get("budget") if isinstance(config.get("iso_budget", {}), dict) else None,
                "iso_policy": (config.get("iso_budget", {}) or {}).get("policy") if isinstance(config.get("iso_budget", {}), dict) else None,
                "processed_height": (resource.get("processed_video_hw") or [{}])[0].get("height") if resource.get("processed_video_hw") else None,
                "processed_width": (resource.get("processed_video_hw") or [{}])[0].get("width") if resource.get("processed_video_hw") else None,
                "processed_video_hw": resource.get("processed_video_hw"),
                "video_grid_thw": resource.get("video_grid_thw"),
                "image_grid_thw": resource.get("image_grid_thw"),
                "golden_duration": item.get("golden_duration", sum(end - start for start, end in item.get("evidence", []) if isinstance(start, (int, float)) and isinstance(end, (int, float)))),
                "evidence_duration": sum(end - start for start, end in item.get("evidence", []) if isinstance(start, (int, float)) and isinstance(end, (int, float))),
                "evidence_duration_bucket": item.get("evidence_duration_bucket") or evidence_duration_bucket(sum(end - start for start, end in item.get("evidence", []) if isinstance(start, (int, float)) and isinstance(end, (int, float)))),
                "full_video_duration_bucket": item.get("full_video_duration_bucket") or video_duration_bucket(float(item.get("video_duration") or 0.0)),
                "evidence_ratio": item.get("evidence_ratio"),
                "evidence_ratio_bucket": item.get("evidence_ratio_bucket"),
                "num_evidence_spans": item.get("num_evidence_spans", len(item.get("evidence", []))),
                "multi_span_policy": config_evidence.get("multi_span_policy", "separate"),
                # E18 rows use these optional fields.  Keeping them in the
                # common raw table makes the complementarity plot automatic
                # once that dual-view experiment is run.
                "temporal_correct": item.get("temporal_correct"),
                "spatial_correct": item.get("spatial_correct"),
                "fusion_correct": item.get("fusion_correct"),
            })
    out = root / "_summary"
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "summary.csv", summary_rows)
    _write_csv(out / "summary_by_type.csv", category_rows)
    _write_csv(out / "raw_plot_data.csv", raw_rows)
    _write_formal_tables(out, raw_rows)
    _write_paired_bootstrap(
        out,
        root,
        reference=reference,
        candidates=candidates,
        bootstrap_seed=bootstrap_seed,
        bootstrap_resamples=bootstrap_resamples,
        include_cross_pipeline=include_cross_pipeline,
    )
    _make_plots(out, summary_rows, raw_rows)
    _write_interim_report(out, summary_rows)
    return out


def _write_formal_tables(out: Path, raw: list[dict[str, Any]]) -> None:
    """Write stable tables consumed by the formal protocol and paper plots."""

    _write_csv(
        out / "table_evidence.csv",
        _aggregate_rows(
            raw,
            ["experiment", "pipeline_family", "evidence_duration_bucket", "evidence_ratio_bucket"],
        ),
    )
    _write_csv(
        out / "table_fps.csv",
        _aggregate_rows(raw, ["experiment", "pipeline_family", "effective_fps", "question_category"]),
    )
    resolution_rows = []
    for row in raw:
        item = dict(row)
        item["tokens_per_frame_bin"] = _bin_tokens_per_frame(row.get("tokens_per_frame"))
        resolution_rows.append(item)
    _write_csv(
        out / "table_resolution.csv",
        _aggregate_rows(
            resolution_rows,
            ["experiment", "pipeline_family", "configured_resolution", "processed_height", "processed_width", "tokens_per_frame_bin", "question_category"],
        ),
    )
    iso_rows = [row for row in raw if row.get("mode") in {"iso_budget", "e9"} or row.get("iso_budget")]
    _write_csv(
        out / "table_iso_token.csv",
        _aggregate_rows(
            iso_rows,
            ["experiment", "pipeline_family", "iso_budget", "iso_policy", "effective_fps", "configured_resolution", "question_category"],
        ),
    )
    _write_csv(
        out / "table_dense_hr.csv",
        _aggregate_rows(
            [row for row in raw if row.get("mode") in {"dense_hr", "e10"} or row.get("num_hr_images") is not None],
            ["experiment", "pipeline_family", "num_hr_images", "question_category"],
        ),
    )
    _write_csv(
        out / "table_audio.csv",
        _aggregate_rows(
            [row for row in raw if row.get("use_audio") is not None or row.get("subtitle")],
            ["experiment", "pipeline_family", "use_audio", "subtitle", "question_category"],
        ),
    )
    _write_csv(
        out / "table_multispan.csv",
        _aggregate_rows(
            raw,
            ["experiment", "pipeline_family", "multi_span_policy", "num_evidence_spans", "question_category"],
        ),
    )
    complementarity = [
        row for row in raw
        if all(row.get(key) is not None for key in ("temporal_correct", "spatial_correct", "fusion_correct"))
    ]
    if complementarity:
        for row in complementarity:
            row["complementarity_group"] = (
                "both_correct" if row["temporal_correct"] and row["spatial_correct"] else
                "temporal_only" if row["temporal_correct"] else
                "spatial_only" if row["spatial_correct"] else "both_wrong"
            )
        _write_csv(
            out / "table_error_complementarity.csv",
            _aggregate_rows(complementarity, ["experiment", "pipeline_family", "complementarity_group"]),
        )
    else:
        _write_csv(out / "table_error_complementarity.csv", [])


def _resolve_run(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_dir() or path.is_file():
        return path
    candidate = root / path
    if candidate.exists():
        return candidate
    raise FileNotFoundError(value)


def _pipeline_from_run(path: Path) -> str:
    config = _run_config(path)
    return _run_pipeline(config)


def _write_paired_bootstrap(
    out: Path,
    root: Path,
    *,
    reference: str | Path | None,
    candidates: list[str | Path] | None,
    bootstrap_seed: int,
    bootstrap_resamples: int,
    include_cross_pipeline: bool,
) -> None:
    if reference is None:
        _write_csv(out / "paired_bootstrap.csv", [])
        _write_csv(out / "paired_bootstrap_by_slice.csv", [])
        return
    reference_path = _resolve_run(root, reference)
    reference_rows = read_prediction_rows(reference_path)
    reference_pipeline = _pipeline_from_run(reference_path)
    if candidates:
        candidate_paths = [_resolve_run(root, value) for value in candidates]
    else:
        candidate_paths = sorted({path.parent for path in root.rglob("predictions*.jsonl") if path.is_file()})
    rows: list[dict[str, Any]] = []
    slice_rows: list[dict[str, Any]] = []
    for candidate_path in candidate_paths:
        if candidate_path.resolve() == reference_path.resolve():
            continue
        candidate_pipeline = _pipeline_from_run(candidate_path)
        if not include_cross_pipeline and candidate_pipeline != reference_pipeline:
            continue
        candidate_rows = read_prediction_rows(candidate_path)
        table = paired_bootstrap_table(
            reference_rows,
            candidate_rows,
            seed=bootstrap_seed,
            resamples=bootstrap_resamples,
        )
        if table:
            overall = next(item for item in table if item["group_by"] == "all")
            rows.append({
                "reference": reference_path.name,
                "candidate": candidate_path.name,
                "reference_pipeline_family": reference_pipeline,
                "candidate_pipeline_family": candidate_pipeline,
                **overall,
            })
            for item in table:
                if item["group_by"] != "all":
                    slice_rows.append({
                        "reference": reference_path.name,
                        "candidate": candidate_path.name,
                        "reference_pipeline_family": reference_pipeline,
                        "candidate_pipeline_family": candidate_pipeline,
                        **item,
                    })
    _write_csv(out / "paired_bootstrap.csv", rows)
    _write_csv(out / "paired_bootstrap_by_slice.csv", slice_rows)


def _write_interim_report(out: Path, summary: list[dict[str, Any]]) -> None:
    lines = [
        "# Golden temporal interim report",
        "",
        "This report is generated from completed prediction JSONL files. "
        "Rows with missing correctness are retained as failures and are not "
        "silently converted to zero accuracy.",
        "",
        "| Experiment | Pipeline | Evaluated | Accuracy | Mean visual tokens | Mean latency (s) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        def fmt(value: Any) -> str:
            return "n/a" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value)
        lines.append(
            f"| {row.get('experiment')} | {row.get('pipeline_family')} | "
            f"{row.get('evaluated', 0)}/{row.get('samples', 0)} | {fmt(row.get('accuracy'))} | "
            f"{fmt(row.get('mean_visual_tokens'))} | {fmt(row.get('mean_latency_sec'))} |"
        )
    (out / "interim_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_plots(out: Path, summary: list[dict[str, Any]], raw: list[dict[str, Any]]) -> None:
    """Create the requested figures when matplotlib is available.

    Missing axes are intentionally left empty when the corresponding sweep
    has not been run yet; the raw CSV is always written.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        (out / "plots_unavailable.txt").write_text("matplotlib is unavailable; raw_plot_data.csv remains authoritative\n", encoding="utf-8")
        return
    try:
        import numpy as np
    except Exception:
        np = None
    figures = out / "figures"
    figures.mkdir(exist_ok=True)

    def grouped_points(x_key: str) -> list[tuple[float, float, int]]:
        groups: dict[float, list[float]] = {}
        for row in raw:
            x = row.get(x_key)
            if isinstance(x, (int, float)) and row.get("correct") is not None:
                groups.setdefault(float(x), []).append(float(bool(row.get("correct"))))
        return [(x, sum(values) / len(values), len(values)) for x, values in sorted(groups.items())]

    def save_figure(filename: str, aliases: tuple[str, ...] = ()) -> None:
        names = (filename,) + aliases
        for name in names:
            plt.savefig(figures / name, dpi=160)
            if name.endswith(".png"):
                plt.savefig(figures / f"{name[:-4]}.pdf")

    def scatter(x_key: str, filename: str, xlabel: str, aliases: tuple[str, ...] = ()) -> None:
        points = grouped_points(x_key)
        if not points:
            return
        xs = [x for x, _, _ in points]
        ys = [y for _, y, _ in points]
        plt.figure(figsize=(6, 4))
        plt.plot(xs, ys, marker="o")
        plt.xlabel(xlabel)
        plt.ylabel("correct (per sample)")
        plt.tight_layout()
        save_figure(filename, aliases)
        plt.close()

    scatter("effective_fps", "figure1_accuracy_vs_fps.png", "effective FPS", ("fig_fps_by_category.png",))
    # Use the processor-observed token density rather than a nominal
    # low/medium/high name.  The CSV retains the exact H/W and grid values.
    resolution_groups: dict[str, list[float]] = {}
    for row in raw:
        tokens = row.get("tokens_per_frame")
        if isinstance(tokens, (int, float)) and row.get("correct") is not None:
            resolution_groups.setdefault(_bin_tokens_per_frame(tokens), []).append(float(bool(row["correct"])))
    if resolution_groups:
        plt.figure(figsize=(6, 4))
        keys = sorted(resolution_groups, key=lambda value: float(value) if value != "unknown" else -1.0)
        plt.bar(keys, [sum(resolution_groups[key]) / len(resolution_groups[key]) for key in keys])
        plt.ylabel("accuracy")
        plt.xlabel("actual visual tokens / sampled frame")
        plt.tight_layout()
        save_figure("figure2_accuracy_vs_resolution.png", ("fig_resolution_by_category.png",))
        plt.close()
    scatter("visual_tokens", "figure4_accuracy_vs_visual_tokens.png", "visual tokens (exact when available)", ("fig_accuracy_vs_tokens.png",))
    halo_groups: dict[float, list[float]] = {}
    for row in raw:
        halo = row.get("halo_before")
        if isinstance(halo, (int, float)) and row.get("mode") in {"golden_halo", "halo", "e4"} and row.get("correct") is not None:
            halo_groups.setdefault(float(halo), []).append(float(bool(row.get("correct"))))
    if halo_groups:
        xs = sorted(halo_groups)
        plt.figure(figsize=(6, 4))
        plt.plot(xs, [sum(halo_groups[x]) / len(halo_groups[x]) for x in xs], marker="o")
        plt.xlabel("halo before/after (s)")
        plt.ylabel("accuracy")
        plt.tight_layout()
        save_figure("figure5_accuracy_vs_halo.png", ("fig_evidence_halo.png",))
        plt.close()
    scatter("num_hr_images", "figure7_accuracy_vs_hr_keyframes.png", "high-resolution keyframes", ("fig_dense_hr.png",))

    # E9: actual accuracy heatmap.  The vertical axis is the processor's
    # observed visual tokens/frame, not a nominal resolution name.
    iso = [row for row in raw if row.get("mode") in {"iso_budget", "e9"} and row.get("effective_fps") is not None and row.get("configured_resolution")]
    iso = [row for row in iso if isinstance(row.get("tokens_per_frame"), (int, float))]
    if iso:
        fps_values = sorted({float(row["effective_fps"]) for row in iso})
        token_values = sorted({float(row["tokens_per_frame"]) for row in iso})
        matrix = np.full((len(token_values), len(fps_values)), np.nan) if np is not None else None
        if matrix is not None:
            for yi, token_value in enumerate(token_values):
                for xi, fps in enumerate(fps_values):
                    values = [
                        float(bool(row["correct"])) for row in iso
                        if float(row["tokens_per_frame"]) == token_value
                        and float(row["effective_fps"]) == fps
                        and row.get("correct") is not None
                    ]
                    if values:
                        matrix[yi, xi] = sum(values) / len(values)
            plt.figure(figsize=(7, 4))
            plt.imshow(matrix, vmin=0, vmax=1, aspect="auto", cmap="viridis")
            plt.xticks(range(len(fps_values)), [str(x) for x in fps_values])
            plt.yticks(range(len(token_values)), [f"{x:.0f}" for x in token_values])
            plt.xlabel("effective FPS")
            plt.ylabel("actual visual tokens / frame")
            plt.colorbar(label="accuracy")
            for yi in range(matrix.shape[0]):
                for xi in range(matrix.shape[1]):
                    if not math.isnan(float(matrix[yi, xi])):
                        plt.text(xi, yi, f"{matrix[yi, xi]:.2f}", ha="center", va="center", color="white")
            plt.tight_layout()
            save_figure("figure3_iso_budget_heatmap.png", ("fig_iso_token_heatmap.png",))
            plt.close()

    # E13: audio/subtitle factorial, when the corresponding runs exist.
    factorial: dict[tuple[str, str], list[float]] = {}
    for row in raw:
        if row.get("correct") is None or row.get("use_audio") is None:
            continue
        key = ("audio" if bool(row.get("use_audio")) else "no-audio", "subtitle" if bool(row.get("subtitle")) else "no-subtitle")
        factorial.setdefault(key, []).append(float(bool(row.get("correct"))))
    if factorial:
        labels = [f"{audio}\n{subtitle}" for audio, subtitle in sorted(factorial)]
        plt.figure(figsize=(7, 4))
        plt.bar(range(len(labels)), [sum(factorial[key]) / len(factorial[key]) for key in sorted(factorial)])
        plt.xticks(range(len(labels)), labels)
        plt.ylabel("accuracy")
        plt.tight_layout()
        save_figure("figure6_audio_subtitle_factorial.png", ("fig_audio_factorial.png",))
        plt.close()

    complementarity = [row for row in raw if all(row.get(key) is not None for key in ("temporal_correct", "spatial_correct", "fusion_correct"))]
    if complementarity:
        counts = {
            "both_correct": sum(bool(row["temporal_correct"]) and bool(row["spatial_correct"]) for row in complementarity),
            "temporal_only": sum(bool(row["temporal_correct"]) and not bool(row["spatial_correct"]) for row in complementarity),
            "spatial_only": sum(not bool(row["temporal_correct"]) and bool(row["spatial_correct"]) for row in complementarity),
            "both_wrong": sum(not bool(row["temporal_correct"]) and not bool(row["spatial_correct"]) for row in complementarity),
            "fusion_correct": sum(bool(row["fusion_correct"]) for row in complementarity),
        }
        plt.figure(figsize=(7, 4))
        plt.bar(list(counts), list(counts.values()))
        plt.xticks(range(len(counts)), list(counts), rotation=25, ha="right")
        plt.ylabel("samples")
        plt.tight_layout()
        save_figure("figure8_temporal_spatial_complementarity.png", ("fig_error_complementarity.png",))
        plt.close()
    else:
        # Keep the requested artifact present while making the missing E18
        # run explicit instead of fabricating a zero-valued result.
        plt.figure(figsize=(7, 4))
        plt.text(0.5, 0.5, "E18 dual-view results not available", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        save_figure("figure8_temporal_spatial_complementarity.png", ("fig_error_complementarity.png",))
        plt.close()

    # Always persist plot-ready tables, including empty tables for sweeps that
    # have not been launched yet.  This makes missing experiments explicit.
    _write_csv(out / "plot_accuracy_vs_fps.csv", [row for row in raw if row.get("effective_fps") is not None])
    _write_csv(out / "plot_accuracy_vs_resolution.csv", [row for row in raw if row.get("resolution") is not None])
    _write_csv(out / "plot_audio_factorial.csv", [row for row in raw if row.get("use_audio") is not None])
    _write_csv(out / "plot_keyframes.csv", [row for row in raw if row.get("num_hr_images") is not None])
    _write_csv(out / "plot_iso_budget.csv", iso)
    _write_csv(out / "plot_halo.csv", [row for row in raw if row.get("halo_before") is not None])
    _write_csv(out / "plot_error_complementarity.csv", complementarity)


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--reference", default=None, help="reference run path or name for paired bootstrap")
    parser.add_argument("--candidate", action="append", default=[], help="candidate run path/name; repeatable")
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--include-cross-pipeline", action="store_true")
    args = parser.parse_args(argv)
    print(f"summary output: {summarize(args.root, reference=args.reference, candidates=args.candidate or None, bootstrap_seed=args.bootstrap_seed, bootstrap_resamples=args.bootstrap_resamples, include_cross_pipeline=args.include_cross_pipeline)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
