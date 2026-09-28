#!/usr/bin/env python3
"""Translate the review-page QA / observations / captions into Chinese.

Calls qwen-flash on the DashScope compatible endpoint, caches every translation
by content hash in translations.json (so re-runs and the web server are free)
and writes a merged translation file the review server can serve.

Usage:
    python scripts/translate_worldsense_review.py --limit 20 --workers 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from worldsense_review_server import build_items  # noqa: E402

CACHE_PATH = PROJECT_ROOT / "output" / "worldsense_api_review" / "translations.json"
OUT_PATH = PROJECT_ROOT / "output" / "worldsense_api_review" / "translations_merged.json"
CHUNK_CHARS = 3500

SYSTEM_PROMPT = (
    "你是专业的视频标注翻译。把用户给的英文内容翻译成简体中文，要求："
    "1) 保留所有时间戳（如 00:12、1:05）、数字、单位不变；"
    "2) 人名、地名、品牌名可音译或保留英文；"
    "3) 选项列表保留 A. B. C. D. 前缀；"
    "4) 逐段对应，不要合并或省略内容；"
    "5) 只输出译文，不要任何解释、前言或后记。"
)


def load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    env_path = PROJECT_ROOT / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            line = re.sub(r"^export\s+", "", line)
            if "=" in line:
                key, value = line.split("=", 1)
                env[key.strip()] = value.strip().strip('"').strip("'")
    env.update({k: v for k, v in os.environ.items() if k.startswith("DASHSCOPE")})
    return env


def content_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def chunks(text: str, size: int = CHUNK_CHARS) -> list[str]:
    if len(text) <= size:
        return [text]
    parts: list[str] = []
    current = ""
    for line in text.splitlines(keepends=True):
        if len(current) + len(line) > size and current:
            parts.append(current)
            current = ""
        current += line
        while len(current) > size * 2:
            parts.append(current[:size])
            current = current[size:]
    if current:
        parts.append(current)
    return parts


class Translator:
    def __init__(self, base_url: str, api_key: str, model: str, workers: int):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=180)
        self.model = model
        self.workers = workers
        self.cache: dict[str, str] = {}
        if CACHE_PATH.is_file():
            try:
                self.cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self.cache = {}
        self.calls = 0

    def _one(self, text: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0.2,
            max_tokens=8192,
        )
        self.calls += 1
        return (response.choices[0].message.content or "").strip()

    def translate(self, text: str) -> str:
        if not text or not text.strip():
            return ""
        key = content_key(text)
        if key in self.cache:
            return self.cache[key]
        pieces = chunks(text)
        if len(pieces) == 1:
            result = self._one(text)
        else:
            results = [self._one(piece) for piece in pieces]
            result = "\n".join(results)
        self.cache[key] = result
        return result

    def save(self) -> None:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(self.cache, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--model", default="qwen-flash")
    parser.add_argument("--force", action="store_true", help="ignore the cache")
    args = parser.parse_args()

    env = load_env()
    base_url = env.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    api_key = env.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        raise SystemExit("DASHSCOPE_API_KEY not found")

    items = build_items(limit=args.limit)
    translator = Translator(base_url, api_key, args.model, args.workers)
    if args.force:
        translator.cache = {}

    # Collect every text that needs a translation, deduplicated.
    jobs: dict[str, str] = {}
    for item in items:
        qa = item["qa"]
        if qa.get("question"):
            jobs[content_key(qa["question"])] = qa["question"]
        for candidate in qa.get("candidates") or []:
            jobs[content_key(candidate)] = candidate
        if item["annotation"].get("observation"):
            jobs[content_key(item["annotation"]["observation"])] = item["annotation"]["observation"]
        for run in item.get("runs") or []:
            text = (run.get("annotation") or {}).get("observation")
            if text:
                jobs[content_key(text)] = text
        if item["caption"].get("text"):
            jobs[content_key(item["caption"]["text"])] = item["caption"]["text"]
        if item["video"].get("video_caption"):
            jobs[content_key(item["video"]["video_caption"])] = item["video"]["video_caption"]
        if item["local"] and item["local"].get("observation"):
            jobs[content_key(item["local"]["observation"])] = item["local"]["observation"]

    todo = [(k, t) for k, t in jobs.items() if k not in translator.cache]
    print(f"items={len(items)} texts={len(jobs)} cached={len(jobs) - len(todo)} todo={len(todo)}")

    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(translator.translate, text): key for key, text in todo}
        done = 0
        for future in futures:
            future.result()
            done += 1
            if done % 10 == 0 or done == len(todo):
                print(f"  {done}/{len(todo)} done ({time.time() - started:.0f}s)")
    translator.save()

    merged: dict[str, Any] = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "items": {}}
    for item in items:
        qa = item["qa"]
        entry: dict[str, Any] = {
            "question": translator.cache.get(content_key(qa.get("question") or ""), ""),
            "candidates": [
                translator.cache.get(content_key(c), "") for c in (qa.get("candidates") or [])
            ],
            "observation": translator.cache.get(
                content_key(item["annotation"].get("observation") or ""), ""
            ),
            "caption": translator.cache.get(content_key(item["caption"].get("text") or ""), ""),
            "video_caption": translator.cache.get(
                content_key(item["video"].get("video_caption") or ""), ""
            ),
        }
        if item["local"]:
            entry["local_observation"] = translator.cache.get(
                content_key(item["local"].get("observation") or ""), ""
            )
        merged["items"][item["question_id"]] = entry
    OUT_PATH.write_text(json.dumps(merged, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {OUT_PATH} | api calls this run: {translator.calls} | cache size: {len(translator.cache)}")


if __name__ == "__main__":
    main()
