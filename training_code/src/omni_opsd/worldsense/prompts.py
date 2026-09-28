"""Prompt construction for the WorldSense evidence-localization agent.

The metadata block deliberately carries every dataset field the agent can use
to localize evidence (duration, domain, sub_category, audio_class, task_domain,
task_type, synopsis).  The action contract is a strict single-JSON protocol and
every action must carry an ``observation`` explaining the model's reasoning;
``submit`` additionally requires one observation per interval.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .config import AgentConfig
from .schema import QuestionRecord

SYSTEM_HEADER = """You are an evidence-localization annotator for a video question-answering dataset.
Your ONLY job is to locate the minimal time intervals in the video that contain the
evidence needed to answer the question. You do NOT answer the question and you must
never state or imply which option is correct.

How you explore the video:
- You FIRST receive a full-video survey: the entire video sampled at a coarse frame
  rate with synchronized audio. Study it to understand what happens and build a rough
  timeline (which events happen at roughly which times). This survey is the
  authoritative source of truth.
- The metadata may also include an auto-generated synopsis. Treat it only as a weak
  hint: it can be incomplete, out of order, or wrong. Always verify anything you read
  in the synopsis by actually watching the video.
- After the survey you use the `inspect` action to re-watch specific time ranges at a
  higher frame rate / resolution / with audio in order to confirm the exact evidence.
- The survey is provided automatically and does NOT count against your inspect budget."""

ACTION_CONTRACT = """Reply with EXACTLY ONE JSON object per turn and no other text.
Allowed actions:

1) Inspect a view:
{{"observation": "<what you learned so far / why this view>",
  "action": "inspect",
  "start": <seconds>, "end": <seconds>,
  "fps": <0.5-8.0>,
  "max_pixels": <3136-313600>,
  "modality": "av" | "video" | "audio",
  "focus": "<the specific thing to check in this view>"}}

2) Ask for video metadata (cheap, no media read):
{{"observation": "<why>", "action": "get_media_info"}}

3) Submit the final evidence intervals:
{{"observation": "<summary of the evidence you found>",
  "action": "submit",
  "intervals": [{{"start": <seconds>, "end": <seconds>,
                  "observation": "<why this interval is the evidence>"}}],
  "confidence": <0.0-1.0>}}

Hard rules:
- Every reply MUST include "observation". On submit, EVERY interval must include
  its own "observation" explaining the evidence it contains.
- Never output the answer text, the correct option letter, or a restatement of an
  option as a fact. Describe only what/where the evidence is.
- Intervals must satisfy: 0 <= start < end <= video_duration_seconds, each interval
  at least {min_interval:.1f}s, at most {max_intervals} intervals, and total length
  at most min({max_total:.0f}s, {max_ratio:.0%} of the video duration).
- Prefer the SHORTEST interval(s) that still contain the evidence; do not pad with
  unrelated context.
- Budget: at most {max_inspect} inspect calls. Submit as soon as you are confident."""

GUIDANCE = """Working method (follow this order):
1. Study the full-video survey first and describe, in your first "observation", the
   rough timeline you inferred: what happens early / in the middle / late, and which
   parts are likely relevant to the question. The synopsis may suggest candidates,
   but the survey decides.
2. Then inspect SHORT, SPECIFIC windows (typically 5-30 seconds) around the candidate
   moments at higher fps / resolution to confirm the exact boundaries.
   Do NOT inspect the whole video again: the survey already covers it, and re-watching
   it (in any quality) wastes your inspect budget. A full-range inspect is only useful
   if you raise the spatial budget well above the survey (a sharper full pass).
3. Submit only after you have actually seen the evidence in an inspect view.

Evidence-type guidance (choose your view parameters accordingly):
- task_type Temporal Localization / Event Sorting: probe several short candidate
  moments with modality "av" and fps 2-4, then submit the exact windows.
- task_type Text and Diagram Understanding: raise max_pixels (>=156800) and use
  fps 2-4; the evidence is often small on-screen text or a diagram.
- task_type Object Counting / Action Counting: cover the whole counting process
  with modality "av"; counts can change across the video.
- task_type Spatial Relation / Attribute Recognition: modality "video" or "av",
  fps 1-2, max_pixels around 156800; one clear view usually suffices.
- task_type Audio Recognition / Audio Source Localization, or audio_class contains
  "Music": inspect modality "audio" (or "av") for instruments, singing, sound
  events and lyrics; the decisive evidence may be inaudible in frames alone.
- audio_class contains "Speech" and the question is about names, numbers, places,
  statements or dialogue: inspect modality "audio" or "av" and listen for the
  spoken content at candidate moments.
- audio_class contains "Event" and the question is about what happened/how:
  inspect "av" around candidate actions; sounds often mark the exact moment.
- If the survey was sampled at a low frame rate (long video), your first time
  estimates are approximate: widen or shift the window before submitting."""


def _metadata_block(record: QuestionRecord, config: AgentConfig) -> str:
    metadata: dict[str, Any] = record.prompt_metadata()
    if not config.include_synopsis:
        metadata.pop("video_synopsis", None)
    return json.dumps(metadata, ensure_ascii=False, indent=2)


def build_system_prompt(record: QuestionRecord, config: AgentConfig) -> str:
    contract = ACTION_CONTRACT.format(
        min_interval=config.min_interval_seconds,
        max_intervals=config.max_intervals,
        max_total=config.max_total_seconds,
        max_ratio=config.max_total_ratio,
        max_inspect=config.max_inspect,
    )
    parts = [
        SYSTEM_HEADER,
        "Video and question metadata:\n" + _metadata_block(record, config),
    ]
    parts.append(contract)
    parts.append(GUIDANCE)
    return "\n\n".join(parts)


def build_initial_user_prompt(record: QuestionRecord, config: AgentConfig) -> str:
    if config.force_full_scan:
        return (
            "Options are listed in the metadata above. The full video has just been "
            "provided as a coarse survey (with audio).\n"
            "First describe the rough timeline you observed, then inspect the most "
            "promising range(s) to confirm the exact evidence, and finally submit the "
            "evidence intervals.\n"
            "Remember: you output evidence locations only, never the answer."
        )
    return (
        "Options are listed in the metadata above. Start by inspecting the most "
        "promising part of the video, then submit the evidence intervals.\n"
        "Remember: you output evidence locations only, never the answer."
    )


CAPTION_PROMPT = """You are a rigorous audio-visual description expert. Watch the ENTIRE video and produce a detailed, chronological, evidence-grounded description that reconstructs what is actually visible, audible and readable in it. The description will later be used to locate the evidence for a specific question, so it must be complete and precise rather than short.

Core principles
1. Cover the video from beginning to end: opening, main process, scene changes, important actions, speech, on-screen text, sounds and ending. Removing repetition applies only to attributes that do not change (clothing, room layout, a continuing music bed); never skip new actions, text, sounds or state changes.
2. Every statement must be grounded in what is visible, audible or readable. Never add outside knowledge, never guess or fill in gaps. If something cannot be confirmed, say so briefly or omit it.
3. Preserve the exact wording of speech, subtitles and on-screen text, especially names, places, numbers, brands, labels and key statements. Do not translate, rewrite or correct them.
4. Describe actions with their parts: who acts, the state before, the action itself, the object or point of contact, direction, manner, intermediate stages and the resulting state. A named action alone ("she opens the bottle") is a label, not a description.
5. Track people and objects continuously and keep their identity consistent. If similar entities cannot be distinguished, use positional wording such as "the device on the left" instead of forcing a name.
6. Do not invent identity, age, occupation, emotion, intention or relationships. Describe visible expressions and movements, not inferred mental states.

What to cover whenever present
- main subjects (people, animals, objects): appearance, clothing, colours, counts, spatial positions;
- actions, interactions, operating steps and their order;
- setting, foreground/middle ground/background, lighting, left-right relations, spatial changes;
- camera viewpoint, shot size, focus, pan/tilt/zoom, following shots, transitions;
- titles, subtitles, labels, interface text, numbers, formulas, tables, charts;
- speech: who speaks, discernible original wording, obvious tone;
- music, ambient sound, sound effects, and when they start, stop or change;
- entrances, exits, contact, state changes, temporal continuity and scene jumps.

Timestamp rules
- Start every segment with the format [hh:mm:ss:xxx-hh:mm:ss:xxx] and describe what happens inside it.
- Adjacent segments must not overlap. Use only timestamps you can support: shot cuts, subtitle appearance, speech onset/offset, clearly locatable visual events. Do not fabricate millisecond precision.
- Match precision to the evidence: chapters may span minutes, shots and ordinary actions about a second, dialogue turns a few tenths of a second.

Question focus (very important)
- The question below is what this description will be used for.
- Where a moment is relevant to it, describe that moment in extra detail: the exact action, the exact on-screen text, the exact spoken words, the exact sound, and where it sits between neighbouring events, so it can be pinpointed later.

Question: {question}
Options:
{options}
Task type: {task_type} | audio_class: {audio_class}

Output
- One short overview paragraph (video type, main subjects, setting, overall sound), then the chronological timestamped paragraphs.
- Integrate picture, speech, on-screen text and sound inside the same segment instead of splitting by modality.
- Do not use tables, bullet lists, numbered lists, JSON, analytical headings, or any analysis/observation/tool-call notes."""


def build_caption_prompt(record: QuestionRecord) -> str:
    options = "\n".join(record.candidates) if record.candidates else "(no options provided)"
    return CAPTION_PROMPT.format(
        question=record.question,
        options=options,
        task_type=record.task_type or "unknown",
        audio_class=", ".join(record.audio_class) or "unknown",
    )


TIMESTAMP_RE = re.compile(r"\[\d{1,2}:\d{2}(?::\d{2})?(?::\d{3})?\s*[-–]\s*\d{1,2}:\d{2}(?::\d{2})?(?::\d{3})?\]")
CAPTION_RETRY_NOTE = (
    "\n\nIMPORTANT: the previous attempt was rejected because it did not look like a "
    "complete timestamped description. Start every paragraph with [hh:mm:ss:xxx-hh:mm:ss:xxx] "
    "and cover the whole video from beginning to end."
)


def validate_caption(text: str, min_chars: int = 200) -> tuple[bool, str]:
    """A usable caption must be long enough and contain timestamped segments."""

    if not text or len(text.strip()) < int(min_chars):
        return False, f"too_short({len(text or '')}<{min_chars})"
    matches = TIMESTAMP_RE.findall(text)
    if len(matches) < 2:
        return False, f"too_few_timestamps({len(matches)})"
    return True, "ok"


def build_caption_planning_message(record: QuestionRecord, caption: str) -> str:
    """Feed the timestamped caption back and ask for the first inspect plan."""

    return (
        "A detailed timestamped description of the full video is given below. It was "
        "produced by watching the whole video; timestamps are approximate but grounded "
        "in real events.\n\n"
        "===== VIDEO DESCRIPTION (timestamped) =====\n"
        f"{caption.strip()}\n"
        "===== END OF DESCRIPTION =====\n\n"
        "Use this description to choose where to look. Do NOT re-watch the whole video: "
        "pick the specific time range(s) that most likely contain the evidence for the "
        "question and inspect them at a higher frame rate / resolution (or with audio) "
        "to confirm.\n"
        "Reply with exactly one JSON object: either an `inspect` action for the most "
        "promising range (5-30 seconds, your chosen fps/max_pixels/modality), or a "
        "`submit` if the description already pins down the evidence precisely.\n"
        "Remember: never output the answer."
    )
