#!/usr/bin/env bash
set -euo pipefail

project_root="${OMNI_OPSD_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
swift_root="${1:?usage: apply_ms_swift_patches.sh /path/to/ms-swift}"
patch_files=(
  "${project_root}/patches/ms-swift-infer-protocol.patch"
  "${project_root}/patches/ms-swift-qwen25-bounded-av.patch"
  "${project_root}/patches/ms-swift-qwen25-bounded-audio.patch"
  "${project_root}/patches/ms-swift-gkd-topk-tail.patch"
)

[[ -d "${swift_root}/.git" ]] || { echo "Not an ms-swift checkout: ${swift_root}" >&2; exit 2; }

apply_adapted_source() {
  local target="$1"
  local kind="$2"
  python - "${target}" "${kind}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
kind = sys.argv[2]
text = path.read_text(encoding='utf-8')

def replace_once(old, new):
    count = text.count(old)
    if count != 1:
        raise SystemExit(f'{path}: expected one source block for {kind}, found {count}')
    return text.replace(old, new, 1)

if kind == 'protocol':
    text = replace_once(
        '        videos (List[str]):\n'
        '            Optional, a list of video resources associated with the request.\n',
        '        videos (List[Any]):\n'
        '            Optional, a list of video resources or structured temporal video mappings\n'
        '            associated with the request.\n')
    text = replace_once(
        '    videos: List[str] = field(default_factory=list)',
        '    # OmniVideo rows may carry video_start/video_end/fps/max_frames/max_pixels\n'
        '    # mappings.  Qwen-Omni consumes these mappings during media processing.\n'
        '    videos: List[Any] = field(default_factory=list)')
elif kind == 'bounded_audio':
    text = replace_once(
        """        elif media_type == 'audio':
            if self.mode != 'vllm':
                inputs.audios[index] = load_audio(inputs.audios[index], sampling_rate)
""",
        """        elif media_type == 'audio':
            audio = inputs.audios[index]
            if isinstance(audio, dict):
                # Structured audio mappings carry audio_start/audio_end and must
                # be decoded with qwen-omni-utils' offset-aware loader.
                from qwen_omni_utils import process_audio_info
                message = {'role': 'user', 'content': [{'type': 'audio', **audio}]}
                bounded = process_audio_info([message], use_audio_in_video=False)
                if bounded is None or len(bounded) != 1:
                    raise ValueError(f'failed to extract mapped audio: {audio.get("audio")}')
                inputs.audios[index] = bounded[0]
            elif self.mode != 'vllm':
                inputs.audios[index] = load_audio(audio, sampling_rate)
""")
elif kind == 'bounded_av':
    text = replace_once(
        """        elif media_type == 'video':
            video = inputs.videos[index]
            video_inputs = {'video': video, **inputs.chat_template_kwargs}
            if isinstance(video, list):  # image list
                from qwen_omni_utils import vision_process
                video_inputs['sample_fps'] = vision_process.FPS
""",
        """        elif media_type == 'video':
            video = inputs.videos[index]
            # Structured OmniVideo mappings keep temporal and pixel sampling
            # fields beside the source path. Unwrap only the source for
            # qwen-omni-utils while preserving the mapping fields.
            video_source = video.get('video') if isinstance(video, dict) else video
            if isinstance(video, dict):
                video_inputs = dict(video)
                video_inputs['video'] = video_source
                video_inputs.update(inputs.chat_template_kwargs)
            else:
                video_inputs = {'video': video_source, **inputs.chat_template_kwargs}
            if isinstance(video_source, list):  # image list
                from qwen_omni_utils import vision_process
                video_inputs['sample_fps'] = vision_process.FPS
""")
    text = replace_once(
        """            if self.use_audio_in_video:
                if isinstance(video, list):  # image list
                    raise ValueError('image list as video input does not support use_audio_in_video')
                audio = load_audio(video, sampling_rate)
""",
        """            if self.use_audio_in_video:
                if isinstance(video_source, list):  # image list
                    raise ValueError('image list as video input does not support use_audio_in_video')
                if isinstance(video, dict):
                    # Keep the audio interval identical to the structured video
                    # mapping.  Loading the source path directly would expose
                    # the full audio to a clue-only teacher.
                    from qwen_omni_utils import process_audio_info
                    message = {'role': 'user', 'content': [{'type': 'video', **video}]}
                    bounded = process_audio_info([message], use_audio_in_video=True)
                    if bounded is None or len(bounded) != 1:
                        video_source = video.get('video')
                        raise ValueError(f'failed to extract mapped video audio: {video_source}')
                    audio = bounded[0]
                else:
                    audio = load_audio(video_source, sampling_rate)
""")
else:
    raise SystemExit(f'unknown adapted source kind: {kind}')

path.write_text(text, encoding='utf-8')
PY
}

apply_teacher_media_adaptation() {
  local data_target="${swift_root}/swift/rl_core/data.py"
  local helper_target="${swift_root}/swift/rlhf_trainers/gkd_helpers.py"
  python - "${data_target}" "${helper_target}" <<'PY'
from pathlib import Path
import sys

data_path, helper_path = map(Path, sys.argv[1:])
data = data_path.read_text(encoding='utf-8')
helper = helper_path.read_text(encoding='utf-8')

def once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise SystemExit(f'{label}: expected one source block, found {count}')
    return text.replace(old, new, 1)

data = once(
    data,
    '    teacher_images: Optional[List[Any]] = None  # OPSD: explicit teacher-side images; None reuses student images\n',
    '    teacher_images: Optional[List[Any]] = None  # OPSD: explicit teacher-side images; None reuses student images\n'
    '    teacher_videos: Optional[List[Any]] = None  # OPSD/CLUE: explicit teacher-side video mappings\n'
    '    teacher_audios: Optional[List[Any]] = None  # OPSD/CLUE: explicit teacher-side audio mappings\n',
    str(data_path))
data = once(
    data,
    '        When only ``teacher_images`` is provided, retain the student messages and vary\n'
    '        only the visual input. Teacher and student share ``response_token_ids``. Returns\n'
    '        ``True`` when either teacher-side input is explicitly provided (idempotent), and\n'
    '        ``False`` when both are unset (non-OPSD).\n',
    '        When teacher-side media are provided, retain the student messages and vary only\n'
    '        the requested media fields. Teacher and student share ``response_token_ids``.\n'
    '        Returns ``True`` when a teacher-side input is explicitly provided (idempotent),\n'
    '        and ``False`` when all teacher-side fields are unset (non-OPSD).\n',
    str(data_path))
data = once(
    data,
    '        if not self.teacher_prompt and self.teacher_images is None:\n',
    '        if (not self.teacher_prompt and self.teacher_images is None and\n'
    '                self.teacher_videos is None and self.teacher_audios is None):\n',
    str(data_path))
data = once(
    data,
    "            d['images'] = self.teacher_images\n",
    "            d['images'] = self.teacher_images\n"
    '        if self.teacher_videos is not None:\n'
    "            d['videos'] = self.teacher_videos\n"
    '        if self.teacher_audios is not None:\n'
    "            d['audios'] = self.teacher_audios\n",
    str(data_path))

helper = once(
    helper,
    '        if s.teacher_images is not None:\n'
    '            request_sample = copy.copy(s)\n'
    '            request_sample.images = s.teacher_images\n',
    '        if (s.teacher_images is not None or s.teacher_videos is not None or\n'
    '                s.teacher_audios is not None):\n'
    '            request_sample = copy.copy(s)\n'
    '            if s.teacher_images is not None:\n'
    '                request_sample.images = s.teacher_images\n'
    '            if s.teacher_videos is not None:\n'
    '                request_sample.videos = s.teacher_videos\n'
    '            if s.teacher_audios is not None:\n'
    '                request_sample.audios = s.teacher_audios\n',
    str(helper_path))

data_path.write_text(data, encoding='utf-8')
helper_path.write_text(helper, encoding='utf-8')
PY
}

apply_qwen_rope_float_adaptation() {
  local target="${swift_root}/swift/template/templates/qwen.py"
  python - "${target}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding='utf-8')
marker = '        # Transformers 5.16\'s Qwen2.5-Omni ``get_rope_index`` expects each\n'
generate_marker = '        # Rollout generation calls the HF model directly, bypassing\n'
old = (
    '        if attention_mask is None:\n'
    '            attention_mask = torch.ones_like(input_ids)\n'
    '        position_ids, _ = self._get_get_rope_index()(\n'
)
new = (
    '        if attention_mask is None:\n'
    '            attention_mask = torch.ones_like(input_ids)\n'
    '        # Transformers 5.16\'s Qwen2.5-Omni ``get_rope_index`` expects each\n'
    '        # ``second_per_grid`` entry to be a tensor and calls ``.cpu()`` on it.\n'
    '        # The current qwen-omni-utils path can return Python floats for\n'
    '        # structured video mappings, so normalize the gathered values here.\n'
    '        if video_second_per_grid is not None:\n'
    '            if isinstance(video_second_per_grid, (list, tuple)):\n'
    '                video_second_per_grid = [\n'
    '                    value if torch.is_tensor(value) else torch.as_tensor(value, dtype=torch.float32)\n'
    '                    for value in video_second_per_grid\n'
    '                ]\n'
    '            elif not torch.is_tensor(video_second_per_grid):\n'
    '                video_second_per_grid = torch.as_tensor(video_second_per_grid, dtype=torch.float32)\n'
    '        position_ids, _ = self._get_get_rope_index()(\n'
)
if marker not in text:
    if text.count(old) != 1:
        raise SystemExit(f'{path}: expected one Qwen rope source block, found {text.count(old)}')
    text = text.replace(old, new, 1)
old_generate = (
    "    def generate(self, model, *args, **kwargs):\n"
    "        if kwargs.get('video_grid_thw') is not None:\n"
    "            kwargs['use_audio_in_video'] = self.use_audio_in_video\n"
    "        return super().generate(model, *args, **kwargs)\n"
)
new_generate = (
    "    def generate(self, model, *args, **kwargs):\n"
    "        if kwargs.get('video_grid_thw') is not None:\n"
    "            kwargs['use_audio_in_video'] = self.use_audio_in_video\n"
    "        # Rollout generation calls the HF model directly, bypassing\n"
    "        # ``_get_position_ids`` above.  Normalize float temporal spacings here\n"
    "        # as Transformers 5.16's Qwen-Omni implementation calls ``.cpu()`` on\n"
    "        # every ``video_second_per_grid`` entry.\n"
    "        video_second_per_grid = kwargs.get('video_second_per_grid')\n"
    "        if video_second_per_grid is not None:\n"
    "            if isinstance(video_second_per_grid, (list, tuple)):\n"
    "                kwargs['video_second_per_grid'] = [\n"
    "                    value if torch.is_tensor(value) else torch.as_tensor(value, dtype=torch.float32)\n"
    "                    for value in video_second_per_grid\n"
    "                ]\n"
    "            elif not torch.is_tensor(video_second_per_grid):\n"
    "                kwargs['video_second_per_grid'] = torch.as_tensor(video_second_per_grid, dtype=torch.float32)\n"
    "        return super().generate(model, *args, **kwargs)\n"
)
if generate_marker not in text:
    if text.count(old_generate) != 1:
        raise SystemExit(f'{path}: expected one Qwen generate source block, found {text.count(old_generate)}')
    text = text.replace(old_generate, new_generate, 1)
path.write_text(text, encoding='utf-8')
PY
}

for patch_file in "${patch_files[@]}"; do
  [[ -f "${patch_file}" ]] || { echo "Patch is missing: ${patch_file}" >&2; exit 2; }

  # The bundle's patches were created against an older ms-swift revision.  The
  # current public checkout moved the surrounding code, so the equivalent
  # changes may already be present with different context.  Recognize those
  # adapted changes before trying the literal patch.
  case "$(basename "${patch_file}")" in
    ms-swift-infer-protocol.patch)
      target="${swift_root}/swift/infer_engine/protocol.py"
      if grep -q 'videos: List\[Any\]' "${target}"; then
        echo "Patch equivalent already present: ${patch_file}"
        continue
      fi
      ;;
    ms-swift-qwen25-bounded-av.patch)
      target="${swift_root}/swift/template/templates/qwen.py"
      if grep -q 'video_source = video.get' "${target}" && \
         grep -q 'failed to extract mapped video audio' "${target}"; then
        echo "Patch equivalent already present: ${patch_file}"
        continue
      fi
      ;;
    ms-swift-qwen25-bounded-audio.patch)
      target="${swift_root}/swift/template/templates/qwen.py"
      if grep -q 'failed to extract mapped audio' "${target}"; then
        echo "Patch equivalent already present: ${patch_file}"
        continue
      fi
      ;;
    ms-swift-gkd-topk-tail.patch)
      target="${swift_root}/swift/rlhf_trainers/gkd_loss.py"
      if grep -q 'student_tail_logprob' "${target}" && \
         grep -q 'topk=self.gkd_logits_topk' "${swift_root}/swift/rlhf_trainers/gkd_trainer.py"; then
        echo "Patch equivalent already present: ${patch_file}"
        continue
      fi
      ;;
  esac

  if git -C "${swift_root}" apply --reverse --check "${patch_file}" 2>/dev/null; then
    echo "Patch is already applied: ${patch_file}"
    continue
  fi
  if git -C "${swift_root}" apply --check "${patch_file}" 2>/dev/null; then
    git -C "${swift_root}" apply "${patch_file}"
    echo "Applied ${patch_file} to ${swift_root}"
  else
    case "$(basename "${patch_file}")" in
      ms-swift-infer-protocol.patch)
        apply_adapted_source "${swift_root}/swift/infer_engine/protocol.py" protocol
        ;;
      ms-swift-qwen25-bounded-av.patch)
        apply_adapted_source "${swift_root}/swift/template/templates/qwen.py" bounded_av
        ;;
      ms-swift-qwen25-bounded-audio.patch)
        apply_adapted_source "${swift_root}/swift/template/templates/qwen.py" bounded_audio
        ;;
      ms-swift-gkd-topk-tail.patch)
        echo "Cannot apply ${patch_file} to ${swift_root}; use the pinned ms-swift commit or apply the GKD patch manually" >&2
        exit 2
        ;;
      *)
        echo "Cannot apply ${patch_file} to ${swift_root}" >&2
        exit 2
        ;;
    esac
    echo "Applied adapted equivalent of ${patch_file} to ${swift_root}"
  fi
done

# These fields allow CLUE rows to replace the teacher's video/audio media.
# Dynamic standard OPSD additionally requires the LoRA-shadow EMA implementation
# already present in the checked-out ms-swift tree; this script verifies it
# below rather than silently launching a current-policy teacher.
for target in \
  "${swift_root}/swift/rl_core/data.py" \
  "${swift_root}/swift/rlhf_trainers/gkd_helpers.py"; do
  [[ -f "${target}" ]] || { echo "Missing teacher-media target: ${target}" >&2; exit 2; }
done
if ! grep -q 'teacher_videos:' "${swift_root}/swift/rl_core/data.py" || \
   ! grep -q 's.teacher_videos is not None' "${swift_root}/swift/rlhf_trainers/gkd_helpers.py"; then
  apply_teacher_media_adaptation
  echo "Applied adapted non-EMA teacher media support to ${swift_root}"
fi
grep -q 'teacher_videos:' "${swift_root}/swift/rl_core/data.py" || {
  echo "Missing adapted teacher_videos support in ${swift_root}" >&2
  exit 2
}
grep -q 's.teacher_videos is not None' "${swift_root}/swift/rlhf_trainers/gkd_helpers.py" || {
  echo "Missing adapted teacher media request support in ${swift_root}" >&2
  exit 2
}
echo "Verified teacher media support in ${swift_root}"

apply_qwen_rope_float_adaptation
grep -q 'normalize the gathered values here' "${swift_root}/swift/template/templates/qwen.py" || {
  echo "Missing Qwen rope float compatibility adaptation in ${swift_root}" >&2
  exit 2
}
grep -q 'Rollout generation calls the HF model directly' "${swift_root}/swift/template/templates/qwen.py" || {
  echo "Missing Qwen rollout rope float compatibility adaptation in ${swift_root}" >&2
  exit 2
}
echo "Verified Qwen rope float compatibility adaptation in ${swift_root}"

if grep -q 'OpsdShadowEMACallback' "${swift_root}/swift/rlhf_trainers/gkd_trainer.py" && \
   grep -q 'opsd_ema_alpha' "${swift_root}/swift/rlhf_trainers/gkd_trainer.py"; then
  echo "Verified OPSD LoRA-shadow EMA teacher implementation in ${swift_root}"
else
  echo "Missing OPSD LoRA-shadow EMA teacher implementation in ${swift_root}" >&2
  exit 2
fi
