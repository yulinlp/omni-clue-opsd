#!/usr/bin/env bash
set -euo pipefail

# Apply the qwen_omni_utils runtime patches used by the full AV training arms.
# The target is a standalone package directory, not necessarily a Git checkout.
project_root="${OMNI_OPSD_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
qwen_root="${1:?usage: apply_qwen_omni_utils_patches.sh /path/to/qwen_omni_utils_root}"
target="${qwen_root}/qwen_omni_utils/v2_5/vision_process.py"

[[ -f "${target}" ]] || { echo "Missing qwen_omni_utils target: ${target}" >&2; exit 2; }
for patch_file in \
  "${project_root}/patches/qwen_omni_utils_training.patch" \
  "${project_root}/patches/qwen_omni_utils_edge_decode.patch"; do
  [[ -f "${patch_file}" ]] || { echo "Patch is missing: ${patch_file}" >&2; exit 2; }
  if patch --dry-run --reverse --silent -p1 -d "${qwen_root}" < "${patch_file}"; then
    echo "Patch is already applied: ${patch_file}"
    continue
  fi
  patch --dry-run --forward --silent -p1 -d "${qwen_root}" < "${patch_file}"
  patch --forward --silent -p1 -d "${qwen_root}" < "${patch_file}"
  echo "Applied ${patch_file} to ${qwen_root}"
done

python -m py_compile "${target}"
sha256sum "${target}"
