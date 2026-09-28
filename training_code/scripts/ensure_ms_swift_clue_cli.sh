#!/usr/bin/env bash
set -euo pipefail

# The patched GKDConfig already understands CLUE's LoRA-shadow EMA field, but
# ms-swift's top-level `swift rlhf` parser is built from RLHFArguments.  Keep
# the parser patch explicit and idempotent so a fresh checkout cannot silently
# drop --clue_ema_alpha before a CLUE-OPSD run.

ms_swift_root="${1:?usage: ensure_ms_swift_clue_cli.sh /path/to/ms-swift}"
target="${ms_swift_root}/swift/arguments/rlhf_args.py"
[[ -f "${target}" ]] || { echo "missing ms-swift RLHF arguments: ${target}" >&2; exit 2; }

lock="${ms_swift_root}/.omniopsd_clue_cli.lock"
while ! mkdir "${lock}" 2>/dev/null; do
  sleep 1
done
trap 'rmdir "${lock}"' EXIT

if ! grep -q '^    clue_ema_alpha: Optional\[float\] = None$' "${target}"; then
  backup="${target}.pre_omniopsd_clue_cli.bak"
  if [[ ! -e "${backup}" ]]; then
    cp -p "${target}" "${backup}"
  fi
  python - "${target}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
needle = "    # GKD\n    sft_alpha: float = 0\n    lmbda: float = 0.5\n"
replacement = (
    "    # GKD\n"
    "    sft_alpha: float = 0\n"
    "    # CLUE-OPSD LoRA-shadow EMA teacher.\n"
    "    clue_ema_alpha: Optional[float] = None\n"
    "    lmbda: float = 0.5\n"
)
if text.count(needle) != 1:
    raise SystemExit(f"expected one RLHF GKD field block, found {text.count(needle)}")
path.write_text(text.replace(needle, replacement), encoding="utf-8")
PY
fi

grep -q '^    clue_ema_alpha: Optional\[float\] = None$' "${target}"
printf '%s\n' "CLUE_CLI_PATCHED target=${target}" 
