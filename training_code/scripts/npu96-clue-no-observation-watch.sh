#!/usr/bin/env bash
# Dynamic in-place NPU utilization table for npu96 worker-10 + worker-11.
#
# Usage: npu96-openqa-watch.sh [interval_seconds] [count]
#   interval_seconds: refresh interval, default 2 (data fetch takes ~1-3s)
#   count:            number of refreshes, default infinite (Ctrl-C to stop)
set -uo pipefail

interval="${1:-2}"
count="${2:-0}"

run_root="/opt/huawei/dataset/hyl_ulan/ylhu/omni-clue-opsd/training_runs/worldsense_clue_full_no_observation_npu96_w10w11_20260930"
snapshot_script="/opt/huawei/dataset/hyl_ulan/ylhu/omni-opsd-runtime/worldsense_sft_lora_npu24_20260928/scripts/npu-snapshot.sh"
npu96_job="ma-job-7b525feb-b862-4352-8403-9c3a018c05e8"
npu96_host="dev-modelarts-cnnorth9.huaweicloud.com"
npu96_key="/home/ma-user/.ssh/ulan_31445_npu96.pem"
tmpdir="$(mktemp -d /tmp/npu96-openqa-watch.XXXXXX)"
trap 'rm -rf "${tmpdir}"; printf "\033[?25h\n"' EXIT

fetch() {
  local worker="$1" out="$2"
  ssh -F /dev/null -p 31445 -i "${npu96_key}" -o BatchMode=yes \
    "ma-user@${npu96_host}" \
    "ssh -F /dev/null -p 2222 -o BatchMode=yes ma-user@${npu96_job}-worker-${worker}.${npu96_job} 'bash ${snapshot_script}'" \
    > "${out}" 2>/dev/null &
}

render() {
  local ts="$1"
  printf '\033[H\033[J'
  printf 'npu96 WorldSense full CLUE without observation (worker-10 + worker-11)   %s   刷新 %ss   Ctrl-C 退出\n' "${ts}" "${interval}"
  printf '%-6s | %-32s | %-32s\n' "NPU" "worker-10 (AI% / HBM% / W / C)" "worker-11 (AI% / HBM% / W / C)"
  printf -- '-------+----------------------------------+----------------------------------\n'
  local i w10 w11
  for i in 0 1 2 3 4 5 6 7; do
    w10="$(awk -v k="npu${i}" '$1==k {printf "%3s / %3s / %4s / %2s", $4, $5, $2, $3}' "${tmpdir}/w10")"
    w11="$(awk -v k="npu${i}" '$1==k {printf "%3s / %3s / %4s / %2s", $4, $5, $2, $3}' "${tmpdir}/w11")"
    printf 'npu%-3s | %-32s | %-32s\n' "${i}" "${w10:-n/a}" "${w11:-n/a}"
  done
  printf -- '-------+----------------------------------+----------------------------------\n'
  local prog
  prog="$(grep -a -o "{'loss'[^}]*}" "${run_root}/logs/formal_worker-10.log" 2>/dev/null | tail -1 \
    | sed -E "s/'loss': '([^']*)'.*'global_step\/max_steps': '([^']*)'.*'elapsed_time': '([^']*)'.*/loss \1  step \2  elapsed \3/")"
  printf '训练进度(rank0): %s\n' "${prog:-等待首个日志行…}"
}

printf '\033[?25l'
n=0
while true; do
  fetch 10 "${tmpdir}/w10"
  fetch 11 "${tmpdir}/w11"
  wait
  render "$(date +%H:%M:%S)"
  n=$((n + 1))
  [[ "${count}" -gt 0 && "${n}" -ge "${count}" ]] && break
  sleep "${interval}"
done
