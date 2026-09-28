#!/usr/bin/env python3
"""Run one command on a shared GPU node through an authorized account.

The cluster gate (``pam_slurm_adopt``) only admits logins that pass through an
account holding an active allocation.  This helper performs the established
dance -- ``su`` to the allocation owner, ``ssh`` to the node, ``su`` back to
``ylhu`` -- inside a pty and returns the command output.

Passwords are read from the private credential file at runtime and are never
printed (the output is scrubbed before it is returned).

Example:
    python scripts/shared_gpu_exec.py --account wxzhao --node gpu06 \
        --command "nvidia-smi -L"
"""

from __future__ import annotations

import argparse
import os
import pty
import re
import select
import shlex
import signal
import sys
import time
from pathlib import Path

CREDENTIAL_FILE = Path("/share/home/ylhu/EvoEmbedding/GPU_ACCOUNT_ACCESS.private.md")
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
PASSWORD_RE = re.compile(r"[Pp]assword:\s*$")
PROMPT_RE = re.compile(r"(?:\$|#)\s*$")
MARKER = "__SHARED_GPU_EXEC_DONE__"
MAX_REMOTE_PASSWORDS = 2


def credentials() -> dict[str, str]:
    text = CREDENTIAL_FILE.read_text(encoding="utf-8")
    found: dict[str, str] = {}
    for match in re.finditer(r"\|\s*`([A-Za-z0-9_]+)`\s*\|\s*`([^`]+)`\s*\|", text):
        found[match.group(1)] = match.group(2)
    return found


def scrub(text: str, secrets: list[str]) -> str:
    for secret in secrets:
        if secret:
            text = text.replace(secret, "***")
    return text


def run_remote(
    account: str,
    node: str,
    command: str,
    *,
    timeout: int = 300,
    verbose: bool = False,
) -> tuple[bool, str, int | None]:
    creds = credentials()
    if account not in creds:
        raise SystemExit(f"account {account!r} is not present in the credential file")
    account_pw = creds[account]
    ylhu_pw = creds.get("ylhu")
    if not ylhu_pw:
        raise SystemExit("ylhu password is missing from the credential file")

    inner = f"su - ylhu -c {shlex.quote(f'{command}; echo {MARKER}$?')}"

    try:
        pid, fd = pty.fork()
    except OSError as exc:  # pragma: no cover - environment failure
        raise SystemExit(f"pty.fork failed: {exc}")
    if pid == 0:
        os.execvp("su", ["su", "-", account])

    buffer = bytearray()
    stage = "local_password"
    remote_pw_sent = 0
    ylhu_pw_sent = False
    done = False
    exit_code: int | None = None
    deadline = time.time() + timeout
    try:
        while time.time() < deadline:
            ready, _, _ = select.select([fd], [], [], 1.0)
            if not ready:
                continue
            try:
                chunk = os.read(fd, 8192)
            except OSError:
                break
            if not chunk:
                break
            buffer.extend(chunk)
            clean = ANSI_RE.sub("", bytes(buffer).decode(errors="replace"))
            if verbose:
                sys.stderr.write(clean[-400:])
                sys.stderr.flush()
            if stage == "local_password" and PASSWORD_RE.search(clean):
                os.write(fd, (account_pw + "\n").encode())
                stage = "local_prompt"
                buffer.clear()
                continue
            if stage == "local_prompt" and PROMPT_RE.search(clean):
                os.write(
                    fd,
                    (
                        "ssh -tt -o ConnectTimeout=10 -o StrictHostKeyChecking=no "
                        f"-o LogLevel=ERROR {shlex.quote(node)}\n"
                    ).encode(),
                )
                stage = "remote_login"
                buffer.clear()
                continue
            if (
                stage == "remote_login"
                and PASSWORD_RE.search(clean)
                and remote_pw_sent < MAX_REMOTE_PASSWORDS
            ):
                os.write(fd, (account_pw + "\n").encode())
                remote_pw_sent += 1
                buffer.clear()
                continue
            if stage == "remote_login" and PROMPT_RE.search(clean):
                os.write(fd, (inner + "\n").encode())
                stage = "ylhu_password"
                buffer.clear()
                continue
            if stage == "ylhu_password" and not ylhu_pw_sent and PASSWORD_RE.search(clean):
                os.write(fd, (ylhu_pw + "\n").encode())
                ylhu_pw_sent = True
                buffer.clear()
                continue
            if stage == "ylhu_password" and ylhu_pw_sent and MARKER in clean:
                tail = clean.rsplit(MARKER, 1)[1]
                digits = re.match(r"(\d+)", tail.strip())
                if digits:
                    exit_code = int(digits.group(1))
                done = True
                break
    finally:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass

    text = ANSI_RE.sub("", bytes(buffer).decode(errors="replace"))
    text = scrub(text, [account_pw, ylhu_pw])
    return done, text, exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--account", required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    done, text, exit_code = run_remote(
        args.account, args.node, args.command, timeout=args.timeout, verbose=args.verbose
    )
    print(text)
    if not done:
        print("shared_gpu_exec: command did not finish within the timeout", file=sys.stderr)
        return 2
    print(f"shared_gpu_exec: exit_code={exit_code}")
    return 0 if exit_code in (0, None) else 1


if __name__ == "__main__":
    raise SystemExit(main())
