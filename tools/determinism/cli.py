"""Sweep every ``configs/dbp_games/*.json`` through Protocol 1 (determinism).

    python -m tools.determinism.cli

Exit code is 0 only if every testable config came back bit-identical,
same-seed, cross-process. Protocol 2/3 (replay fidelity, savestate restore)
are slower and not part of this sweep; run them directly on one config when
you touch ``fmri_gym/replay.py``, ``fmri_gym/logging.py``, or a backend's
``capture``/``restore``/``warm_up``:

    python -m tools.determinism.replay reconstruction configs/dbp_games/<config>.json
    python -m tools.determinism.replay restore configs/dbp_games/<config>.json
"""
from __future__ import annotations

import glob
import importlib.util
import os
import platform
import subprocess
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# STK: no binary at all on darwin/arm64; needs a GL display on Linux, which CI lacks.
if platform.system() == "Darwin":
    _STK_REASON = ("stk_gym.binary.BinaryNotFound: no darwin/arm64 prebuilt binary "
                   "in the gym-v0.1.3 release")
else:
    _STK_REASON = "needs a real GL display (no Xvfb/EGL set up in this environment)"

# arc3 needs --extra arc3 (Python >=3.12) plus a one-time `python -m arc3_gym
# <game>` fetch -- checked live, not hardcoded, so an env with it set up still runs it.
_ARC3_REASON = "arc3_gym not importable here -- needs --extra arc3 on a Python >=3.12 interpreter"
_arc3_usable = importlib.util.find_spec("arc3_gym") is not None

# Configs this machine can't run, name -> why. Only for a real env/hardware gap.
_KNOWN_UNTESTABLE = {
    f"stk__{name}": _STK_REASON
    for name in ("cornfield_crossing", "hacienda", "lighthouse", "scotland", "snowmountain")
}
if not _arc3_usable:
    _KNOWN_UNTESTABLE.update({
        f"arc3__{name}": _ARC3_REASON
        for name in ("g50t", "ls20", "re86", "tr87", "tu93", "wa30")
    })


def main() -> int:
    configs = sorted(glob.glob(os.path.join(_REPO_ROOT, "configs", "dbp_games", "*.json")))
    failures = []
    tested = 0
    for path in configs:
        name = os.path.splitext(os.path.basename(path))[0]
        if name in _KNOWN_UNTESTABLE:
            print(f"SKIP  {name:45s} {_KNOWN_UNTESTABLE[name]}")
            continue
        tested += 1
        p = subprocess.run([sys.executable, "-m", "tools.determinism.cross_process", path],
                            capture_output=True, text=True, cwd=_REPO_ROOT)
        print(f"{'PASS' if p.returncode == 0 else 'FAIL'}  {name}")
        if p.returncode != 0:
            failures.append((name, p.stdout, p.stderr))

    print(f"\n{tested - len(failures)}/{tested} testable configs passed.")
    if failures:
        print("\nFAILURES:")
        for name, out, err in failures:
            print(f"--- {name} ---\n{out}{err}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
