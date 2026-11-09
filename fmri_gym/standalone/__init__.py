"""The tools around ``fmri-play`` that are not a run: each is a command of its own, and nothing
outside this package imports from it.

- ``rig-config`` (:mod:`.rig_setup.window`) and ``rig-measure`` (:mod:`.rig_setup.measure`):
  make, check and change a rig file, and measure the rig.
- ``session-editor`` (:mod:`.session_editor`): sessions, their runs' configs and the games'
  keys; Play hands over to ``fmri-play``. A session is a script (:mod:`.session_script`).
- ``pad-calibration`` (:mod:`.pad_scope`): calibrate a controller and see what it sends.
- :mod:`.photodiode`: the flip-to-photon offset of a rig.

The windows import PySide6 at the top, so the launchers here import them only when a command
starts one, and say how to install it when it is missing.
"""

from __future__ import annotations


def _needs_pyside6(exc: ImportError) -> None:
    """Re-raise a missing PySide6 with the line that installs it."""
    if not (exc.name or "").startswith("PySide6"):
        raise exc
    raise ImportError("this window needs PySide6: `uv sync --extra gui` (name the other "
                      "extras you use too, or uv removes them), or `pip install "
                      "PySide6-Essentials`") from exc


def rig_config_main() -> None:
    """``rig-config``: see :func:`fmri_gym.standalone.rig_setup.window.main`."""
    try:
        from .rig_setup.window import main
    except ImportError as exc:
        _needs_pyside6(exc)
    main()


def session_editor_main() -> None:
    """``session-editor``: see :func:`fmri_gym.standalone.session_editor.main`."""
    try:
        from .session_editor import main
    except ImportError as exc:
        _needs_pyside6(exc)
    main()
