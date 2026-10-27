"""The fmri-gym GUIs: one window per job, around ``fmri-play``, which none of them changes.

- ``rig-config`` (:mod:`.rig_config`): one rig file -- how a rig plays every run.
- ``session-editor`` (:mod:`.session_editor`): sessions, their runs' configs and
  the games' keys; Play hands over to ``fmri-play``.

What they share is here, and needs no Qt: a form's :class:`Field` and the
parsing of what is typed into one. The widgets they share are
:mod:`.widgets`. The windows import PySide6 at the top, so this module imports
them only when a command starts one, and says how to install it when it is
missing. Field labels are the config keys and each field's tip says what it
does, so the dialog and the file read alike.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class Field:
    """One labelled row of a form.

    ``kind``: ``str`` | ``int`` | ``float`` | ``bool`` | ``choice`` (fixed
    list) | ``combo`` (list plus free text) | ``text`` (multi-line string,
    always written) | ``lines`` (multi-line, one list item per line) | ``list``
    (one line, items separated by commas or spaces; numbers stay numbers). For the
    other kinds a blank entry means "not set", so the engine default applies;
    a ``bool`` is only written when it differs from ``default``.
    """

    key: str
    label: str
    kind: str = "str"
    choices: Sequence[str] = ()
    tip: str = ""
    default: Any = None


def parse_value(field: Field, raw: Any) -> Any:
    """Turn a widget's raw value into the config value (``None`` = not set).

    :raises ValueError: naming the field, when a number does not parse.
    """
    if field.kind == "bool":
        return bool(raw)
    text = str(raw)
    if field.kind == "text":
        return text  # a message's text is always stated, even when empty
    if not text.strip():
        return None
    if field.kind == "lines":
        return [line for line in text.split("\n") if line.strip()]
    if field.kind == "list":
        return [_item(item) for item in re.split(r"[,\s]+", text.strip()) if item]
    if field.kind not in ("int", "float"):
        return text.strip()
    try:
        return _number(text, field.kind)
    except ValueError as exc:
        raise ValueError(f"{field.label}: expected a number, got {text.strip()!r}") from exc


def _number(text: str, kind: str) -> int | float:
    """``"30"`` stays the int 30 in a float field, so Save does not turn it into 30.0."""
    try:
        return int(text)
    except ValueError:
        if kind == "int":
            raise
        return float(text)


def _item(text: str) -> int | float | str:
    """A ``list`` item: a number when it reads as one."""
    try:
        return _number(text, "float")
    except ValueError:
        return text


def format_value(field: Field, value: Any) -> Any:
    """The inverse of :func:`parse_value`: config value -> widget value."""
    if field.kind == "bool":
        return bool(field.default if value is None else value)
    if value is None:
        return ""
    if field.kind == "list" and isinstance(value, list):
        return ", ".join(str(v) for v in value)
    if isinstance(value, (list, tuple)):
        return "\n".join(str(v) for v in value)
    return str(value)


def field_values(fields: Sequence[Field], raw: dict[str, Any]) -> dict[str, Any]:
    """Parse a whole form; blanks are dropped, bools only kept when non-default."""
    out: dict[str, Any] = {}
    for f in fields:
        value = parse_value(f, raw.get(f.key, ""))
        if value is None or (f.kind == "bool" and value == f.default):
            continue
        out[f.key] = value
    return out


def parse_action(text: str) -> Any:
    """``"2"`` -> 2, ``"[1,0,0]"`` -> list, anything else -> the string itself."""
    text = text.strip()
    try:
        return json.loads(text)
    except ValueError:
        return text


def format_action(action: Any) -> str:
    """The inverse of :func:`parse_action`.

    A string is shown bare only if it reads back as itself: the action ``"1"``
    is shown quoted, or it would come back as the int 1 and rebind the key.
    """
    if isinstance(action, str) and parse_action(action) == action:
        return action
    return json.dumps(action)


def _needs_pyside6(exc: ImportError) -> None:
    """Re-raise a missing PySide6 with the line that installs it."""
    if not (exc.name or "").startswith("PySide6"):
        raise exc
    raise ImportError("this window needs PySide6: `uv sync --extra gui` (name the other "
                      "extras you use too, or uv removes them), or `pip install "
                      "PySide6-Essentials`") from exc


def rig_config_main() -> None:
    """``rig-config``: see :func:`fmri_gym.gui.rig_config.main`."""
    try:
        from .rig_config import main
    except ImportError as exc:
        _needs_pyside6(exc)
    main()


def session_editor_main() -> None:
    """``session-editor``: see :func:`fmri_gym.gui.session_editor.main`."""
    try:
        from .session_editor import main
    except ImportError as exc:
        _needs_pyside6(exc)
    main()
