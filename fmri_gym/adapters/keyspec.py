"""KeySpec and its concrete flavors (Single / Multi / Passthrough)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class KeySpec(ABC):
    """A keyboard->action mapping for one game.

    Subclasses differ only in how they turn the matching combos into one
    action: :class:`SingleKeySpec` picks one, :class:`MultiKeySpec` ORs button
    vectors, :class:`PassthroughKeySpec` joins their names.

    :ivar combos: maps a frozenset of pressed key NAMES (pygame key names
        without the "K_" prefix, upper-case: "LEFT", "SPACE", "Z", ...) to the
        action to send. A combo overrides its parts (see :meth:`maximal`).
    :ivar noop: the action to send when no combo matches.
    """

    combos: dict[frozenset[str], Any]
    noop: Any

    @abstractmethod
    def resolve(self, held: frozenset[str]) -> Any:
        """Map currently-held keys to an action.

        :param held: frozenset of currently pressed key NAMES.
        :return: the action to send to the env this frame.
        """

    def _combos_for_overrides(self) -> dict[frozenset[str], Any]:
        """Base combos to merge curriculum overrides into.

        Default copies current combos so a partial remap keeps adapter defaults
        for unmentioned keys (e.g. Pong remaps UP/DOWN, keeps SPACE=FIRE).
        """
        return dict(self.combos)

    def maximal(self, held: frozenset[str]) -> list[frozenset[str]]:
        """Return the most specific combos fully held in ``held``.

        A combo matches when all of its keys are held; a match is dropped when
        another match is a strict superset of it, so e.g. an ``UP+SPACE`` combo
        shadows the plain ``UP`` and ``SPACE`` ones.

        :param held: frozenset of currently pressed key NAMES.
        :return: the matching combo key-sets, most specific ones only.
        """
        matched = [keys for keys in self.combos if keys <= held]
        return [keys for keys in matched
                if not any(keys < other for other in matched)]

    def apply_overrides(self, overrides: dict) -> None:
        """Merge curriculum-provided key combo overrides into this keymap.

        :param overrides: ``{"LEFT": action, "LEFT+SPACE": action}`` map from
            the curriculum (caller skips the call when empty / absent).
        """
        combos = self._combos_for_overrides()
        for combo_str, action in overrides.items():
            keys = frozenset(k.strip().upper() for k in combo_str.split("+"))
            combos[keys] = action
        self.combos = combos

    def key_to_action_map(self) -> dict:
        """Map single pressed KEY names to actions (for turn-based play).

        Resolving each key on its own (rather than reading ``combos`` directly)
        keeps the action in whatever shape the keymap flavour produces, e.g. a
        button vector for a :class:`MultiKeySpec`.

        :return: ``{key_name: action}`` for single-key combos only.
        """
        return {next(iter(ks)): self.resolve(ks) for ks in self.combos
                if len(ks) == 1}


@dataclass
class SingleKeySpec(KeySpec):
    """One action at a time: the most specific fully-held combo wins.

    The right choice for a ``Discrete`` action space, where the env can only be
    told about one action per step.
    """

    def resolve(self, held: frozenset[str]) -> Any:
        """Return the most specific matching combo's action, else ``noop``.

        :param held: frozenset of currently pressed key NAMES.
        :return: the action for the most specific matching combo, or ``noop``.
        """
        matches = self.maximal(held)
        if not matches:
            return self.noop
        return self.combos[max(matches, key=len)]


@dataclass
class MultiKeySpec(KeySpec):
    """Several buttons at once: OR the button vectors of all matching combos.

    The right choice for a ``MultiBinary`` action space, where held keys should
    combine (drive forward while turning, run while shooting, ...).

    :ivar button_map: optional ``action -> per-button 0/1 row`` table. When set,
        combo values (and ``noop``) are indices into it, so a curriculum keymap
        can stay written in the env's ``Discrete`` action indices; when ``None``
        the combo values are already 0/1 button vectors.
    """

    button_map: list[list[int]] | None = None

    def resolve(self, held: frozenset[str]) -> list[int]:
        """Return the OR of the button vectors of all matching combos.

        :param held: frozenset of currently pressed key NAMES.
        :return: a 0/1 button vector (``noop``'s when nothing matches).
        """
        vec = self.expand(self.noop)
        for keys in self.maximal(held):
            for i, pressed in enumerate(self.expand(self.combos[keys])):
                if pressed:
                    vec[i] = 1
        return vec

    def expand(self, action: Any) -> list[int]:
        """Return ``action`` as a per-button 0/1 vector.

        :param action: a :attr:`button_map` index, or a 0/1 vector already.
        :return: the button vector for ``action``.
        """
        if self.button_map is None:
            return [int(v) for v in action]
        return list(self.button_map[int(action)])


@dataclass
class PassthroughKeySpec(KeySpec):
    """Name the engine inputs to apply: the env itself interprets the names.

    For backends with no action space to index into, where ``step()`` turns the
    named inputs into engine input (browser games press the corresponding
    keyboard keys for real). Combo values are input NAMES, so the adapter defaults are
    usually identity (``{"SPACE"} -> "SPACE"``) while a curriculum keymap can
    point any physical key at any input (``{"B3"} -> "SPACE"``) -- which is what
    lets a scanner button box drive a game built around a keyboard.

    The action is those names, sorted and "+"-joined, which logs cleanly to npz
    ("" == nothing held).
    """

    def resolve(self, held: frozenset[str]) -> str:
        """Return the matching combos' input names, "+"-joined.

        :param held: frozenset of currently pressed key NAMES.
        :return: sorted input names joined with "+", or ``noop`` if no combo
            matches.
        """
        matches = self.maximal(held)
        if not matches:
            return self.noop
        return "+".join(sorted(str(self.combos[keys]) for keys in matches))

    def _combos_for_overrides(self) -> dict[frozenset[str], Any]:
        """Curriculum keys replace the adapter whitelist entirely.

        A game phase lists exactly the keys it forwards; empty start so
        unmentioned adapter defaults are dropped, not merged.
        """
        return {}
