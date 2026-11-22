"""An input device's keys, buttons and axes as the OSes number them, named; and each keyboard
key's USB HID usage, the number every layer agrees on.

A keyboard key is one usage on the USB HID keyboard page (``0x07``) in macOS's IOKit values,
and the same number is SDL's scancode, which pygame's key events carry. Linux's evdev numbers
its keys otherwise: :data:`EVDEV_KEYS` gives each its usage and its name, as the kernel's
``input-event-codes.h`` defines them. Windows' Raw Input gives a key's PC scancode (set 1),
which is its evdev code for most keys (:func:`evdev_code_of_windows_key`).
"""

from __future__ import annotations

#: The USB HID usage page of keyboard keys.
HID_KEYBOARD_PAGE = 0x07
#: The USB HID usage page of buttons (a game controller's, a button box's).
HID_BUTTON_PAGE = 0x09
#: evdev's event types.
EVDEV_TYPES = {0: "EV_SYN", 1: "EV_KEY", 2: "EV_REL", 3: "EV_ABS", 4: "EV_MSC"}
EV_SYN, EV_KEY, EV_ABS = 0, 1, 3
#: The ``EV_SYN`` code the kernel sends when a reader fell behind and events were dropped.
SYN_DROPPED = 3
#: evdev key code -> (its USB HID keyboard usage, its name without ``KEY_``).
EVDEV_KEYS: dict[int, tuple[int, str]] = {
    1: (41, "ESC"),
    2: (30, "1"),
    3: (31, "2"),
    4: (32, "3"),
    5: (33, "4"),
    6: (34, "5"),
    7: (35, "6"),
    8: (36, "7"),
    9: (37, "8"),
    10: (38, "9"),
    11: (39, "0"),
    12: (45, "MINUS"),
    13: (46, "EQUAL"),
    14: (42, "BACKSPACE"),
    15: (43, "TAB"),
    16: (20, "Q"),
    17: (26, "W"),
    18: (8, "E"),
    19: (21, "R"),
    20: (23, "T"),
    21: (28, "Y"),
    22: (24, "U"),
    23: (12, "I"),
    24: (18, "O"),
    25: (19, "P"),
    26: (47, "LEFTBRACE"),
    27: (48, "RIGHTBRACE"),
    28: (40, "ENTER"),
    29: (224, "LEFTCTRL"),
    30: (4, "A"),
    31: (22, "S"),
    32: (7, "D"),
    33: (9, "F"),
    34: (10, "G"),
    35: (11, "H"),
    36: (13, "J"),
    37: (14, "K"),
    38: (15, "L"),
    39: (51, "SEMICOLON"),
    40: (52, "APOSTROPHE"),
    41: (53, "GRAVE"),
    42: (225, "LEFTSHIFT"),
    43: (49, "BACKSLASH"),
    44: (29, "Z"),
    45: (27, "X"),
    46: (6, "C"),
    47: (25, "V"),
    48: (5, "B"),
    49: (17, "N"),
    50: (16, "M"),
    51: (54, "COMMA"),
    52: (55, "DOT"),
    53: (56, "SLASH"),
    54: (229, "RIGHTSHIFT"),
    55: (85, "KPASTERISK"),
    56: (226, "LEFTALT"),
    57: (44, "SPACE"),
    58: (57, "CAPSLOCK"),
    59: (58, "F1"),
    60: (59, "F2"),
    61: (60, "F3"),
    62: (61, "F4"),
    63: (62, "F5"),
    64: (63, "F6"),
    65: (64, "F7"),
    66: (65, "F8"),
    67: (66, "F9"),
    68: (67, "F10"),
    69: (83, "NUMLOCK"),
    70: (71, "SCROLLLOCK"),
    71: (95, "KP7"),
    72: (96, "KP8"),
    73: (97, "KP9"),
    74: (86, "KPMINUS"),
    75: (92, "KP4"),
    76: (93, "KP5"),
    77: (94, "KP6"),
    78: (87, "KPPLUS"),
    79: (89, "KP1"),
    80: (90, "KP2"),
    81: (91, "KP3"),
    82: (98, "KP0"),
    83: (99, "KPDOT"),
    86: (100, "102ND"),
    87: (68, "F11"),
    88: (69, "F12"),
    96: (88, "KPENTER"),
    97: (228, "RIGHTCTRL"),
    98: (84, "KPSLASH"),
    99: (70, "SYSRQ"),
    100: (230, "RIGHTALT"),
    102: (74, "HOME"),
    103: (82, "UP"),
    104: (75, "PAGEUP"),
    105: (80, "LEFT"),
    106: (79, "RIGHT"),
    107: (77, "END"),
    108: (81, "DOWN"),
    109: (78, "PAGEDOWN"),
    110: (73, "INSERT"),
    111: (76, "DELETE"),
    116: (102, "POWER"),
    117: (103, "KPEQUAL"),
    119: (72, "PAUSE"),
    125: (227, "LEFTMETA"),
    126: (231, "RIGHTMETA"),
    127: (101, "COMPOSE"),
    183: (104, "F13"),
    184: (105, "F14"),
    185: (106, "F15"),
    186: (107, "F16"),
    187: (108, "F17"),
    188: (109, "F18"),
    189: (110, "F19"),
    190: (111, "F20"),
    191: (112, "F21"),
    192: (113, "F22"),
    193: (114, "F23"),
    194: (115, "F24"),
}
#: evdev's game controller and mouse buttons, by code.
EVDEV_BUTTONS = {
    272: "BTN_LEFT", 273: "BTN_RIGHT", 274: "BTN_MIDDLE", 288: "BTN_TRIGGER", 289: "BTN_THUMB",
    290: "BTN_THUMB2", 291: "BTN_TOP", 292: "BTN_TOP2", 293: "BTN_PINKIE", 294: "BTN_BASE",
    295: "BTN_BASE2", 296: "BTN_BASE3", 297: "BTN_BASE4", 298: "BTN_BASE5", 299: "BTN_BASE6",
    304: "BTN_SOUTH", 305: "BTN_EAST", 306: "BTN_C", 307: "BTN_NORTH", 308: "BTN_WEST",
    309: "BTN_Z", 310: "BTN_TL", 311: "BTN_TR", 312: "BTN_TL2", 313: "BTN_TR2",
    314: "BTN_SELECT", 315: "BTN_START", 316: "BTN_MODE", 317: "BTN_THUMBL", 318: "BTN_THUMBR",
    544: "BTN_DPAD_UP", 545: "BTN_DPAD_DOWN", 546: "BTN_DPAD_LEFT", 547: "BTN_DPAD_RIGHT",
}
#: evdev's absolute axes, by code.
EVDEV_AXES = {0: "ABS_X", 1: "ABS_Y", 2: "ABS_Z", 3: "ABS_RX", 4: "ABS_RY", 5: "ABS_RZ",
              9: "ABS_GAS", 10: "ABS_BRAKE", 16: "ABS_HAT0X", 17: "ABS_HAT0Y"}
#: Windows' Raw Input key flags that mark the scancode's E0 and E1 prefixes.
WINDOWS_KEY_E0, WINDOWS_KEY_E1 = 2, 4
#: A Windows key whose make code another key shares, told apart by its prefix -> its evdev
#: code.
WINDOWS_PREFIXED_KEYS: dict[tuple[int, int], int] = {
    (0x1C, WINDOWS_KEY_E0): 96, (0x1D, WINDOWS_KEY_E0): 97, (0x35, WINDOWS_KEY_E0): 98,
    (0x37, WINDOWS_KEY_E0): 99, (0x38, WINDOWS_KEY_E0): 100, (0x47, WINDOWS_KEY_E0): 102,
    (0x48, WINDOWS_KEY_E0): 103, (0x49, WINDOWS_KEY_E0): 104, (0x4B, WINDOWS_KEY_E0): 105,
    (0x4D, WINDOWS_KEY_E0): 106, (0x4F, WINDOWS_KEY_E0): 107, (0x50, WINDOWS_KEY_E0): 108,
    (0x51, WINDOWS_KEY_E0): 109, (0x52, WINDOWS_KEY_E0): 110, (0x53, WINDOWS_KEY_E0): 111,
    (0x5B, WINDOWS_KEY_E0): 125, (0x5C, WINDOWS_KEY_E0): 126, (0x5D, WINDOWS_KEY_E0): 127,
    (0x5E, WINDOWS_KEY_E0): 116, (0x1D, WINDOWS_KEY_E1): 119,
}
#: F13 to F24's make codes on Windows -> their evdev codes.
WINDOWS_F13_TO_F24 = {**{0x64 + index: 183 + index for index in range(11)}, 0x76: 194}
#: The highest make code with no prefix that is its own evdev code.
_LAST_PLAIN_MAKE_CODE = 88


def evdev_code_of_windows_key(make_code: int, flags: int) -> int | None:
    """The evdev code of a key Windows' Raw Input gives by its make code and flags; ``None``
    for a key :data:`EVDEV_KEYS` does not have."""
    prefix = flags & (WINDOWS_KEY_E0 | WINDOWS_KEY_E1)
    if prefix:
        return WINDOWS_PREFIXED_KEYS.get((make_code, prefix))
    if make_code in WINDOWS_F13_TO_F24:
        return WINDOWS_F13_TO_F24[make_code]
    return make_code if make_code <= _LAST_PLAIN_MAKE_CODE and make_code in EVDEV_KEYS else None


#: USB HID keyboard usage -> the key's name.
KEY_NAME_OF_USAGE = {usage: name for usage, name in EVDEV_KEYS.values()}
