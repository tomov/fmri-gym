"""Build the supplied NetHackDS engine with a desktop window transport."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

REVISION = "3.6.0-desktop-4-cursor"


def default_home() -> Path:
    """Return the explicitly overridable native build directory."""
    return Path(os.environ.get("NETHACK_GAMEPAD_HOME",
                               Path.home() / ".cache" / "fmri-gym" / REVISION)).expanduser()


def source_root() -> Path:
    """Return the supplied source checkout adjacent to this environment package."""
    return Path(__file__).resolve().parents[3] / "external" / "nethack_ds"


def function_source(text: str, signature: str) -> str:
    """Extract one C function verbatim, respecting braces inside strings and comments."""
    start = text.index(signature)
    masked = re.sub(r'/\*.*?\*/|//[^\n]*|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
                    lambda match: " " * len(match[0]), text, flags=re.S)
    opening = masked.index("{", start)
    depth = 1
    position = opening + 1
    while depth:
        depth += (masked[position] == "{") - (masked[position] == "}")
        position += 1
    return text[start:position] + "\n"


def ds_commands(text: str) -> list[dict]:
    """Read the original DS command table, excluding only DS configuration commands."""
    block = text.split("nds_cmd_t cmdlist[] = {", 1)[1].split("};", 1)[0]
    block = re.sub(r"/\*.*?\*/|//[^\n]*", "", block, flags=re.S)
    entries = []
    for match in re.finditer(r'\{\s*((?:[MC]\()?\'(?:\\.|[^\'])+\'\)?)\s*,\s*"([^"]+)"', block):
        expression, label = match.groups()
        modifier = expression[0] if expression.startswith(("M(", "C(")) else ""
        literal = expression[2:-1] if modifier else expression
        key = ord(ast.literal_eval(literal))
        key = key | 128 if modifier == "M" else key & 31 if modifier == "C" else key
        entries.append({"label": label, "key": key})
    if len(entries) < 50:
        raise ValueError("The supplied NetHackDS command table was not recognized")
    return entries


def ds_keyboard(source: Path) -> dict:
    """Read the DS keyboard's physical rows, widths and shift-pair table."""
    keyboard = (source / "sys/nds/arm9/include/nds_kbd_layout.h").read_text()
    rows = []
    for block in re.findall(r"const nds_kbd_key row\d\[\] = \{(.*?)\};", keyboard, re.S):
        row = []
        for match in re.finditer(r"\.width\s*=\s*(\d+)\s*,\s*(?:\.code\s*=\s*)?"
                                 r"(?:\(u16\))?\s*('(?:\\.|[^'])*'|K_\w+)", block):
            width, code = match.groups()
            row.append({"width": int(width), "code": ord(ast.literal_eval(code))
                        if code.startswith("'") else code})
        rows.append(row)
    text = (source / "sys/nds/arm9/src/nds_kbd.c").read_text()
    shifts = ast.literal_eval(re.search(r'char shifts\[\] = ("(?:\\.|[^"\\])*");', text)[1])
    return {"rows": rows, "shifts": shifts}


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if text.count(old) != 1:
        raise ValueError(f"Expected one patch anchor in {path}: {old[:60]}")
    path.write_text(text.replace(old, new, 1))


def prepare(source: Path, home: Path) -> Path:
    """Copy upstream files; change only the platform transport and optional replay hooks."""
    work = home / "source"
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(source, work)
    package = Path(__file__).resolve().parent
    shutil.copy2(package / "bridge.c", work / "src" / "desktop.c")
    replace_once(work / "src/do_name.c", "        c = nh_poskey(&tx, &ty, &sidx);",
                 "        { extern boolean desktop_targeting;\n"
                 "          desktop_targeting = TRUE;\n"
                 "          c = nh_poskey(&tx, &ty, &sidx);\n"
                 "          desktop_targeting = FALSE; } ")
    windows = (source / "sys/nds/arm9/src/nds_win.c").read_text()
    commands = (source / "sys/nds/arm9/src/nds_cmd.c").read_text()
    (work / "src/nds_selection.inc").write_text(
        function_source(windows, "void nds_player_selection()"))
    (work / "src/nds_extended.inc").write_text(
        function_source(commands, "int nds_get_ext_cmd()"))
    questions = (source / "sys/nds/arm9/src/nds_yn.c").read_text()
    (work / "src/nds_choices.inc").write_text("\n".join(function_source(questions, signature)
        for signature in ("struct obj *obj_for_let", "int class_slot_for_class",
                          "void _nds_insert_choice", "char *_nds_parse_choices")))
    replace_once(work / "src/windows.c", '#include "hack.h"',
                 '#include "hack.h"\nextern struct window_procs desktop_procs;')
    replace_once(work / "src/windows.c", '} winchoices[] = {',
                 '} winchoices[] = {\n    { &desktop_procs, 0 },')
    replace_once(work / "util/makedefs.c", 'static struct win_info window_opts[] = {',
                 'static struct win_info window_opts[] = {\n'
                 '    { "desktop", "NetHackDS single-screen desktop transport" },')
    # The legacy BMP writer assumes Windows' 32-bit long, including on LP64 hosts.
    bitmap = work / "win/share/tile2bmp.c"
    bitmap_text = bitmap.read_text()
    start = bitmap_text.index("typedef struct tagBMIH")
    stop = bitmap_text.index("#endif /* __GNUC__ */", start)
    header = re.sub(r"\blong\b", "int", bitmap_text[start:stop])
    bitmap.write_text(bitmap_text[:start] + header + bitmap_text[stop:])
    replace_once(work / "src/hacklib.c", '    /* the types are different enough here',
                 '    if (getenv("FMRI_NETHACK_SEED"))\n'
                 '        seed = strtoul(getenv("FMRI_NETHACK_SEED"), 0, 10);\n\n'
                 '    /* the types are different enough here')
    replace_once(work / "src/hacklib.c", '    (void) time((TIME_type) &datetime);',
                 '    if (getenv("FMRI_NETHACK_EPOCH"))\n'
                 '        return (time_t) strtol(getenv("FMRI_NETHACK_EPOCH"), 0, 10);\n'
                 '    (void) time((TIME_type) &datetime);')
    # Opt-in GnollHack-derived rules; patches apply only to the disposable copy.
    shutil.copy2(package / "modern.inc", work / "src/modern.inc")
    replace_once(work / "include/you.h", '    int umortality;',
                 '    boolean desktop_modern, desktop_revival_pending, desktop_entry_valid;\n'
                 '    long desktop_revivals, desktop_shield_until, desktop_entry_id;\n'
                 '    d_level desktop_entry_level;\n'
                 '    xchar desktop_entry_x, desktop_entry_y;\n'
                 '    char desktop_entry_reason[32];\n'
                 '    int desktop_death_how;\n'
                 '    char desktop_death_cause[BUFSZ];\n'
                 '    int umortality;')
    replace_once(work / "include/extern.h", '#undef E',
                 'E long NDECL(desktop_score_divisor);\n'
                 'E void NDECL(desktop_modern_init);\n'
                 'E void FDECL(desktop_record_entry, (const char *));\n'
                 'E void NDECL(desktop_finish_revival);\n'
                 'E void FDECL(desktop_revive, (int));\n'
                 'E void FDECL(desktop_game_event, (const char *, int, const char *));\n#undef E')
    replace_once(work / "src/allmain.c", '    u_on_upstairs();',
                 '    u_on_upstairs();\n    desktop_modern_init();')
    replace_once(work / "src/end.c", '    really_done(how);\n}',
                 '    if (u.desktop_modern && how <= GENOCIDED) {\n'
                 '        desktop_revive(how);\n'
                 '        killer.name[0] = 0;\n        killer.format = 0;\n        return;\n    }\n'
                 '    if (how <= GENOCIDED) desktop_game_event("death", how, killer.name);\n'
                 '    really_done(how);\n}')
    ending = work / "src/end.c"
    ending.write_text(ending.read_text().replace("u.urexp, plur(u.urexp)",
        "u.urexp / desktop_score_divisor(), plur(u.urexp / desktop_score_divisor())"))
    replace_once(ending, "    topten(how, endtime);",
                 "    u.urexp /= desktop_score_divisor();\n    topten(how, endtime);")
    with (work / "src/end.c").open("a") as output:
        output.write('\n#include "modern.inc"\n')
    # Record placement before stair/fall damage and other arrival effects.
    for anchor in ('        u_on_newpos(ttrap->tx, ttrap->ty);',
                   '                u_on_dnstairs();',
                   '                u_on_upstairs();',
                   '        u_on_rndspot((up ? 1 : 0) | (was_in_W_tower ? 2 : 0));'):
        replace_once(work / "src/do.c", anchor, anchor + '\n'
            '        desktop_record_entry(portal ? "portal" : falling ? "fall"\n'
            '            : at_stairs ? (at_ladder ? "ladder" : newdungeon ? "branch" : "stairs")\n'
            '            : "level teleport");')
    combat = work / "src/mhitu.c"
    combat.write_text(combat.read_text().replace('if (u.uinvulnerable)',
        'if (u.uinvulnerable || (u.desktop_modern && (u.desktop_revival_pending '
        '|| moves < u.desktop_shield_until)))'))
    for name in ("include/you.h", "include/extern.h", "src/allmain.c", "src/end.c",
                 "src/do.c", "src/mhitu.c"):
        path = work / name
        path.write_text('/* Modified 2026-11-12: desktop Modern-style revival. '
                        'See modern.inc and build.py. */\n' + path.read_text())
    runtime = home / "runtime"
    runtime.mkdir(exist_ok=True)
    (work / "desktop.hints").write_text(f'''#-PRE
PREFIX={home}
HACKDIR={runtime}
INSTDIR={runtime}
VARDIR={runtime}
SHELLDIR=
CC=cc
LINK=$(CC)
CFLAGS=-O2 -std=gnu89 -fcommon -I../include -DNOTTYGRAPHICS -DSELF_RECOVER -DTEXT_TOMBSTONE -DDEFAULT_WINDOW_SYS='"desktop"'
WINSRC=desktop.c tile.c
WINOBJ=desktop.o tile.o
WINLIB=
CHOWN=true
CHGRP=true
GAMEPERM=0755
FILEPERM=0644
EXEPERM=0755
DIRPERM=0755
VARDIRPERM=0755
VARFILEPERM=0600
''')
    subprocess.run(["sh", "sys/unix/setup.sh", "desktop.hints"], cwd=work, check=True)
    # Upstream ships generated lex/yacc sources; no parser generator download is needed.
    for name in ("lev_lex.c", "lev_yacc.c", "dgn_lex.c", "dgn_yacc.c"):
        shutil.copy2(work / "sys/share" / name, work / "util" / name)
        (work / "util" / name).touch()
    for name in ("lev_comp.h", "dgn_comp.h"):
        candidate = work / "sys/share" / name
        if candidate.exists():
            shutil.copy2(candidate, work / "include" / name)
            (work / "include" / name).touch()
    return work


def main() -> None:
    """Build offline using the platform C compiler and make."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=source_root())
    parser.add_argument("--home", type=Path, default=default_home())
    parser.add_argument("--test-driver", action="store_true",
                        help="compile isolated engine test commands; never use for participant play")
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    if not shutil.which("cc") or not shutil.which("make"):
        raise RuntimeError("A C compiler and make are required (macOS: xcode-select --install)")
    if not (args.source / "sys/nds/arm9/src/nds_win.c").is_file():
        raise FileNotFoundError(f"NetHackDS source is missing at {args.source}")
    home = args.home.resolve()
    if any(character.isspace() for character in str(home)):
        raise ValueError("The upstream Makefiles require a build path without spaces; "
                         "set NETHACK_GAMEPAD_HOME to such a path")
    home.mkdir(parents=True, exist_ok=True)
    work = prepare(args.source.resolve(), home)
    if args.test_driver:
        hints = work / "src/Makefile"
        hints.write_text(hints.read_text().replace("-DNOTTYGRAPHICS",
                                                 "-DNOTTYGRAPHICS -DDESKTOP_TEST_DRIVER"))
    log = home / "build.log"
    with log.open("w") as output:
        try:
            subprocess.run(["make", "-C", "src", "../include/pm.h"], cwd=work,
                           stdout=output, stderr=subprocess.STDOUT, check=True)
            subprocess.run(["make", "-C", "util", "../src/tile.c"], cwd=work,
                           stdout=output, stderr=subprocess.STDOUT, check=True)
            subprocess.run(["make", "-C", "src", f"-j{args.jobs}"], cwd=work,
                           stdout=output, stderr=subprocess.STDOUT, check=True)
            # The old data generators share grep.tmp and must run serially.
            subprocess.run(["make", "install"], cwd=work,
                           stdout=output, stderr=subprocess.STDOUT, check=True)
            subprocess.run(["make", "-C", "util", "tile2bmp"], cwd=work,
                           stdout=output, stderr=subprocess.STDOUT, check=True)
            subprocess.run([str(work / "util/tile2bmp"), str(home / "runtime/tiles.bmp")],
                           cwd=work / "util",
                           stdout=output, stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"Native build failed; read {log}") from error
    runtime = home / "runtime"
    commands = (args.source / "sys/nds/arm9/src/nds_cmd.c").read_text()
    (runtime / "commands.json").write_text(json.dumps(ds_commands(commands), indent=2))
    (runtime / "keyboard.json").write_text(json.dumps(ds_keyboard(args.source)))
    shutil.copy2(args.source / "sys/nds/NetHack/font.bdf", runtime / "font.bdf")
    shutil.copy2(args.source / "sys/nds/NetHack/nhlogo.bmp", runtime / "logo.bmp")
    from PIL import BdfFontFile
    with (runtime / "font.bdf").open("rb") as bitmap_font:
        BdfFontFile.BdfFontFile(bitmap_font).save(str(runtime / "font"))
    shutil.copy2(args.source / "dat/license", runtime / "license")
    manifest = {"revision": REVISION, "engine": "NetHack 3.6.0", "test_driver": args.test_driver, "source": str(args.source),
                "modern_sha256": hashlib.sha256((work / "src/modern.inc").read_bytes()).hexdigest(),
                "engine_sha256": hashlib.sha256((runtime / "nethack").read_bytes()).hexdigest(),
                "transport_sha256": hashlib.sha256((work / "src/desktop.c").read_bytes()).hexdigest()}
    (runtime / "build.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Built {runtime / 'nethack'}")


if __name__ == "__main__":
    main()
