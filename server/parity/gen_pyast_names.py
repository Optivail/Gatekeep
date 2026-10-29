#!/usr/bin/env python3
"""Build the Unicode name table used by engines/pyast_names.go.

CPython resolves \\N{name} in string literals through its own name database
(Unicode 15.0.0 for CPython 3.12). That database is not in the Go standard
library, so this script derives it from two sources and checks each entry
against CPython before writing it:

  * the single-code-point entries of a Perl charnames table (Name.pl, Unicode
    15.0.0), which list primary names and name aliases. Entries that CPython
    rejects (for example "LINE FEED (LF)") are dropped;
  * the algorithmic families that CPython resolves outside its table: CJK
    UNIFIED IDEOGRAPH ranges (stored as ranges), and HANGUL SYLLABLE names,
    KHITAN, NUSHU and CJK COMPATIBILITY IDEOGRAPH names (stored as names).

Named sequences are not included: a string literal rejects them.

Output (gzip-compressed text, one record per line):
    N<TAB>HEX<TAB>NAME     a name CPython accepts in \\N{...}, case as written
    R<TAB>LO<TAB>HI        a CJK UNIFIED IDEOGRAPH range (hex, inclusive)

Usage (run with the Gatekeep venv):
    python -I server/parity/gen_pyast_names.py --perl-name-file PATH --out OUT.gz
"""

from __future__ import annotations

import argparse
import gzip
import re
import sys
import unicodedata


def perl_entries(path: str) -> list[tuple[int, str]]:
    text = open(path, encoding="utf-8").read()
    body = text.split("return <<'END';\n", 1)[1].rsplit("END", 1)[0]
    out: list[tuple[int, str]] = []
    for block in body.split("\n\n"):
        lines = block.split("\n")
        if len(lines) < 2 or not lines[0].strip():
            continue
        cps = lines[0].split()
        if len(cps) != 1:  # named sequence
            continue
        for name in lines[1:]:
            out.append((int(cps[0], 16), name))
    return out


def cpython_lookup(name: str) -> int | None:
    try:
        ch = unicodedata.lookup(name)
    except KeyError:
        return None
    return ord(ch) if len(ch) == 1 else None


ALGORITHMIC = ("HANGUL SYLLABLE ", "CJK UNIFIED IDEOGRAPH-")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--perl-name-file", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    records: dict[str, int] = {}
    rejected: list[str] = []
    for cp, name in perl_entries(args.perl_name_file):
        if name.startswith(ALGORITHMIC):
            continue  # derived below, not stored
        if cpython_lookup(name) == cp:
            records.setdefault(name, cp)
        else:
            rejected.append(name)

    # Names CPython knows that Perl's table lacks: round trip every code point.
    fam_names = 0
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        try:
            name = unicodedata.name(chr(cp))
        except ValueError:
            continue
        if name.startswith(ALGORITHMIC) or name in records:
            continue
        if cpython_lookup(name) == cp:
            records[name] = cp
            fam_names += 1
        else:
            print(f"warning: name() of U+{cp:04X} does not round-trip", file=sys.stderr)

    # Hangul syllables: CPython builds them from jamo names; store each one.
    hangul = 0
    for cp in range(0xAC00, 0xD7A4):
        name = unicodedata.name(chr(cp))
        if cpython_lookup(name) != cp:
            raise SystemExit(f"hangul round trip failed at U+{cp:04X}")
        records[name] = cp
        hangul += 1

    # CJK UNIFIED IDEOGRAPH: CPython accepts 4 or 5 uppercase hex digits.
    ranges: list[tuple[int, int]] = []
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        ok = cpython_lookup(f"CJK UNIFIED IDEOGRAPH-{cp:04X}") == cp
        if not ok and cp > 0xFFFF:
            ok = cpython_lookup(f"CJK UNIFIED IDEOGRAPH-{cp:05X}") == cp
        if ok:
            if ranges and ranges[-1][1] == cp - 1:
                ranges[-1] = (ranges[-1][0], cp)
            else:
                ranges.append((cp, cp))

    lines = []
    for name, cp in sorted(records.items(), key=lambda kv: kv[1]):
        lines.append(f"N\t{cp:X}\t{name}")
    for lo, hi in ranges:
        lines.append(f"R\t{lo:X}\t{hi:X}")
    with gzip.GzipFile(args.out, "wb", mtime=0) as fh:
        fh.write(("\n".join(lines) + "\n").encode("utf-8"))

    print(f"names: {len(records)} (families {fam_names}, hangul {hangul}); "
          f"cjk ranges: {len(ranges)}; rejected perl names: {rejected}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
