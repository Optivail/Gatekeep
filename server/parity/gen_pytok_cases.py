#!/usr/bin/env python3
"""Generate the cl100k_base oracle fixture for server/pytok.

Run with CPython (needs tiktoken and its offline cache):

    python3 -I server/parity/gen_pytok_cases.py

It writes pytok/testdata/cl100k_cases.jsonl: one JSON object per line with the
input text, the token count and a 16-hex-digit SHA-256 of the token ids (the
ids joined by spaces). Entries that tiktoken refuses carry "err":"special".
The Go test regenerates nothing; it checks the committed file.

The text sample is deterministic (fixed seed) and covers prose, code, JSON,
CJK, emoji, CRLF, long lines, unusual code points, empty input, and near-miss
and exact special-token strings.
"""
import hashlib
import json
import os
import random
import sys

import tiktoken

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "pytok", "testdata", "cl100k_cases.jsonl")
SEED = 20261029
TOTAL = 5000

ENC = tiktoken.get_encoding("cl100k_base")
SPECIALS = sorted(ENC.special_tokens_set)

WORDS = ("the of and to in a is that for it as was with be by on not he this are or his from at "
         "which but have an they you were her she there one all we their state when up also has "
         "error build failed test server request response token model context cache file path "
         "config value return function variable null true false import export class method").split()
PUNCT = [".", ",", ";", ":", "!", "?", " - ", " (", ") ", "'", "\"", "...", "/", "#", "@", "%"]
CODE_LINES = [
    "def {f}({a}, {b}=None):", "    return {a} + {b} * 2", "func {f}(ctx context.Context) error {{",
    "\tif err != nil {{ return err }}", "const {a} = require('{b}');", "for i := 0; i < {n}; i++ {{",
    "SELECT {a}, COUNT(*) FROM {b} GROUP BY {a};", "import {a} from \"{b}\"", "    # {f} helper",
    "x = [{n}, {n}, {n}]", "}}", "{a}: {n}", "value->{a}[{n}]", "<div class=\"{a}\">{b}</div>",
]
IDENTS = ["alpha", "beta", "gamma", "config", "server", "handler", "tokenCount", "x_1", "items", "fooBar"]
CJK_PUNCT = "。，、：；！？「」"
EMOJI = ["\U0001F600", "\U0001F680", "\U0001F44D\U0001F3FD", "\U0001F468‍\U0001F469‍\U0001F467",
         "\U0001F1F3\U0001F1F1", "❤️", "\U0001F926\U0001F3FF‍♂️", "✅", "\U0001F980"]


def prose(rng, n):
    out = []
    for _ in range(n):
        w = rng.choice(WORDS)
        if rng.random() < 0.1:
            w = w.capitalize()
        out.append(w)
        if rng.random() < 0.15:
            out.append(rng.choice(PUNCT))
        else:
            out.append(" ")
        if rng.random() < 0.02:
            out.append("\n")
    return "".join(out).strip()


def code(rng):
    lines = []
    for _ in range(rng.randint(2, 8)):
        lines.append(rng.choice(CODE_LINES).format(
            f=rng.choice(IDENTS), a=rng.choice(IDENTS), b=rng.choice(IDENTS), n=rng.randint(0, 999)))
    sep = "\r\n" if rng.random() < 0.2 else "\n"
    return sep.join(lines)


def json_value(rng, depth=0):
    r = rng.random()
    if depth > 2 or r < 0.35:
        return rng.choice([rng.randint(-10**6, 10**6), rng.random() * 100, None, True, False,
                           prose(rng, rng.randint(1, 4)), "café 中文 \U0001F600"])
    if r < 0.65:
        return [json_value(rng, depth + 1) for _ in range(rng.randint(0, 4))]
    return {prose(rng, 1): json_value(rng, depth + 1) for _ in range(rng.randint(0, 4))}


def cjk(rng):
    pools = [range(0x4E00, 0x9FFF), range(0x3041, 0x3096), range(0x30A1, 0x30FA), range(0xAC00, 0xD7A3)]
    out = []
    for _ in range(rng.randint(3, 60)):
        r = rng.random()
        if r < 0.1:
            out.append(rng.choice(CJK_PUNCT))
        elif r < 0.15:
            out.append(" ")
        else:
            out.append(chr(rng.choice(rng.choice(pools))))
    return "".join(out)


def emoji(rng):
    out = []
    for _ in range(rng.randint(1, 20)):
        out.append(rng.choice(EMOJI))
        if rng.random() < 0.3:
            out.append(rng.choice(" ,.!"))
    return "".join(out)


def unicode_edge(rng):
    specials = [0x0, 0x7F, 0x80, 0xA0, 0xAD, 0x200B, 0x200D, 0x200E, 0x202E, 0xFEFF, 0xFFFD, 0xFFFE, 0xFFFF,
                0xE000, 0xF8FF, 0xFFFFD, 0x10FFFF, 0x0300, 0x0301, 0x1F3FB, 0xE0001, 0xE0080, 0x1D11E, 0x10000]
    out = []
    for _ in range(rng.randint(1, 40)):
        if rng.random() < 0.5:
            c = rng.choice(specials)
        else:
            c = rng.randint(0, 0x10FFFF)
            if 0xD800 <= c <= 0xDFFF:
                c = 0x41
        out.append(chr(c))
    return "".join(out)


def special_like(rng):
    near = ["<|endoftext|", "|endoftext|>", "<|endoft ext|>", "<|fim_prefix", "<|endofprompt>",
            "endoftext", "<|im_start|>", "<|endoftext|>x"[:-1]]
    parts = [prose(rng, rng.randint(1, 6))]
    for _ in range(rng.randint(1, 3)):
        parts.append(rng.choice(near))
        parts.append(rng.choice([" ", "", "\n"]))
    return "".join(parts)


def special_exact(rng):
    tok = rng.choice(SPECIALS)
    return prose(rng, rng.randint(0, 5)) + rng.choice(["", " "]) + tok + rng.choice(["", "\n", " tail"])


def sample(rng):
    """Return (text, category)."""
    if rng.random() < 0.003:
        # A few long inputs with many lines, to exercise long-line handling.
        parts = []
        for _ in range(rng.randint(40, 120)):
            parts.append(prose(rng, rng.randint(5, 30)))
            parts.append("\n")
        return "".join(parts), "long"
    r = rng.random()
    if r < 0.20:
        return prose(rng, rng.randint(1, 60)), "prose"
    if r < 0.32:
        return code(rng), "code"
    if r < 0.42:
        obj = json_value(rng)
        indent = rng.choice([None, 2])
        return json.dumps(obj, ensure_ascii=rng.random() < 0.5, indent=indent), "json"
    if r < 0.52:
        return cjk(rng), "cjk"
    if r < 0.59:
        return emoji(rng), "emoji"
    if r < 0.67:
        return prose(rng, rng.randint(2, 40)).replace(" ", "\r\n", 3), "crlf"
    if r < 0.80:
        return unicode_edge(rng), "unicode"
    if r < 0.88:
        return special_like(rng), "near-special"
    if r < 0.93:
        return special_exact(rng), "special"
    return rng.choice(["", " ", "\n", "\t\n", "   ", "\r\n"]), "blank"


def encode_entry(text, cat):
    entry = {"cat": cat, "text": text}
    try:
        ids = ENC.encode(text)
    except ValueError:
        entry["err"] = "special"
        return entry
    entry["n"] = len(ids)
    entry["hash"] = hashlib.sha256(" ".join(map(str, ids)).encode("utf-8")).hexdigest()[:16]
    return entry


def main():
    rng = random.Random(SEED)
    rows = []
    # Fixed edge cases first so they are always present.
    for text, cat in [("", "blank"), ("<|endoftext|>", "special"), ("<|fim_prefix|>", "special"),
                      ("<|endofprompt|>", "special"), ("<|fim_suffix|>", "special"), ("<|fim_middle|>", "special"),
                      ("a<|endoftext|>b", "special"), ("<|endoftext|", "near-special"),
                      ("xé\U0001F600", "unicode"), ("‍", "unicode")]:
        rows.append(encode_entry(text, cat))
    while len(rows) < TOTAL:
        text, cat = sample(rng)
        rows.append(encode_entry(text, cat))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("# cl100k_base oracle, tiktoken %s. Generated by server/parity/gen_pytok_cases.py.\n"
                % tiktoken.__version__)
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
    errs = sum(1 for r in rows if "err" in r)
    print("wrote", os.path.basename(OUT), "rows=%d special_errors=%d bytes=%d" % (len(rows), errs, os.path.getsize(OUT)))


if __name__ == "__main__":
    main()
