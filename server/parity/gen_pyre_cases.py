#!/usr/bin/env python3
"""Generate CPython-measured expectations for server/pyre.

Run with CPython (no third-party deps):

    python3 -I server/parity/gen_pyre_cases.py

It writes two things:

  pyre/testdata/engine_sweep.txt  JSON lines: every regex the backend engines
                                  compile, run against a fixed corpus of
                                  inputs. Each line holds CPython's search
                                  groups, finditer spans, sub and split results.
  pyre/pyre_test.go               the block between the BEGIN/END CPYTHON CASES
                                  markers is replaced with hand-picked cases.

Offsets are UTF-8 byte offsets, because that is what the Go API returns. The
Go tests never run Python; they only read these committed expectations.
"""
import ast
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MAXCP = 0x110000
PYRE = os.path.join(HERE, "..", "pyre")
ENGINES = os.path.join(HERE, "..", "..", "backend", "engines")

FLAG_BITS = {"I": 1, "IGNORECASE": 1, "M": 2, "MULTILINE": 2, "S": 4, "DOTALL": 4}
FLAG_NAMES = {1: "IgnoreCase", 2: "Multiline", 4: "DotAll"}


def bytepos(s, i):
    # surrogatepass: pyjson keeps lone surrogates as WTF-8, so count them as 3 bytes.
    return len(s[:i].encode("utf-8", "surrogatepass"))


def spans_bytes(s, flat):
    return [(-1 if v < 0 else bytepos(s, v)) for v in flat]


def py_flags(bits):
    """Map our bit labels (I=1, M=2, S=4) to CPython re flag values."""
    return (re.I if bits & 1 else 0) | (re.M if bits & 2 else 0) | (re.S if bits & 4 else 0)


def cpython_result(pat, flags, s, repl="<R>", count=0, maxsplit=0, findn=None):
    rx = re.compile(pat, py_flags(flags))
    out = {}
    m = rx.search(s)
    if m is None:
        out["search"] = None
    else:
        flat = []
        for g in range(rx.groups + 1):
            a, b = m.span(g)
            flat += [a, b]
        out["search"] = spans_bytes(s, flat)
    spans = []
    for m in rx.finditer(s):
        a, b = m.span()
        spans.append([bytepos(s, a), bytepos(s, b)])
    if findn is not None:
        spans = spans[:findn] if findn >= 0 else spans
    out["all"] = spans
    out["sub"] = rx.sub(repl, s, count=count)
    if rx.groups == 0:
        out["split"] = rx.split(s, maxsplit=maxsplit)
    else:
        out["split"] = None
    return out


def flags_of_expr(node):
    """Evaluate a re-flags expression such as re.M | re.I."""
    if node is None:
        return 0
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "re":
        return FLAG_BITS[node.attr]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return flags_of_expr(node.left) | flags_of_expr(node.right)
    raise SystemExit("unsupported flags expression: " + ast.unparse(node))


def engine_patterns():
    found = {}
    for f in sorted(glob.glob(os.path.join(ENGINES, "*.py"))):
        tree = ast.parse(open(f, encoding="utf-8").read())
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "re"):
                continue
            if node.func.attr not in ("compile", "search", "sub", "split", "findall", "finditer", "match", "fullmatch"):
                continue
            if not node.args or not (isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
                continue
            pat = node.args[0].value
            flags_node = None
            if node.func.attr == "compile" and len(node.args) > 1:
                flags_node = node.args[1]
            for kw in node.keywords:
                if kw.arg == "flags":
                    flags_node = kw.value
            flags = flags_of_expr(flags_node)
            found.setdefault((pat, flags), os.path.basename(f))
    return sorted(found.keys())


# Corpus: texts shaped like the engines' real inputs, plus Unicode edge cases.
CORPUS = [
    "", "a", "x", "a\n", "\n", "\n\n\n\n", "\r\n\r\n\r\n", "abxd", "xa",
    "  hi  \n  there\t\n",
    "Certainly! Sure thing! I hope this helps!\nThank you for your patience.\n",
    "Traceback (most recent call last):\n  File \"x.py\", line 1\nValueError: boom\n",
    "stderr: boom\nexit code 2\nerror: bad\nERROR: no\n",
    "\x1b[31mred\x1b[0m\r\n\x1b]0;title\x07done\r\n",
    "[=====>   ] 45% ETA 0:00:02 downloading\n⠋⠙⠹⠸ spinner\n",
    "```python\nprint('hi')\n```\nafter", "```\ncode\n```\nafter",
    "data: {\"a\": 1}\n\ndata: [DONE]\n",
    "<system-reminder>\nhidden\n</system-reminder>\nvisible",
    "# Committing changes with git\nbla\n\n# Creating pull requests\nmore",
    "IMPORTANT: use HEREDOC for git commit\n\nnext",
    "Git Safety Protocol: never force push\n\nafter",
    "/usr/lib/x.py:12: error: 2023-01-02T03:04:05Z current date: 2024-05-06\n",
    "Today's date: 2024-01-01\nCurrent time = noon\ncc_version: 1.2\n",
    "a  b   c \t d\n\n\n\nend  \n",
    "é café naïve ß ſ K ı İ",
    "一二三 ١٢٣ ² ½ _ x",
    "x y z　w​v\x1cu",
    "AAAA" + "b" * 450 + "=",
    "data:image/png;base64," + "QUJD" * 120,
    "Line one\n\n\nLine two\n   \nLine\t\tthree", "1\n2\n3",
    "key=value; other:thing 12:34", "src/main.go:42 and lib/x.py\\y.txt",
    "FAILED test_x - pytest.ErrorX", "ok\nxyz\n",
    "Output format guidelines: be terse.\nmore\nmore2\n",
    "<permissions instructions>x</permissions instructions>tail",
    "## Tone\nfoo\n## Next\nbar", "⠀⣿⠁", "\U0001F600 emoji \U0001F600\n",
    "SELECT 1; -- 中文 é\n", "{\"k\": [1, 2, {\"z\": null}]}\n",
]


def sweep():
    rows = []
    for pat, flags in engine_patterns():
        for s in CORPUS:
            r = cpython_result(pat, flags, s)
            r.update({"pat": pat, "flags": flags, "in": s})
            rows.append(r)
    return rows


# Hand-picked cases. Each runs search, finditer, sub, split (or search only).
HAND = [
    # Lone surrogates (WTF-8 from pyjson): one code point each, not bytes.
    (r"\w", 0, "a\ud800b"), (".", 0, "\ud800"), ("[^a]", 0, "\ud800"),
    (r"\s", 0, "\ud800 "), (r"x*", 0, "\ud800"), (r"\b", 0, "\ud800a"), (r"\W+", 0, "\ud800\ud801"),
    ("(?m)^", 0, "\ud800\n\ud800"), ("a$", 0, "a\ud800"),
    # Empty-match semantics (CPython 3.7+).
    ("x*", 0, "abxd"), ("x*", 0, "xa"), ("x*", 0, "x"), ("x*", 0, ""),
    ("|x", 0, "x"), ("x|", 0, "x"), ("a*", 0, "baaa"),
    (r"\b", 0, "a b"), (r"\b", 0, ""), (r"\B", 0, "ab cd"),
    ("(?m)^", 0, "a\nb"), ("(?m)^", 0, "a\n"), ("^", 0, "a\nb"), ("^", 0, ""),
    (r"\n{2,}", 0, "a\n\nb\n\n\nc"), (r"\s*", 0, "a b"),
    # $ and \Z.
    ("c$", 0, "abc\n"), ("c$", 0, "abc\n\n"), ("c$", 0, "abc"), ("c$", 0, "c\nc"),
    (r"c\Z", 0, "abc\n"), (r"c\Z", 0, "abc"),
    ("(?m)a$", 0, "a\nb"), ("(?m)$", 0, "a\nb\n"), ("(?m)b$", 0, "a\nb\n"),
    (r"[ \t]+\n", 0, "x  \ny\t\n"),
    # Unicode \w \d \s.
    (r"\w+", 0, "café naïve"), (r"\w", 0, "́"), (r"\w", 0, "²"),
    (r"\w", 0, "一"), (r"\w", 0, "_"), (r"\w", 0, "​"),
    (r"\d+", 0, "١٢ ² ½"), (r"\d", 0, "①"),
    (r"\s+", 0, "a  　\x1c\x85b​c"), (r"\s", 0, "᠎"),
    (r"\b\w+\b", 0, "x́y z"), (r"\bfoo\b", 0, "foo_bar foo.baz"),
    (r"\bcafé\b", 0, "cafés café"), (r"\W+", 0, "ab́ cd"),
    (r"[\w.-]+", 0, "a.b-c d"), (r"[^\s]+", 0, " a b"),
    # Character classes and escapes.
    (r"[\x09\x0a\x0d\x20-\x7e]", 0, "\x01\x09a~\x7fé"),
    (r"\x1b\[[0-9;]*[A-Za-z]", 0, "\x1b[1;32mok"), (r"[\\/][\w.-]+\.\w+", 0, "x/a.py y\\b.c"),
    (r"[^:\n]+:\d+", 0, "a:1\nb:22"), (r"^[^:\n]+:\d+", 0, "a:1\nb:22"),
    (r"[\s\-\\/|.⠀-⣿]*$", 0, "  ⠀-|/\n"),
    # Lookaround.
    ("foo(?=bar)", 0, "foobaz foobar"), ("foo(?!bar)", 0, "foobar foobaz"),
    ("(?<=\\$)\\d+", 0, "$12 13"), ("(?<!x)y", 0, "xy zy"),
    (r"(?<=\n)a", 0, "\na\nb"), (r"\b(?=\w)", 0, "ab cd"),
    (r"(?<=ab|cd)x", 0, "abx cdx ex"),
    # IGNORECASE: literals, classes, and CPython's special folds.
    ("k", 1, "KK k"), ("s", 1, "Sſ s"), ("i", 1, "Iİıi"),
    ("[a-z]+", 1, "ABCKſı"), ("[^a]", 1, "aAİ"),
    ("à", 1, "Àà"), ("σ", 1, "Σσς"),
    ("ß", 1, "ẞß"), ("µ", 1, "Μμµ"),
    ("ex", 1, "EX ex Ex"), ("hello", 1, "HELLO world"),
    # Flags and inline flags.
    ("a.b", 0, "a\nb a-b"), ("a.b", 4, "a\nb a-b"), ("(?s)a.b", 0, "a\nb"),
    ("(?i)abc", 0, "xABC"), ("(?im)^abc$", 0, "x\nABC\n"),
    ("Traceback \\(most recent call last\\):", 1, "traceback (MOST RECENT call last):"),
    ("^```python\\s*\\n(.*)\\n```$", 5, "```Python\nx=1\n```"),
    # Groups and captures.
    ("(a)|(b)", 0, "b"), ("(?:(a)|b)+", 0, "ab"), ("(a)?b", 0, "b"),
    ("(a|ab)(c|bcd)(d*)", 0, "abcd"), ("(x)*", 0, "xxx"),
    ("(?P<name>[a-c]+)-(\\d)", 0, "zz abc-7"), ("(a)(?:b)(c)", 0, "xabc"),
    # Misc.
    ("[ \\t]+", 0, "a \t b"), ("(```.*?```)", 4, "a ```x``` b ```y```"),
    ("\\d{4}-\\d{2}", 0, "2024-01 1999-12"), ("a{2,3}?", 0, "aaaa"),
    ("a{,2}", 0, "aaa"), ("a{0}", 0, "aa"), ("a{2}", 0, "aaaa"),
    ("(?:)", 0, "ab"), ("", 0, "ab"),
]

# Results with explicit count / maxsplit / findall limits.
LIMITS = [
    ("x*", 0, "abxd", "-", 1, 1, 1), ("x*", 0, "abxd", "-", 2, 2, 2),
    ("a", 0, "aaaa", "b", 2, 2, 3), ("a", 0, "aaaa", "b", -1, -1, 0),
    ("\\n{2,}", 0, "a\n\nb\n\n\nc", "\n\n", 1, 1, 1),
    ("\\s*", 0, "a b", "_", 0, 0, 2),
    ("|x", 0, "x", "-", 1, 1, 1), ("|x", 0, "x", "-", 2, 2, 2),
]


def hand_rows():
    rows = []
    for pat, flags, s in HAND:
        r = cpython_result(pat, flags, s)
        r.update({"pat": pat, "flags": flags, "in": s})
        rows.append(r)
    return rows


def limit_rows():
    rows = []
    for pat, flags, s, repl, count, maxsplit, findn in LIMITS:
        rx = re.compile(pat, py_flags(flags))
        spans = []
        for m in rx.finditer(s):
            a, b = m.span()
            spans.append([bytepos(s, a), bytepos(s, b)])
        rows.append({
            "pat": pat, "flags": flags, "in": s, "repl": repl, "count": count,
            "maxsplit": maxsplit, "findn": findn,
            "sub": rx.sub(repl, s, count=count),
            "split": rx.split(s, maxsplit=maxsplit) if rx.groups == 0 else None,
            "all_limited": spans[:findn] if findn >= 0 else spans,
        })
    return rows


def go_quote(s):
    """Go string literal. Lone surrogates become their WTF-8 bytes, since a Go
    string cannot hold them as runes."""
    out = []
    for ch in s:
        o = ord(ch)
        if 0xD800 <= o <= 0xDFFF:
            out.append("".join("\\x%02x" % b for b in ch.encode("utf-8", "surrogatepass")))
        else:
            out.append(json.dumps(ch, ensure_ascii=False))
    return "\"" + "".join(x[1:-1] if x.startswith("\"") else x for x in out) + "\""


def write_sweep(rows):
    path = os.path.join(PYRE, "testdata", "engine_sweep.txt")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("# CPython %s results for every regex in backend/engines, JSON lines.\n" % sys.version.split()[0])
        f.write("# Generated by server/parity/gen_pyre_cases.py. Offsets are UTF-8 bytes.\n")
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
    return path


def flags_go(bits):
    names = [FLAG_NAMES[b] for b in (1, 2, 4) if bits & b]
    return " | ".join(names) if names else "0"


def go_span_list(v):
    if v is None:
        return "nil"
    return "[]int{" + ", ".join(str(x) for x in v) + "}"


def write_hand(rows, limits):
    lines = ["// Generated by server/parity/gen_pyre_cases.py from CPython. Do not edit.", "var cpythonCases = []pyCase{"]
    for r in rows:
        all_ = "nil" if not r["all"] else "[][2]int{" + ", ".join("{%d, %d}" % tuple(x) for x in r["all"]) + "}"
        split = "nil" if r["split"] is None else "[]string{" + ", ".join(go_quote(x) for x in r["split"]) + "}"
        lines.append("\t{pat: %s, flags: %s, in: %s, search: %s, all: %s, sub: %s, split: %s},"
                     % (go_quote(r["pat"]), flags_go(r["flags"]), go_quote(r["in"]),
                        go_span_list(r["search"]), all_, go_quote(r["sub"]), split))
    lines.append("}")
    lines.append("")
    lines.append("var cpythonLimitCases = []pyLimitCase{")
    for r in limits:
        all_ = "nil" if not r["all_limited"] else "[][2]int{" + ", ".join("{%d, %d}" % tuple(x) for x in r["all_limited"]) + "}"
        split = "nil" if r["split"] is None else "[]string{" + ", ".join(go_quote(x) for x in r["split"]) + "}"
        lines.append("\t{pat: %s, flags: %s, in: %s, repl: %s, count: %d, maxsplit: %d, findn: %d, sub: %s, split: %s, all: %s},"
                     % (go_quote(r["pat"]), flags_go(r["flags"]), go_quote(r["in"]), go_quote(r["repl"]),
                        r["count"], r["maxsplit"], r["findn"], go_quote(r["sub"]), split, all_))
    lines.append("}")
    body = "\n".join(lines) + "\n"

    path = os.path.join(PYRE, "pyre_test.go")
    text = open(path, encoding="utf-8").read()
    begin, end = "// BEGIN CPYTHON CASES", "// END CPYTHON CASES"
    i = text.index(begin) + len(begin)
    j = text.index(end)
    text = text[:i] + "\n" + body + text[j:]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return path


# IGNORECASE classes, measured over every code point (surrogates excluded).
CLASS_FOLD = [
    (r"[a-z]", 1), (r"[A-Z]", 1), (r"[^a]", 1), (r"k", 1), (r"[k]", 1),
    (r"[^\u212a]", 1), (r"[\u0100-\u017f]", 1), (r"[\u0370-\u03ff]", 1),
    (r"[\u0400-\u04ff]", 1), (r"[\u00e0-\u00ff]", 1), (r"[\u03c3-\u03c9]", 1),
    (r"[\u0130]", 1), (r"[\u0131]", 1), (r"[\u1e9e]", 1), (r"[\u00b5]", 1),
    (r"[\u0392-\u03b2]", 1), (r"\w", 1), (r"\W", 1), (r"[\w]", 1), (r"[^\w]", 1),
    (r"\d", 1), (r"[\U00010400-\U00010427]", 1), (r"\U00010400", 1),
    (r"[a-z]", 0), (r"[^a]", 0),
]


def class_fold_rows():
    cps = [c for c in range(MAXCP) if not 0xD800 <= c <= 0xDFFF]
    big = "".join(chr(c) for c in cps)
    rows = []
    for pat, bits in CLASS_FOLD:
        rx = re.compile(pat, py_flags(bits))
        hits = [cps[m.start()] for m in rx.finditer(big)]
        spans = []
        for c in hits:
            if spans and spans[-1][1] == c - 1:
                spans[-1][1] = c
            else:
                spans.append([c, c])
        rows.append({"pat": pat, "flags": bits, "spans": spans})
    return rows


def write_class_fold(rows):
    path = os.path.join(PYRE, "testdata", "ignorecase_classes.txt")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("# CPython %s: code points each class matches over every code point\n" % sys.version.split()[0])
        f.write("# (surrogates excluded), as inclusive decimal spans. Generated by\n")
        f.write("# server/parity/gen_pyre_cases.py.\n")
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
    return path


if __name__ == "__main__":
    rows = sweep()
    p1 = write_sweep(rows)
    print("wrote", os.path.basename(p1), "rows=%d" % len(rows))
    p3 = write_class_fold(class_fold_rows())
    print("wrote", os.path.basename(p3))
    hand = hand_rows()
    lim = limit_rows()
    if os.path.exists(os.path.join(PYRE, "pyre_test.go")):
        p2 = write_hand(hand, lim)
        print("updated", os.path.basename(p2), "hand=%d limits=%d" % (len(hand), len(lim)))
