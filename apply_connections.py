#!/usr/bin/env python3
"""Insert 'Connections' sections into unlinked passages.

Reads connection suggestions (name -> [{target, why}]) produced by the
batch analysis, validates every target against real passage names, and
appends a Connections section to each card's passage in index.html:

    !! Connections

    [[Target]] — why sentence.

Inserted before the Footer include when present (so it sits above the
footer nav), else appended. Text is HTML-encoded to match Twine's
storage format. A .bak copy of index.html is written first.

Usage: python3 apply_connections.py out0.json out1.json ...
"""

import html as html_mod
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parent
SOURCE = ROOT / "index.html"

FOOTER_MARK = "&lt;&lt;include &quot;Footer&quot;&gt;&gt;"


def encode(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace('"', "&quot;").replace("'", "&#39;"))


def main(paths):
    text = SOURCE.read_text(encoding="utf-8")
    names = {html_mod.unescape(m.group(1))
             for m in re.finditer(r'<tw-passagedata[^>]*name="([^"]*)"', text)}

    suggestions = {}
    for p in paths:
        for item in json.load(open(p)):
            conns = [c for c in item.get("connections", [])
                     if c.get("target") in names and c.get("why")]
            if conns:
                suggestions[item["name"]] = conns[:3]

    shutil.copyfile(SOURCE, str(SOURCE) + ".pre-connections.bak")

    applied = 0
    skipped_missing = []

    def patch(m):
        nonlocal applied
        attrs, body = m.group(1), m.group(2)
        nm = re.search(r'name="([^"]*)"', attrs)
        if not nm:
            return m.group(0)
        name = html_mod.unescape(nm.group(1))
        if name not in suggestions:
            return m.group(0)
        if "!! Connections" in html_mod.unescape(body):
            return m.group(0)  # already has one
        lines = ["", "", "!! Connections", ""]
        for c in suggestions[name]:
            lines.append(f"[[{c['target']}]] — {c['why']}")
            lines.append("")
        block = encode("\n".join(lines).rstrip() + "\n")
        if FOOTER_MARK in body:
            new_body = body.replace(FOOTER_MARK, block + "\n" + FOOTER_MARK, 1)
        else:
            new_body = body + block
        applied += 1
        return f"<tw-passagedata{attrs}>{new_body}</tw-passagedata>"

    new_text = re.sub(r'<tw-passagedata([^>]*)>(.*?)</tw-passagedata>',
                      patch, text, flags=re.S)
    SOURCE.write_text(new_text, encoding="utf-8")

    for name in suggestions:
        if name not in names:
            skipped_missing.append(name)
    print(f"applied Connections to {applied} passages "
          f"({len(suggestions)} suggested; backup at index.html.pre-connections.bak)")
    if skipped_missing:
        print("unknown passage names skipped:", skipped_missing[:5])


if __name__ == "__main__":
    main(sys.argv[1:])
