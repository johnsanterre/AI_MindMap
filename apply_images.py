#!/usr/bin/env python3
"""Insert <img> tags for newly generated asset SVGs into passages.

Reads a JSON array [{name, stem, alt}] and inserts the site's standard
image embed directly below each passage's `! Title` line (or after the
Header include when there is no title line). Skips passages that
already reference an image. Backs up index.html first.

Usage: python3 apply_images.py images.json
"""

import html as html_mod
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parent
SOURCE = ROOT / "index.html"
HEADER_MARK = "&lt;&lt;include &quot;Header&quot;&gt;&gt;"


def encode(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace('"', "&quot;").replace("'", "&#39;"))


def main(path):
    items = {i["name"]: i for i in json.load(open(path))}
    text = SOURCE.read_text(encoding="utf-8")
    shutil.copyfile(SOURCE, str(SOURCE) + ".pre-images.bak")
    applied = []

    def patch(m):
        attrs, body = m.group(1), m.group(2)
        nm = re.search(r'name="([^"]*)"', attrs)
        if not nm:
            return m.group(0)
        name = html_mod.unescape(nm.group(1))
        if name not in items:
            return m.group(0)
        if "assets/" in html_mod.unescape(body):
            return m.group(0)  # already has an image
        it = items[name]
        svg = ROOT / "assets" / f"{it['stem']}.svg"
        if not svg.exists():
            return m.group(0)
        img = encode(
            f'<img src="assets/{it["stem"]}.svg" alt="{it["alt"]}" '
            f'style="width:100%;max-width:620px;display:block;'
            f'margin:16px auto 24px auto;">'
        )
        # Prefer inserting after the `! Title` line.
        tm = re.search(r'^(!\s[^\n]*\n)', body, re.M)
        if tm:
            new_body = body[:tm.end()] + "\n" + img + "\n" + body[tm.end():]
        elif HEADER_MARK in body:
            new_body = body.replace(HEADER_MARK, HEADER_MARK + "\n\n" + img, 1)
        else:
            new_body = img + "\n\n" + body
        applied.append(name)
        return f"<tw-passagedata{attrs}>{new_body}</tw-passagedata>"

    new_text = re.sub(r'<tw-passagedata([^>]*)>(.*?)</tw-passagedata>',
                      patch, text, flags=re.S)
    SOURCE.write_text(new_text, encoding="utf-8")
    print(f"images embedded in {len(applied)} passages:", applied)


if __name__ == "__main__":
    main(sys.argv[1])
