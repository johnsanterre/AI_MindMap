#!/usr/bin/env python3
"""Insert new cards (from decoded .md files) as Twine passages.

Each input file: first line `TITLE: <passage name>`, rest is the
decoded SugarCube content. Inserts <tw-passagedata> elements with
fresh pids before </tw-storydata>, HTML-encoded to match Twine's
storage. Positions are spread on a fresh row (the map viewer computes
its own layout anyway; Twine's editor just needs non-overlapping
coordinates). Skips titles that already exist. Backs up first.

Usage: python3 insert_new_cards.py <dir-with-md-files>
"""

import html as html_mod
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parent
SOURCE = ROOT / "index.html"


def encode(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace('"', "&quot;").replace("'", "&#39;"))


def main(md_dir):
    text = SOURCE.read_text(encoding="utf-8")
    existing = {html_mod.unescape(m.group(1))
                for m in re.finditer(r'<tw-passagedata[^>]*name="([^"]*)"', text)}
    max_pid = max(int(m.group(1))
                  for m in re.finditer(r'pid="(\d+)"', text))
    max_y = max(float(m.group(1).split(",")[1])
                for m in re.finditer(r'<tw-passagedata[^>]*position="([\d.]+,[\d.]+)"', text))

    shutil.copyfile(SOURCE, str(SOURCE) + ".pre-newcards.bak")

    blocks = []
    added, skipped = [], []
    x, y = 100, max_y + 300
    for md in sorted(Path(md_dir).glob("*.md")):
        raw = md.read_text(encoding="utf-8")
        first, _, body = raw.partition("\n")
        if not first.startswith("TITLE:"):
            skipped.append((md.name, "no TITLE line")); continue
        title = first[len("TITLE:"):].strip()
        if title in existing:
            skipped.append((md.name, f"'{title}' already exists")); continue
        max_pid += 1
        blocks.append(
            f'<tw-passagedata pid="{max_pid}" name="{encode(title)}" '
            f'tags="" position="{x},{y}" size="100,100">'
            f'{encode(body.strip())}</tw-passagedata>'
        )
        added.append(title)
        x += 150
        if x > 1600:
            x, y = 100, y + 200

    if not blocks:
        print("nothing to insert;", skipped)
        return
    assert "</tw-storydata>" in text
    text = text.replace("</tw-storydata>",
                        "\n".join(blocks) + "</tw-storydata>", 1)
    SOURCE.write_text(text, encoding="utf-8")
    print(f"inserted {len(added)} passages:", added)
    if skipped:
        print("skipped:", skipped)


if __name__ == "__main__":
    main(sys.argv[1])
