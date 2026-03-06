#!/usr/bin/env python3
"""Generate index.html — a standalone table of all content pages with deep links."""

import re
import html as htmlmod
from urllib.parse import quote

HTML_PATH = 'AI_MindMap.html'
OUTPUT_PATH = 'index.html'

# System/nav passages to exclude from the table
EXCLUDE = {
    'StoryInit', 'Header', 'Footer', 'StoryBanner', 'StoryCaption',
    'StoryMenu', 'StoryTitle', 'PassageDone', 'PassageHeader', 'PassageFooter',
    'Delete',
}

with open(HTML_PATH, 'r', encoding='utf-8') as f:
    raw = f.read()

# Extract all passages with their content
passages = re.findall(
    r'<tw-passagedata[^>]*name="([^"]+)"[^>]*>(.*?)</tw-passagedata>',
    raw, re.DOTALL
)

rows = []
for name_raw, content_raw in passages:
    name = htmlmod.unescape(name_raw)
    if name in EXCLUDE:
        continue
    decoded = htmlmod.unescape(content_raw)

    # Has SVG = full content page
    has_svg = bool(re.search(r'<img src="assets/[^"]+\.svg"', decoded))

    # Extract first real sentence for a description (skip markup lines)
    desc = ''
    for line in decoded.split('\n'):
        line = line.strip()
        # Skip SugarCube directives, headers, image tags, empty lines
        if not line or line.startswith('<<') or line.startswith('!') or line.startswith('<img') or line.startswith('[['):
            continue
        # Clean up SugarCube markup
        line = re.sub(r"''([^']+)''", r'\1', line)   # bold
        line = re.sub(r'//([^/]+)//', r'\1', line)   # italic
        line = re.sub(r'\[\[[^\]]+\]\]', '', line)    # links
        line = re.sub(r'<<[^>]+>>', '', line)         # macros
        line = line.strip()
        if len(line) > 20:
            desc = line[:120] + ('…' if len(line) > 120 else '')
            break

    rows.append((name, has_svg, desc))

# Sort: full pages first (alphabetical), then stubs
full = sorted([(n, d) for n, f, d in rows if f], key=lambda x: x[0].lower())
stubs = sorted([(n, d) for n, f, d in rows if not f], key=lambda x: x[0].lower())

def make_row(name, desc, badge=''):
    link = f'AI_MindMap.html?passage={quote(name)}'
    badge_html = f'<span class="badge">{badge}</span>' if badge else ''
    return f'<tr><td><a href="{link}">{htmlmod.escape(name)}</a> {badge_html}</td><td>{htmlmod.escape(desc)}</td></tr>'

full_rows = '\n'.join(make_row(n, d) for n, d in full)
stub_rows = '\n'.join(make_row(n, d, 'stub') for n, d in stubs)

html_out = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI MindMap — Page Directory</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: Georgia, serif;
    background: #FDFAF4;
    color: #2C2C2C;
    padding: 32px 24px;
    max-width: 1100px;
    margin: 0 auto;
  }}
  h1 {{
    font-size: 2rem;
    color: #7C2530;
    margin-bottom: 8px;
  }}
  .subtitle {{
    color: #5C3D1E;
    margin-bottom: 24px;
    font-style: italic;
  }}
  .enter-btn {{
    display: inline-block;
    background: #7C2530;
    color: #FDFAF4;
    padding: 10px 24px;
    border-radius: 4px;
    text-decoration: none;
    font-size: 1rem;
    margin-bottom: 32px;
  }}
  .enter-btn:hover {{ background: #5C1A24; }}
  input[type=search] {{
    width: 100%;
    max-width: 400px;
    padding: 8px 12px;
    font-size: 1rem;
    border: 1px solid #C4A87A;
    border-radius: 4px;
    background: #FFF;
    margin-bottom: 20px;
    font-family: Georgia, serif;
  }}
  h2 {{
    font-size: 1.2rem;
    color: #2D5A8E;
    margin: 24px 0 12px;
    border-bottom: 1px solid #C4A87A;
    padding-bottom: 4px;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.95rem;
  }}
  th {{
    text-align: left;
    background: #2D5A8E;
    color: #FDFAF4;
    padding: 8px 12px;
    font-weight: normal;
    letter-spacing: 0.03em;
  }}
  td {{
    padding: 7px 12px;
    border-bottom: 1px solid #E8E0D0;
    vertical-align: top;
  }}
  tr:hover td {{ background: #F5EFE4; }}
  td:first-child {{ width: 260px; white-space: nowrap; }}
  td:last-child {{ color: #555; font-size: 0.88rem; font-style: italic; }}
  a {{ color: #7C2530; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  .badge {{
    font-size: 0.7rem;
    background: #C4A87A;
    color: #FFF;
    padding: 1px 6px;
    border-radius: 10px;
    vertical-align: middle;
    margin-left: 4px;
  }}
  .count {{ color: #888; font-size: 0.9rem; font-weight: normal; margin-left: 8px; }}
  .hidden {{ display: none; }}
</style>
</head>
<body>

<h1>AI MindMap</h1>
<p class="subtitle">Interactive ML / Math / Statistics reference — {len(full) + len(stubs)} pages</p>

<a class="enter-btn" href="AI_MindMap.html">Enter the Map →</a>

<div>
  <input type="search" id="search" placeholder="Filter pages…" oninput="filterTable()">
</div>

<h2>Content Pages <span class="count">({len(full)})</span></h2>
<table id="full-table">
  <thead><tr><th>Page</th><th>Description</th></tr></thead>
  <tbody>
{full_rows}
  </tbody>
</table>

<h2>Navigation &amp; Stub Pages <span class="count">({len(stubs)})</span></h2>
<table id="stub-table">
  <thead><tr><th>Page</th><th>Description</th></tr></thead>
  <tbody>
{stub_rows}
  </tbody>
</table>

<script>
function filterTable() {{
  const q = document.getElementById('search').value.toLowerCase();
  document.querySelectorAll('tbody tr').forEach(row => {{
    row.classList.toggle('hidden', !row.textContent.toLowerCase().includes(q));
  }});
}}
</script>

</body>
</html>
"""

with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    f.write(html_out)

print(f"Wrote {OUTPUT_PATH} ({len(full)} full pages, {len(stubs)} stubs)")
