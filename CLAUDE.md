# Twine ML Knowledge Map

Interactive ML/math/statistics reference built with Twine 2 / SugarCube 2.37.3.

## Key Files

- `index.html` — the single published Twine file containing all ~400 passages
- `page_status.md` — tracks which passages are full vs stubs
- `generate_twine_pages.py` — Python script that uses Claude API to generate passages and insert them
- `generate_index.py` — generates `page_directory.html`, a searchable table of all passages
- `assets/` — SVG diagrams, one per full content page

## Passage Architecture

Passages are stored as `<tw-passagedata name="..." pid="...">` elements inside the HTML file. Content is HTML-encoded (& → &amp;, < → &lt;, > → &gt;, " → &quot;, ' → &#39;).

**Start passage:** Home
**Included on every page:** `<<include "Header">>` at top, `<<include "Footer">>` at bottom.

### Link Types

- **Twine links** `[[Link Text|Passage Name]]` — native SugarCube links. These show as lines in the Twine editor map. Use for content cross-references between topic pages.
- **Engine.play() links** `<a href="javascript:void(0)" onclick="Engine.play('PassageName')">text</a>` — HTML/JS links invisible to Twine's map. The Footer uses this for the Home link. Use this style for navigation back to Home and hub pages to keep the Twine map uncluttered.
- **Deep links** `?passage=PassageName` — URL query param handled by StoryInit. For external entry points.

**Rule:** All links back to Home should use `Engine.play()` (via Footer), NOT `[[Home]]`, to avoid cluttering the Twine visual map.

### Navigation Hierarchy

```
Home
├── Overview (main ML topic hub)
├── Mathmatical Concepts (math hub)
├── Branches of Mathmatics (math categories)
├── Statistics
├── Core Linear Algebra
├── Deep Learning
├── Reinforcement Learning
├── Conferences / ML People / Institutes
└── Holding (staging area)
```

## Content Page Structure (all sections required)

1. `<<include "Header">>`
2. `! Title` + SVG image
3. Opening paragraph (analogy-first, no jargon)
4. `!! The Core Insight`
5. `!! [Topic-specific conceptual section]`
6. `!! What the Math Is Actually Doing` (LaTeX: `$inline$`, `$$display$$`)
7. `!! [Practical section]`
8. `!! When X Breaks Down` (exactly 3 failure modes)
9. `!! Technical Details` (dense reference)
10. `!! Quiz` (7–10 multiple choice, A–D, `<<button>><<replace>>` widget pattern)
11. `<<include "Footer">>`

## SugarCube Markup

- `''bold''` for key term introductions
- `//italic//` for emphasis
- `$...$` inline LaTeX, `$$...$$` display LaTeX
- `[[Link Text|Passage Name]]` for internal cross-references
- `<<button "text">><<replace "#id">>feedback<</replace>><<script>>typesetMath();<</script>><</button>>`

## SVG Style

- `viewBox="0 0 800 400"` (landscape) or `"0 0 600 310"`
- Cream background `#FDFAF4`, border `#D4C5A9`
- Georgia serif font
- 2–4 color palette: deep red `#7C2530`, navy `#2D5A8E`, warm brown `#5C3D1E`, tan `#C4A87A`
- Simple shapes with `<text>` labels, no raster content

## Batch Expansion Workflow

1. Grep HTML for stub passages (short content, <500 chars, not system/nav pages)
2. Verify exact passage names (many have typos — match exactly)
3. Launch parallel background agents: each creates SVG + full passage .txt file
4. Run insertion script: regex-match `<tw-passagedata name="...">`, HTML-encode content, replace
5. Delete temp `passages/` directory
6. Update `page_status.md`

## Known Issues

- **133 orphan pages** — full content but nothing links to them
- **141 dead links** — broken `[[links]]` due to case/spelling mismatches (e.g., `"BERT"` vs `"Bert"`, `"SVD"` vs `"Singular Value Decompostion"`)
- **136 leaf pages** — no outgoing content links (dead ends)
- Many passage names have typos that must be matched exactly (e.g., `"Streching"`, `"Deep Learning Librarys"`, `"emprical methods"`)
