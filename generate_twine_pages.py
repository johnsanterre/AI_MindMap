#!/usr/bin/env python3
"""
Generate full Twine ML topic pages using the Claude API,
then insert them into ml_mindmap_published.html.
"""

import re
import os
import html
import anthropic

client = anthropic.Anthropic()

# Map: passage name in HTML -> (svg filename stem, display name)
TOPICS = {
    "Perceptron":                      ("perceptron",         "The Perceptron"),
    "Q Learning":                      ("q_learning",         "Q-Learning"),
    "Elastic Net":                     ("elastic_net",        "Elastic Net Regularization"),
    "Multi Armed Bandit":              ("multi_armed_bandit", "Multi-Armed Bandits"),
    "Hidden Markov Machine":           ("hidden_markov",      "Hidden Markov Models"),
    "Spectral Clustering":             ("spectral_clustering","Spectral Clustering"),
    "Non-negative matrix factorization": ("nmf",             "Non-negative Matrix Factorization (NMF)"),
    "one shot learning":               ("one_shot_learning",  "One-Shot Learning"),
    "Yolo":                            ("yolo",               "YOLO: You Only Look Once"),
    "Causal Modeling":                 ("causal_modeling",    "Causal Modeling"),
}

STYLE_PROMPT = """
You are writing a page for a Twine/SugarCube interactive ML reference site. Follow the PCA example EXACTLY in structure and style.

STRUCTURE (use this order, every section required):

<<include "Header">>

! {FULL_TITLE}

<img src="assets/{SVG_STEM}.svg" alt="[descriptive alt text for the diagram]" style="width:100%;max-width:620px;display:block;margin:16px auto 24px auto;">

[Opening paragraph: 3–5 sentences. Concrete real-world analogy. No jargon yet. Hook the reader.]

---

!! The Core Insight
[Intuitive, geometric, or narrative explanation. Build the mental model before any math. 3–5 paragraphs.]

---

!! [Second conceptual section — give it a topic-specific descriptive title]
[Deepen the intuition. Explain *why* something works, not just that it does. 2–4 paragraphs.]

---

!! What the Math Is Actually Doing
[Introduce notation and equations. Explain each symbol in plain terms alongside the math. Use $inline$ and $$display$$ LaTeX. 3–5 paragraphs.]

---

!! [Practical section — e.g. "Choosing Hyperparameters", "Reading the Output", "When to Use This"]
[How to actually use this in practice. Rules of thumb, tradeoffs, common thresholds. 2–4 paragraphs.]

---

!! When {SHORT_TITLE} Breaks Down
[Exactly 3 specific failure modes. For each: what goes wrong, why, what to use instead.]

---

!! Technical Details
[Dense reference-style section. Cover: algorithm steps, complexity, 2–3 notable variants/extensions, key theorems or results. Use ''bold'' for term introductions.]

---

!! Quiz

[Write 7–10 numbered multiple-choice questions. Mix conceptual ("explain why") and technical ("walk through the algorithm"). Each question has exactly 4 options (A–D), one correct. Use this EXACT widget pattern:]

''N. Question text?''
<<button "A) First option">><<replace "#{ID_PREFIX}-qN">>'✗ Wrong. Brief explanation of why A is incorrect.'<</replace>><<script>>typesetMath();<</script>><</button>>
<<button "B) Second option">><<replace "#{ID_PREFIX}-qN">>'✓ Correct! Full explanation of why B is right (2–4 sentences).'<</replace>><<script>>typesetMath();<</script>><</button>>
<<button "C) Third option">><<replace "#{ID_PREFIX}-qN">>'✗ Wrong. Brief explanation of why C is incorrect.'<</replace>><<script>>typesetMath();<</script>><</button>>
<<button "D) Fourth option">><<replace "#{ID_PREFIX}-qN">>'✗ Wrong. Brief explanation of why D is incorrect.'<</replace>><<script>>typesetMath();<</script>><</button>>
<span id="{ID_PREFIX}-qN"></span>

[The correct answer should vary across questions. Distractors should be plausible, not obviously wrong.]

<<include "Footer">>

---

WRITING STYLE RULES (strictly follow all):
- Analogy first, math second. Every concept gets a concrete real-world framing before equations.
- Explain the "why" — what problem does this solve, why does this property matter.
- Mix short punchy sentences with longer explanations. Use "This is the whole idea." style summary sentences after key insights.
- American spelling: visualization, standardize, centered, behavior, optimize.
- ''double apostrophes'' for bold key terms when first introduced.
- //double slashes// for italics/emphasis.
- $...$ for inline LaTeX, $$...$$ for display equations (on their own line).
- [[Link Text|Passage Name]] for internal Twine links when referencing related topics.
- Technical Details section is dense and compressed — a quick reference, not a tutorial.
- Quiz questions are multiple choice (A–D). Wrong answers get a brief explanation of why they're wrong. Correct answer gets a full 2–4 sentence explanation. Distractors must be plausible.
- Do NOT add any explanation outside the passage. Output the raw passage text only.
- {ID_PREFIX} must be a short lowercase identifier with no spaces (e.g. "lr" for Linear Regression, "rf" for Random Forest).
""".strip()


def build_prompt(passage_name, svg_stem, display_name):
    short_title = display_name.split("(")[0].strip()
    id_prefix = svg_stem.replace("_", "")
    return STYLE_PROMPT.format(
        FULL_TITLE=display_name,
        SVG_STEM=svg_stem,
        SHORT_TITLE=short_title,
        ID_PREFIX=id_prefix,
    )


def generate_passage(passage_name, svg_stem, display_name):
    prompt = build_prompt(passage_name, svg_stem, display_name)
    print(f"  Generating: {passage_name}...")
    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=4096,
        messages=[{"role": "user", "content": f"Write the Twine passage for: {display_name}\n\n{prompt}"}],
    )
    return message.content[0].text.strip()


def html_encode(text):
    """Encode text for embedding inside a tw-passagedata element."""
    return (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def update_html(html_path, passage_name, new_content):
    with open(html_path) as f:
        raw = f.read()

    # Find the tw-passagedata element for this passage
    pattern = re.compile(
        r'(<tw-passagedata[^>]*name="' + re.escape(passage_name) + r'"[^>]*>)(.*?)(</tw-passagedata>)',
        re.DOTALL,
    )
    match = pattern.search(raw)
    if not match:
        print(f"  WARNING: passage '{passage_name}' not found in HTML — skipping.")
        return

    encoded = html_encode(new_content)
    new_raw = raw[:match.start(2)] + encoded + raw[match.end(2):]

    with open(html_path, "w") as f:
        f.write(new_raw)

    print(f"  Updated: {passage_name}")


def main():
    html_path = os.path.join(os.path.dirname(__file__), "ml_mindmap_published.html")

    for passage_name, (svg_stem, display_name) in TOPICS.items():
        try:
            content = generate_passage(passage_name, svg_stem, display_name)
            update_html(html_path, passage_name, content)
        except Exception as e:
            print(f"  ERROR on {passage_name}: {e}")

    print("\nDone. All passages updated.")


if __name__ == "__main__":
    main()
