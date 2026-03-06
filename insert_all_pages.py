#!/usr/bin/env python3
"""Insert all 9 new passages into index.html."""
import re, html

with open('index.html', 'r', encoding='utf-8') as f:
    content = f.read()

def current_max_pid():
    pids = [int(x) for x in re.findall(r'pid="(\d+)"', content)]
    return max(pids)

print(f"Current max pid: {current_max_pid()}")

PAGES = [
    # (passage_name, pid, svg_filename, raw_content)
]

# ── 1. Expert Systems ────────────────────────────────────────────────────────
EXPERT_SYSTEMS = '''\
! Expert Systems
<img src="assets/expert_systems.svg" alt="Expert Systems diagram" style="max-width:100%;margin:1em 0;">

Imagine hiring a world-class doctor to write down every rule they use to diagnose disease — every "if the patient has a fever AND a rash AND recent travel, then consider X" — and then building a program that applies those rules automatically. That was the promise of expert systems: bottle up human expertise in explicit, auditable logic, and deploy it without the human. For a brief, dazzling moment in the 1970s and 1980s, it looked like it might work.

!! The Core Insight

An expert system has two components. The ''knowledge base'' is a collection of IF-THEN rules encoding domain expertise. The ''inference engine'' is the reasoning machinery that applies those rules to a new case, chaining conclusions together until it reaches a diagnosis or recommendation.

''MYCIN'' (Stanford, 1976) diagnosed bacterial blood infections using roughly 600 rules and in controlled trials matched or exceeded specialist performance. ''XCON'' (Digital Equipment Corporation, 1980) configured VAX minicomputers from customer orders, saving DEC an estimated $40 million per year. These were real systems solving real commercial problems.

!! The Knowledge Acquisition Bottleneck

The central mechanism — encoding expertise as explicit rules — turned out to be its fatal flaw. Experts know far more than they can articulate. Ask a radiologist how they spot a tumour and they give you five rules; watch them work and they apply five hundred micro-judgments they cannot verbalise. This is ''tacit knowledge'', and it resists transcription.

The process required a specialised ''knowledge engineer'' to interview experts, extract rules, encode them, test them, patch failures, and repeat. It was slow, expensive, and produced systems that were ''brittle'': they worked beautifully inside their training distribution and failed spectacularly just outside it. MYCIN could not diagnose a condition it had no rules for. It could not generalise.

Worse, the real world sends cases that fall between rules or require weighing uncertain evidence. IF-THEN rules are crisp; reality is not. Early attempts to handle uncertainty with ad-hoc ''certainty factors'' were mathematically unsound. [[Bayesian Inference]] would eventually provide the correct foundation.

Maintenance was a compounding disaster. A knowledge base of 600 rules has non-obvious interactions. Fixing one bug creates two others. The interpretability advantage evaporated as systems grew.

!! What the Math Is Actually Doing

Two inference strategies drove these systems.

''Forward chaining'' (data-driven): start from known facts, apply all rules whose conditions are satisfied, add conclusions to the fact base, repeat. Formally this is a fixpoint computation: $\text{Facts}^* = \bigcup_{n=0}^{\infty} \text{Apply}^n(\text{Facts}_0, \text{Rules})$.

''Backward chaining'' (goal-driven): start from a goal you want to prove, find rules whose conclusion matches the goal, make their conditions the new subgoals, and recurse. Prolog uses backward chaining with depth-first search and unification.

The underlying logic is ''resolution theorem proving'': to prove a statement, assume its negation and derive a contradiction via resolution steps:

$$\frac{A \lor C \quad \neg A \lor D}{C \lor D}$$

The intractability problem: general first-order logic inference is only semi-decidable. Even restricted to propositional logic, satisfiability is NP-complete. Real systems imposed arbitrary depth limits and search cutoffs — hacks that broke formal completeness guarantees.

!! The Commercial Collapse

By 1988 roughly 2,500 expert systems were in use in U.S. corporations. Then the market inverted. Specialised Lisp machines from Symbolics and LMI were undercut by workstation prices falling faster than anyone anticipated. Clients discovered that systems required permanent expert-engineer attention to stay current. Corporations began decommissioning systems faster than new ones were built.

The field transformed rather than died. [[Bayesian Inference]] and probabilistic graphical models offered mathematically coherent uncertainty handling. [[Statistical Machine Learning]] offered a radically different paradigm: instead of encoding rules, let the system infer patterns from labelled data. The knowledge acquisition bottleneck dissolved — data was the knowledge source.

!! When Expert Systems Break Down

''The knowledge acquisition bottleneck.'' Transcribing expertise into rules requires experts to fully articulate tacit knowledge they do not consciously hold. Systems have systematic blind spots where rules were never written, with no automatic signal when a gap is encountered.

''Brittleness at distribution boundaries.'' Expert systems have no generalisation mechanism. Unlike [[Statistical Machine Learning]] models that learn a continuous function, expert systems have hard binary boundaries — correct inside, undefined outside — with no graceful degradation.

''Maintenance and rule interaction complexity.'' A knowledge base of $N$ rules has potentially $O(N^2)$ pairwise interactions. Rule sets above a few hundred rules become opaque; modifying one rule triggers unforeseen consequences throughout the system.

!! Technical Details

| Property | Detail |
|---|---|
| Paradigm | Symbolic AI; knowledge representation and reasoning |
| Inference strategies | Forward chaining (data-driven), backward chaining (Prolog-style) |
| Uncertainty handling | Certainty factors (MYCIN), Dempster-Shafer theory, early Bayesian nets |
| Key languages | Prolog, Lisp, OPS5, CLIPS |
| Key systems | MYCIN (1976), DENDRAL (1969), XCON/R1 (1980) |
| Primary failure mode | Knowledge acquisition bottleneck + brittleness |
| Modern descendants | Business rule engines (Drools, CLIPS), clinical decision support |
| Successor paradigm | [[Statistical Machine Learning]], [[Bayesian Inference]], [[Neural Networks]] |

!! Quiz

@@#es-q1@@''1. What are the two main components of an expert system?''

<<button "A. Training set and optimizer">><<replace "#es-q1">>''1. What are the two main components of an expert system?''

//Incorrect. Those are components of a machine learning system.//<<script>>typesetMath();<</script>><</replace>><</button>>

<<button "B. Knowledge base and inference engine">><<replace "#es-q1">>''1. What are the two main components of an expert system?''

//Correct. The knowledge base holds IF-THEN rules; the inference engine applies them.//<<script>>typesetMath();<</script>><</replace>><</button>>

<<button "C. Encoder and decoder">><<replace "#es-q1">>''1. What are the two main components of an expert system?''

//Incorrect. Encoder-decoder is a [[Sequence-to-sequence]] architecture pattern.//<<script>>typesetMath();<</script>><</replace>><</button>>

<<button "D. Feature extractor and classifier">><<replace "#es-q1">>''1. What are the two main components of an expert system?''

//Incorrect. That describes a statistical ML pipeline, not a rule-based system.//<<script>>typesetMath();<</script>><</replace>><</button>>
@@

@@#es-q2@@''2. What is the "knowledge acquisition bottleneck"?''

<<button "A. Computers in the 1980s were too slow to run large rule sets">><<replace "#es-q2">>''2. What is the "knowledge acquisition bottleneck"?''

//Partially true but not the main bottleneck. The deeper problem was getting rules out of experts in the first place.//<<script>>typesetMath();<</script>><</replace>><</button>>

<<button "B. The difficulty of extracting and encoding expert tacit knowledge into explicit rules">><<replace "#es-q2">>''2. What is the "knowledge acquisition bottleneck"?''

//Correct. Experts cannot fully articulate their own reasoning, making rule extraction slow, expensive, and systematically incomplete.//<<script>>typesetMath();<</script>><</replace>><</button>>

<<button "C. The lack of large enough databases in the 1980s">><<replace "#es-q2">>''2. What is the "knowledge acquisition bottleneck"?''

//Incorrect. The bottleneck was about encoding human expertise, not data storage.//<<script>>typesetMath();<</script>><</replace>><</button>>

<<button "D. Memory limits preventing storage of large knowledge bases">><<replace "#es-q2">>''2. What is the "knowledge acquisition bottleneck"?''

//Incorrect. Storage was a technical challenge but not the fundamental problem.//<<script>>typesetMath();<</script>><</replace>><</button>>
@@

@@#es-q3@@''3. Forward chaining is best described as:''

<<button "A. Starting from a goal and working backward to find supporting facts">><<replace "#es-q3">>''3. Forward chaining is best described as:''

//Incorrect. That is backward chaining — the strategy used by Prolog.//<<script>>typesetMath();<</script>><</replace>><</button>>

<<button "B. Starting from known facts, applying rules, and accumulating new conclusions">><<replace "#es-q3">>''3. Forward chaining is best described as:''

//Correct. Forward chaining is data-driven: facts trigger rules, rules produce new facts, until a goal is reached.//<<script>>typesetMath();<</script>><</replace>><</button>>

<<button "C. Randomly sampling rules until a consistent answer is found">><<replace "#es-q3">>''3. Forward chaining is best described as:''

//Incorrect. Neither forward nor backward chaining is random.//<<script>>typesetMath();<</script>><</replace>><</button>>

<<button "D. Training rule weights on labeled examples">><<replace "#es-q3">>''3. Forward chaining is best described as:''

//Incorrect. Classic expert systems do not learn weights; rules are hand-coded.//<<script>>typesetMath();<</script>><</replace>><</button>>
@@

@@#es-q4@@''4. Why did the commercial expert system market collapse in the late 1980s?''

<<button "A. Expert systems were proven theoretically unsound">><<replace "#es-q4">>''4. Why did the commercial expert system market collapse?''

//Incorrect. The collapse was economic and practical, not due to a theoretical proof.//<<script>>typesetMath();<</script>><</replace>><</button>>

<<button "B. Neural networks outperformed them on every benchmark">><<replace "#es-q4">>''4. Why did the commercial expert system market collapse?''

//Incorrect. The neural network resurgence came later; the collapse was driven by maintenance costs and hardware changes.//<<script>>typesetMath();<</script>><</replace>><</button>>

<<button "C. Maintenance costs, brittleness, and the collapse of specialised Lisp machine hardware">><<replace "#es-q4">>''4. Why did the commercial expert system market collapse?''

//Correct. Workstations made Lisp machines uncompetitive, and ongoing rule maintenance costs proved unsustainable.//<<script>>typesetMath();<</script>><</replace>><</button>>

<<button "D. Governments banned their use in medical applications">><<replace "#es-q4">>''4. Why did the commercial expert system market collapse?''

//Incorrect. There were no such regulatory bans.//<<script>>typesetMath();<</script>><</replace>><</button>>
@@

@@#es-q5@@''5. Which framework eventually provided a sound replacement for ad-hoc certainty factors?''

<<button "A. Linear algebra and eigendecomposition">><<replace "#es-q5">>''5. Which framework replaced certainty factors?''

//Incorrect. Linear algebra underlies many ML methods but is not the direct replacement for certainty factors.//<<script>>typesetMath();<</script>><</replace>><</button>>

<<button "B. Bayesian probability theory and probabilistic graphical models">><<replace "#es-q5">>''5. Which framework replaced certainty factors?''

//Correct. [[Bayesian Inference]] gave a mathematically coherent way to represent and update uncertainty — something certainty factors could not provide.//<<script>>typesetMath();<</script>><</replace>><</button>>

<<button "C. Fourier analysis">><<replace "#es-q5">>''5. Which framework replaced certainty factors?''

//Incorrect. Fourier analysis is used in signal processing, not uncertainty reasoning in rule systems.//<<script>>typesetMath();<</script>><</replace>><</button>>

<<button "D. Gradient descent">><<replace "#es-q5">>''5. Which framework replaced certainty factors?''

//Incorrect. Gradient descent is an optimisation method, not a framework for handling uncertainty in rule systems.//<<script>>typesetMath();<</script>><</replace>><</button>>
@@'''

# ── 2. AI Winter ─────────────────────────────────────────────────────────────
AI_WINTER = '''\
! AI Winter
<img src="assets/ai_winter.svg" alt="AI Winter timeline" style="max-width:100%;margin:1em 0;">

Technology has always moved in cycles of breathless hype and painful reckoning. AI has done this more dramatically than almost any other field — twice. An ''AI winter'' is what happens when researchers promise things machines cannot yet do, funders believe them, and then everyone discovers the gap between claim and capability. The money stops. The press turns hostile. Talented people quietly leave for other fields. Progress does not stop, but it goes underground, unhyped and unfunded, until the next thaw.

!! The Core Insight

There have been two recognised AI winters. The ''first winter'' (roughly 1974–1980) followed the early optimism of symbolic AI — the Perceptron era and early theorem provers. The ''second winter'' (roughly 1987–1993) followed the expert system boom. Each winter was triggered by the same sequence: overpromising, a credible external critique, funding withdrawal, and a long cold period of slow progress without commercial excitement.

!! The First Winter: Lighthill and the Limits of Search

The first winter is usually dated to 1973 and the ''Lighthill Report''. The British government commissioned mathematician James Lighthill to evaluate the state of AI research. His conclusion was blunt: AI had failed to deliver on its promises, and the field's central techniques — combinatorial search and symbolic reasoning — scaled catastrophically with problem size.

The core technical problem was ''combinatorial explosion''. A chess program searching 4 moves ahead considers $b^4$ positions where $b$ is the average branching factor (~35 for chess). Ten moves ahead: $35^{10} \approx 2.8 \times 10^{15}$ positions. No heuristic search in 1973 could tame this. General game playing, natural language understanding, and robot planning all faced versions of the same problem.

DARPA cut funding sharply after reviewing the Lighthill findings and its own ALPAC report on machine translation (1966). Simultaneously, [[Perceptron]]-based neural networks had been devastated by the Minsky and Papert book (1969), which demonstrated formal limitations of single-layer networks. [[Neural Networks]] went dormant.

!! The Second Winter: Expert Systems and the Lisp Machine Bust

The second winter grew out of success — which made it harder to anticipate. [[Expert Systems]] had genuine commercial deployments. The market grew to roughly $1 billion annually by the mid-1980s. Specialised Lisp machine companies — Symbolics, LMI, Texas Instruments — were public companies with real revenues.

The collapse came on two fronts simultaneously.

The ''hardware front'': general-purpose workstations from Sun, Apollo, and DEC fell in price faster than Lisp machines could compete. By 1987, you could run a C program on a Sun-3 for a fraction of the cost of a Symbolics 3600. Symbolics filed for bankruptcy in 1993.

The ''software front'': expert system maintenance costs proved unsustainable. Knowledge bases drifted out of date. The [[Expert Systems]] that had looked like assets became liabilities. Japan's ''Fifth Generation Computing Project'' — a government-funded 10-year initiative launched in 1982 to build Prolog-based inference machines — ended in 1992 with no transformative products.

!! What the Math Is Actually Doing

Both winters were, at bottom, confrontations with ''computational intractability''. Many natural AI problems are NP-hard or worse. General propositional satisfiability (SAT) is NP-complete. First-order logic inference is only semi-decidable. Planning in general state spaces is PSPACE-complete.

$$\text{Search space grows as } O(b^d) \text{ where } b = \text{branching factor}, d = \text{depth}$$

For a branching factor of 10 and depth 20: $10^{20}$ nodes — larger than the number of atoms in a human body. No polynomial-time heuristic can reliably tame worst cases of NP-hard problems.

The ''Minsky-Papert theorem'' (1969) showed that single-layer [[Perceptron]] networks cannot represent non-linearly separable functions (the XOR problem). This was mathematically correct but limited in scope — it said nothing about multi-layer networks — yet it was widely interpreted as a proof that neural networks were fundamentally limited.

!! What Ended the Winters

Resolution of both winters came through ''statistical methods'' that reframed the problem: instead of searching a combinatorial space for a logical proof, fit a smooth function to data. [[Stochastic Gradient Descent]] on multi-layer networks handled the XOR problem. [[Statistical Machine Learning]] methods like [[Support Vector Machine]] sidestepped symbolic reasoning entirely.

The [[Backpropagation and Chain Rule]] algorithm was popularised in 1986 — during the second winter — and provided the foundation for the deep learning era. The 2012 ImageNet result (a deep [[CNN]] trained on GPUs) ended any remaining winter sentiment and triggered the current era.

!! When the "AI Winter" Framing Breaks Down

''Progress never actually stopped.'' [[Backpropagation and Chain Rule]] was developed and popularised during the second winter. Probabilistic graphical models advanced throughout. "Winter" is a funding and hype narrative, not an accurate description of research activity.

''The framing creates perverse incentives.'' Researchers who lived through a winter learn to underpromise — which is epistemically honest — but also learn that hype attracts funding. This creates pressure to oversell the next wave, making the next winter more likely.

''Not all sub-fields freeze together.'' Some communities made steady funded progress through the supposed winters; others experienced prolonged droughts. Treating "AI" as a unified field with a single seasonal calendar misrepresents how research actually operates.

!! Technical Details

| Event | Date | Key Cause |
|---|---|---|
| ALPAC Report | 1966 | Overpromising on machine translation |
| Minsky-Papert book | 1969 | Formal proof of single-layer limits |
| Lighthill Report | 1973 | Combinatorial explosion critique |
| First AI Winter | 1974–1980 | Intractability of general search |
| Expert system boom | 1980–1987 | Commercial deployments (XCON, MYCIN) |
| Lisp machine bust | 1987–1990 | Workstations undercut specialised hardware |
| Second AI Winter | 1987–1993 | ES maintenance costs + hardware bust |
| Statistical ML | 1990s | SVMs, HMMs, [[Bayesian Inference]] |
| AlexNet | 2012 | Deep [[CNN]] + GPU — current era begins |

!! Quiz

@@#aw-q1@@''1. The Lighthill Report (1973) criticised AI primarily because of:''

<<button "A. Ethical concerns about autonomous weapons">><<replace "#aw-q1">>''1. The Lighthill Report criticised AI primarily because of:''

//Incorrect. The Lighthill Report was a technical assessment, not an ethics report.//<<script>>typesetMath();<</script>><</replace>><</button>>

<<button "B. Combinatorial explosion making general search intractable as problems scaled">><<replace "#aw-q1">>''1. The Lighthill Report criticised AI primarily because of:''

//Correct. Lighthill argued that AI techniques scaled catastrophically — exponential growth in search space made real-world problems unsolvable with the methods of the era.//<<script>>typesetMath();<</script>><</replace>><</button>>

<<button "C. The discovery that neural networks could not learn any useful function">><<replace "#aw-q1">>''1. The Lighthill Report criticised AI primarily because of:''

//Incorrect. The Minsky-Papert critique of [[Perceptron]] networks was separate and came in 1969.//<<script>>typesetMath();<</script>><</replace>><</button>>

<<button "D. Evidence that Soviet AI research had surpassed Western programs">><<replace "#aw-q1">>''1. The Lighthill Report criticised AI primarily because of:''

//Incorrect. The critique was about fundamental technical limits, not geopolitical competition.//<<script>>typesetMath();<</script>><</replace>><</button>>
@@

@@#aw-q2@@''2. What primarily triggered the second AI winter (1987–1993)?''

<<button "A. A new mathematical proof that AI was impossible">><<replace "#aw-q2">>''2. What triggered the second AI winter?''

//Incorrect. No such proof exists or was published in this period.//<<script>>typesetMath();<</script>><</replace>><</button>>

<<button "B. Government regulation banning commercial AI systems">><<replace "#aw-q2">>''2. What triggered the second AI winter?''

//Incorrect. There was no regulatory ban; the collapse was market-driven.//<<script>>typesetMath();<</script>><</replace>><</button>>

<<button "C. The collapse of the Lisp machine hardware market and unsustainable expert system maintenance costs">><<replace "#aw-q2">>''2. What triggered the second AI winter?''

//Correct. Cheap workstations made specialised AI hardware uncompetitive, and expert system maintenance costs proved unsustainable — a double commercial collapse.//<<script>>typesetMath();<</script>><</replace>><</button>>

<<button "D. Neural networks being proven superior to all symbolic methods">><<replace "#aw-q2">>''2. What triggered the second AI winter?''

//Incorrect. The neural network resurgence was a cause of the recovery, not of the winter itself.//<<script>>typesetMath();<</script>><</replace>><</button>>
@@

@@#aw-q3@@''3. The Minsky-Papert theorem correctly described a limit of single-layer perceptrons, but was widely over-interpreted. What was the actual limit?''

<<button "A. Single-layer networks cannot learn any Boolean function">><<replace "#aw-q3">>''3. The Minsky-Papert theorem:''

//Incorrect. Single-layer networks can represent many Boolean functions — just not non-linearly separable ones like XOR.//<<script>>typesetMath();<</script>><</replace>><</button>>

<<button "B. Single-layer networks cannot represent non-linearly separable functions">><<replace "#aw-q3">>''3. The Minsky-Papert theorem:''

//Correct. The result was mathematically valid but said nothing about multi-layer networks. It was interpreted as a death blow to all neural networks, which it was not.//<<script>>typesetMath();<</script>><</replace>><</button>>

<<button "C. Single-layer networks cannot be trained with backpropagation">><<replace "#aw-q3">>''3. The Minsky-Papert theorem:''

//Incorrect. The theorem is about representational capacity, not training algorithms.//<<script>>typesetMath();<</script>><</replace>><</button>>

<<button "D. Single-layer networks cannot generalise beyond the training data">><<replace "#aw-q3">>''3. The Minsky-Papert theorem:''

//Incorrect. The theorem is about which functions can be represented, not about generalisation.//<<script>>typesetMath();<</script>><</replace>><</button>>
@@

@@#aw-q4@@''4. What most accurately describes what ended the AI winters?''

<<button "A. A single breakthrough paper that solved the core problems of symbolic AI">><<replace "#aw-q4">>''4. What ended the AI winters?''

//Incorrect. The recovery was a paradigm shift from symbolic to statistical methods, not a single paper.//<<script>>typesetMath();<</script>><</replace>><</button>>

<<button "B. Government investment programs that mandated AI adoption">><<replace "#aw-q4">>''4. What ended the AI winters?''

//Incorrect. There were no such mandates; the recovery was driven by technical and commercial developments.//<<script>>typesetMath();<</script>><</replace>><</button>>

<<button "C. A shift to statistical and learning-based methods combined with more data and compute">><<replace "#aw-q4">>''4. What ended the AI winters?''

//Correct. [[Statistical Machine Learning]], [[Bayesian Inference]], internet-scale data, and GPU compute collectively ended the winter framing by producing reliable, practical results.//<<script>>typesetMath();<</script>><</replace>><</button>>

<<button "D. AI researchers agreeing to make more modest public promises">><<replace "#aw-q4">>''4. What ended the AI winters?''

//Incorrect. The actual recovery came from technical progress, not a PR strategy.//<<script>>typesetMath();<</script>><</replace>><</button>>
@@

@@#aw-q5@@''5. Which is a valid criticism of the "AI winter" framing?''

<<button "A. AI winters did not actually involve funding cuts">><<replace "#aw-q5">>''5. Valid criticism of the "AI winter" framing:''

//Incorrect. Funding cuts were real and well-documented.//<<script>>typesetMath();<</script>><</replace>><</button>>

<<button "B. Progress never actually stopped — foundational advances continued quietly during both winters">><<replace "#aw-q5">>''5. Valid criticism of the "AI winter" framing:''

//Correct. [[Backpropagation and Chain Rule]] was developed and popularised during the second winter. "Winter" is a funding and hype narrative, not a complete description of research activity.//<<script>>typesetMath();<</script>><</replace>><</button>>

<<button "C. The Lighthill Report was later proven factually incorrect">><<replace "#aw-q5">>''5. Valid criticism of the "AI winter" framing:''

//Incorrect. Lighthill's core claims about combinatorial explosion were and remain technically valid.//<<script>>typesetMath();<</script>><</replace>><</button>>

<<button "D. AI winters only occurred in the United States">><<replace "#aw-q5">>''5. Valid criticism of the "AI winter" framing:''

//Incorrect. They were international — the UK Lighthill Report and Japan's Fifth Generation project failure were both outside the US.//<<script>>typesetMath();<</script>><</replace>><</button>>
@@'''

# ── 3-9: Use agent-generated content (cleaned up) ────────────────────────────
# Read from agent outputs
import pathlib

def clean(txt):
    """Remove markdown code fences, strip leading/trailing whitespace."""
    txt = re.sub(r'^```[^\n]*\n', '', txt, flags=re.MULTILINE)
    txt = re.sub(r'^```\s*$', '', txt, flags=re.MULTILINE)
    # Remove --- horizontal rules (not needed in SugarCube)
    txt = re.sub(r'\n---\n', '\n\n', txt)
    return txt.strip()

# Load agent output files
agent_outputs = {
    'dbn_rbm': '/tmp/claude-1000/-Users-john-Dropbox---AI-MindMap/tasks/acbf048e4338955c7.output',
    'alexnet_imagenet': '/tmp/claude-1000/-Users-john-Dropbox---AI-MindMap/tasks/a2f77d161cd74f6e1.output',
    'seq2seq_encdec': '/tmp/claude-1000/-Users-john-Dropbox---AI-MindMap/tasks/a8a195517f3cd9718.output',
}

def extract_passage(filepath, start_marker, end_marker=None):
    """Extract passage content between markers from agent output."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            txt = f.read()
        # Find start
        idx = txt.find(start_marker)
        if idx == -1:
            print(f"WARNING: '{start_marker}' not found in {filepath}")
            return ""
        content = txt[idx:]
        # Find end
        if end_marker:
            end_idx = content.find(end_marker)
            if end_idx != -1:
                content = content[:end_idx]
        # Remove code fence markers
        content = re.sub(r'^```[^\n]*\n?', '', content, flags=re.MULTILINE)
        content = re.sub(r'```\s*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'\n---\n', '\n\n', content)
        return content.strip()
    except Exception as e:
        print(f"ERROR reading {filepath}: {e}")
        return ""

# DBN
DBN_CONTENT = extract_passage(
    agent_outputs['dbn_rbm'],
    '! Deep Belief Networks',
    'PASSAGE 2: Restricted Boltzmann Machines'
)

# RBM
RBM_CONTENT = extract_passage(
    agent_outputs['dbn_rbm'],
    '! Restricted Boltzmann Machines',
)

# AlexNet
ALEXNET_CONTENT = extract_passage(
    agent_outputs['alexnet_imagenet'],
    '! AlexNet',
    'PASSAGE 2: ImageNet'
)

# ImageNet
IMAGENET_CONTENT = extract_passage(
    agent_outputs['alexnet_imagenet'],
    '! ImageNet',
)

# Seq2Seq
SEQ2SEQ_CONTENT = extract_passage(
    agent_outputs['seq2seq_encdec'],
    '! Sequence-to-sequence',
    'PASSAGE 2: Encoder-Decoder'
)
# Fix SVG path
SEQ2SEQ_CONTENT = SEQ2SEQ_CONTENT.replace('assets/seq2seq.svg', 'assets/sequence_to_sequence.svg')

# Encoder-Decoder
ENCDEC_CONTENT = extract_passage(
    agent_outputs['seq2seq_encdec'],
    '! Encoder-Decoder',
)

# Hardware Lottery - read directly
with open('passages/Hardware_Lottery.txt', 'r', encoding='utf-8') as f:
    HW_LOTTERY_CONTENT = f.read().strip()
# Fix dead link: "Recurrent Networks and LSTMs" → "LSTM"
HW_LOTTERY_CONTENT = HW_LOTTERY_CONTENT.replace(
    '[[Recurrent Networks and LSTMs|LSTMs]]', '[[LSTMs|LSTM]]'
)

# ── Define all pages ─────────────────────────────────────────────────────────
PAGES = [
    ("Expert Systems",              415, EXPERT_SYSTEMS),
    ("AI Winter",                   416, AI_WINTER),
    ("Deep Belief Networks",        417, DBN_CONTENT),
    ("Restricted Boltzmann Machines", 418, RBM_CONTENT),
    ("AlexNet",                     419, ALEXNET_CONTENT),
    ("ImageNet",                    420, IMAGENET_CONTENT),
    ("Sequence-to-sequence",        421, SEQ2SEQ_CONTENT),
    ("Encoder-Decoder",             422, ENCDEC_CONTENT),
    ("Hardware Lottery",            423, HW_LOTTERY_CONTENT),
]

# ── Insert all pages ─────────────────────────────────────────────────────────
with open('index.html', 'r', encoding='utf-8') as f:
    doc = f.read()

inserted = 0
for name, pid, raw in PAGES:
    if not raw:
        print(f"SKIP (empty): {name}")
        continue
    if f'name="{name}"' in doc:
        print(f"EXISTS (skip): {name}")
        continue

    # Add header/footer
    full = f'<<include "Header">>\n\n{raw}\n\n<<include "Footer">>'
    encoded = html.escape(full, quote=False)
    tag = f'\n\t\t<tw-passagedata name="{name}" pid="{pid}" tags="" position="2800,{1400 + pid*100}" size="100,100">{encoded}</tw-passagedata>'
    doc = doc.replace('</tw-storydata>', tag + '\n\t</tw-storydata>')
    print(f"✓ Inserted: {name} (pid={pid}, {len(raw)} chars)")
    inserted += 1

with open('index.html', 'w', encoding='utf-8') as f:
    f.write(doc)

print(f"\n✓ Done. {inserted} passages inserted.")
