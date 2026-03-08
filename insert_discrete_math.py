#!/usr/bin/env python3
"""Insert 50 discrete math pages into index.html."""

import re
import html

HTML_PATH = 'index.html'

PAGES = [
    "Graph Theory", "Bipartite Graphs", "Graph Coloring", "Planar Graphs",
    "Shortest Path Algorithms", "Minimum Spanning Trees", "Maximum Flow",
    "Eulerian and Hamiltonian Paths", "Trees and Forests",
    "Cliques and Independent Sets", "Ramsey Theory", "Combinatorics",
    "Permutations and Combinations", "Pigeonhole Principle",
    "Inclusion-Exclusion Principle", "Catalan Numbers", "Stirling Numbers",
    "Partition Theory", "Propositional Logic", "Predicate Logic",
    "Satisfiability", "Turing Machines", "Halting Problem", "P vs NP",
    "Regular Languages", "Context-Free Grammars", "Lambda Calculus",
    "Modular Arithmetic", "Chinese Remainder Theorem", "RSA Cryptography",
    "Finite Fields", "Error Correcting Codes", "Primality Testing",
    "Partially Ordered Sets", "Lattice Theory", "Sorting Algorithms",
    "Hashing", "Binary Search Trees", "Balanced Trees",
    "Heaps and Priority Queues", "Union-Find", "Divide and Conquer",
    "Greedy Algorithms", "Backtracking", "Random Graphs",
    "Probabilistic Method", "Recurrence Relations", "Automata Theory",
    "Graph Isomorphism", "Network Flow",
]

with open(HTML_PATH, 'r', encoding='utf-8') as f:
    content = f.read()

pids = [int(m) for m in re.findall(r'<tw-passagedata[^>]*pid="(\d+)"', content)]
next_pid = max(pids) + 1

inserted = 0
updated = 0
failed = []

for name in PAGES:
    try:
        with open(f'passages/{name}.txt', 'r', encoding='utf-8') as f:
            raw = f.read().strip()
    except FileNotFoundError:
        failed.append(name)
        continue

    encoded = html.escape(raw, quote=False)
    escaped_name = html.escape(name, quote=True)

    pattern = re.compile(
        r'(<tw-passagedata[^>]*name="' + re.escape(escaped_name) + r'"[^>]*>)(.*?)(</tw-passagedata>)',
        re.DOTALL
    )
    m = pattern.search(content)
    if m:
        content = content[:m.start(2)] + encoded + content[m.end(2):]
        updated += 1
    else:
        px = 100 + (next_pid % 10) * 200
        py = 100 + (next_pid // 10) * 200
        tag = f'<tw-passagedata pid="{next_pid}" name="{escaped_name}" tags="" position="{px},{py}" size="100,200">{encoded}</tw-passagedata>'
        content = content.replace('</tw-storydata>', tag + '\n</tw-storydata>')
        next_pid += 1
        inserted += 1

with open(HTML_PATH, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Done: {updated} updated, {inserted} inserted, {len(failed)} failed")
if failed:
    print(f"Failed: {failed}")
