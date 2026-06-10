"""
论文投稿前最终验证 — APBC/BIBM 标准
"""
import re

with open('FoldPathLLM_v4.tex', 'r', encoding='utf-8') as f:
    text = f.read()

body = text.split(r'\begin{document}')[1].split(r'\end{document}')[0]
bib = text.split(r'\begin{thebibliography}')[1].split(r'\end{thebibliography}')[0]

errors = []
warnings = []

# ════════════════════════════════
# 1. Abstract
# ════════════════════════════════
abs_start = text.find(r'\begin{abstract}') + len(r'\begin{abstract}')
abs_end = text.find(r'\end{abstract}')
abstract = text[abs_start:abs_end]
clean = re.sub(r'\\(?:textbf|mathbf|text|textit|texttt|emph)\{([^}]*)\}', r'\1', abstract)
clean = re.sub(r'\\[a-zA-Z]+(\{[^}]*\})*', ' ', clean)
clean = re.sub(r'[$_{}^~&%#]', ' ', clean)
clean = re.sub(r'[{}]', '', clean)
words = [w for w in clean.split() if len(w) > 1]
print(f'[1] Abstract: {len(words)} words (target ~250)')
if len(words) > 300:
    errors.append(f'Abstract too long: {len(words)} words')
elif len(words) < 180:
    errors.append(f'Abstract too short: {len(words)} words')

# ════════════════════════════════
# 2. Cross-references
# ════════════════════════════════
refs_fig = set(re.findall(r'\\ref\{fig:(\w+)\}', text))
refs_tab = set(re.findall(r'\\ref\{tab:(\w+)\}', text))
refs_eq = set(re.findall(r'\\ref\{eq:(\w+)\}', text))
labels_fig = set(re.findall(r'\\label\{fig:(\w+)\}', text))
labels_tab = set(re.findall(r'\\label\{tab:(\w+)\}', text))
labels_eq = set(re.findall(r'\\label\{eq:(\w+)\}', text))

print(f'\n[2] Cross-references:')
print(f'    Figures: labels={sorted(labels_fig)}, refs={sorted(refs_fig)}')
print(f'    Tables:  labels={sorted(labels_tab)}, refs={sorted(refs_tab)}')
print(f'    Equations: labels={sorted(labels_eq)}, refs={sorted(refs_eq)}')

for ref in refs_fig - labels_fig:
    errors.append(f'Missing figure label: fig:{ref}')
for lab in labels_fig - refs_fig:
    warnings.append(f'Unreferenced figure: fig:{lab}')
for ref in refs_tab - labels_tab:
    errors.append(f'Missing table label: tab:{ref}')
for lab in labels_tab - refs_tab:
    if lab != 'struct':  # 'struct' is the only labeled table
        warnings.append(f'Unreferenced table: tab:{lab}')
for ref in refs_eq - labels_eq:
    errors.append(f'Missing equation label: eq:{ref}')

# ════════════════════════════════
# 3. Citations
# ════════════════════════════════
cites = re.findall(r'\\cite\{([^}]+)\}', body)
all_cited = set()
for c in cites:
    for k in c.split(','):
        all_cited.add(k.strip())

bib_keys = set(re.findall(r'\\bibitem\{(\w+)\}', bib))
print(f'\n[3] Citations: {len(all_cited)} cited, {len(bib_keys)} bibitems')

uncited = bib_keys - all_cited
if uncited:
    errors.append(f'Uncited references: {uncited}')
missing = all_cited - bib_keys
if missing:
    errors.append(f'Missing bibitems: {missing}')

# Check for duplicate bibitems
bib_names = re.findall(r'\\bibitem\{(\w+)\}', bib)
if len(bib_names) != len(set(bib_names)):
    dupes = [n for n in bib_names if bib_names.count(n) > 1]
    errors.append(f'Duplicate bibitems: {set(dupes)}')

# ════════════════════════════════
# 4. LaTeX syntax
# ════════════════════════════════
# Check for unmatched braces
brace_depth = 0
for i, c in enumerate(text):
    if c == '{':
        brace_depth += 1
    elif c == '}':
        brace_depth -= 1
    if brace_depth < 0:
        errors.append(f'Unmatched closing brace near position {i}')
        brace_depth = 0
if brace_depth != 0:
    errors.append(f'Unmatched braces: depth={brace_depth}')

# Check for double spaces
if '  ' in body:
    warnings.append('Double spaces found in body text')

# Check for common typos
typos = [
    (r'\\textbf\{RITA', 'Check RITA formatting'),
    (r'\$\\sim\$', 'Use $\\sim$ not \\$\\sim\\$'),
]
for pattern, msg in typos:
    if re.search(pattern, text):
        warnings.append(msg)

# Check \ref{} before \label{}
refs_positions = [(m.start(), m.group()) for m in re.finditer(r'\\ref\{[^}]+\}', text)]
labels_positions = {m.group(): m.start() for m in re.finditer(r'\\label\{[^}]+\}', text)}
for pos, ref in refs_positions:
    lab_key = ref.replace('\\ref', '\\label')
    if lab_key in labels_positions and pos < labels_positions[lab_key]:
        warnings.append(f'{ref} appears before its label')

# ════════════════════════════════
# 5. Content checks
# ════════════════════════════════
tables = len(re.findall(r'\\begin\{table\}', body))
figures = len(re.findall(r'\\begin\{figure\}', body))
print(f'\n[4] Content: {tables} tables, {figures} figures')

# Check section headings
sections = re.findall(r'\\section\{([^}]+)\}', body)
print(f'    Sections: {len(sections)}')
for s in sections:
    print(f'      - {s}')

subsections = re.findall(r'\\subsection\{([^}]+)\}', body)
print(f'    Subsections: {len(subsections)}')
for s in subsections:
    print(f'      - {s}')

# Check that all \begin{} have matching \end{}
begins = re.findall(r'\\begin\{(\w+)\}', body)
ends = re.findall(r'\\end\{(\w+)\}', body)
from collections import Counter
begin_counts = Counter(begins)
end_counts = Counter(ends)
for env in set(list(begin_counts.keys()) + list(end_counts.keys())):
    bc = begin_counts.get(env, 0)
    ec = end_counts.get(env, 0)
    if bc != ec:
        errors.append(f'Environment mismatch: {env} (begin={bc}, end={ec})')

# ════════════════════════════════
# 6. Specific MDPI/APBC checks
# ════════════════════════════════
print(f'\n[5] Format checks:')
# Check for numbered sections (MDPI style)
if re.search(r'\\section\{1\.', body):
    warnings.append('Sections have manual numbers (may conflict with auto-numbering)')

# Check for H-float overuse
h_floats = len(re.findall(r'\\begin\{table\}\[H\]', body))
h_figures = len(re.findall(r'\\begin\{figure\}\[H\]', body))
print(f'    [H] floats: {h_floats} tables, {h_figures} figures')
if h_floats + h_figures > 5:
    warnings.append('Many [H] floats — consider allowing floating for better page layout')

# Check fontsize
fontsize = re.search(r'\\documentclass\[(\d+pt)', text)
if fontsize:
    print(f'    Base font size: {fontsize.group(1)}')

# Check abstract structure
if '(1) Background:' in abstract and '(4) Conclusions:' in abstract:
    print(f'    Abstract structure: MDPI-style numbered ✓')
else:
    warnings.append('Abstract may not follow MDPI numbered structure')

# ════════════════════════════════
# Summary
# ════════════════════════════════
print(f'\n{"="*60}')
if errors:
    print(f'  ERRORS ({len(errors)}):')
    for e in errors:
        print(f'    ✗ {e}')
else:
    print(f'  No errors found')

if warnings:
    print(f'  WARNINGS ({len(warnings)}):')
    for w in warnings:
        print(f'    ⚠ {w}')
else:
    print(f'  No warnings')

if errors:
    print(f'\n  STATUS: FIX ERRORS BEFORE SUBMISSION')
elif warnings:
    print(f'\n  STATUS: READY (minor warnings — review before submission)')
else:
    print(f'\n  STATUS: READY FOR SUBMISSION ✓')
print(f'{"="*60}')
