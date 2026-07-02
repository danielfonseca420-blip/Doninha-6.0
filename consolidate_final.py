#!/usr/bin/env python3

import re
from pathlib import Path

FILES = [
    "layer_titles.py", "config_loader.py", "chat_session.py", "paraconsistent_rules.py",
    "custom_tokenizer.py", "custom_lm_model.py", "neural_truth_model.py", "knowledge_base.py",
    "l4_russell_equivalence.py", "l4_chain_verification.py", "corpus_utils.py", "l1_concept_table.py",
    "l5_generation.py", "rag_hybrid_context_injection.py", "l2_kantian_judgments.py",
    "l3_paraconsistent.py", "metrics.py", "l4_synthesis.py", "l1_l2_rag_integration.py",
    "syllogism_module.py", "l6_final_response.py", "l7_final_text.py", "agente_sintese_final.py",
    "pipeline.py", "api.py",
]

INTERNAL = {"layer_titles", "config_loader", "chat_session", "paraconsistent_rules", 
            "custom_tokenizer", "custom_lm_model", "neural_truth_model", "knowledge_base", 
            "l4_russell_equivalence", "l4_chain_verification", "corpus_utils", "l1_concept_table", 
            "l5_generation", "rag_hybrid_context_injection", "l2_kantian_judgments", "l3_paraconsistent", 
            "metrics", "l4_synthesis", "l1_l2_rag_integration", "syllogism_module", 
            "l6_final_response", "l7_final_text", "agente_sintese_final", "pipeline", "api"}

ROOT = Path(__file__).parent

output = ['# Doninha IA Standalone\n', 'from __future__ import annotations\n\n']
imports = set()

# Collect imports first
for fname in FILES:
    fpath = ROOT / fname
    if not fpath.exists():
        continue
    
    with open(fpath, 'r', encoding='ascii', errors='ignore') as f:
        for line in f:
            s = line.strip()
            if s.startswith('import ') or s.startswith('from '):
                skip = False
                for mod in INTERNAL:
                    if mod in s:
                        skip = True
                        break
                if not skip and s not in imports:
                    imports.add(s)

# Sort and add imports
stdlib = sorted([x for x in imports if any(y in x for y in ['os', 'sys', 're', 'json', 'time', 'math', 'typing', 'logging'])])
other = sorted([x for x in imports if x not in stdlib])

for imp in stdlib:
    output.append(imp + '\n')
if other:
    output.append('\n')
    for imp in other:
        output.append(imp + '\n')

output.append('\n')

# Process each file
for fname in FILES:
    fpath = ROOT / fname
    if not fpath.exists():
        continue
    
    with open(fpath, 'r', encoding='ascii', errors='ignore') as f:
        lines = f.readlines()
    
    # Find where code starts (after all imports)
    start_idx = 0
    in_docstring = False
    for i, line in enumerate(lines):
        s = line.strip()
        
        # Track docstrings
        if '"""' in s or "'''" in s:
            in_docstring = not in_docstring
        
        # Skip imports and __future__ declarations
        if s.startswith('from __future__'):
            continue
        
        if (s.startswith('import ') or s.startswith('from ')) and not in_docstring:
            # Check if internal
            skip = False
            for mod in INTERNAL:
                if mod in s:
                    skip = True
                    break
            if skip:
                continue
        
        # Find first non-import line
        if s and not s.startswith(('import ', 'from ', 'from __future__', '#')) and not s.startswith('"""') and not s.startswith("'''"):
            if i > 0 and (lines[i-1].strip().startswith('"""') or lines[i-1].strip().startswith("'''")):
                # Previous line was start of docstring, so include this
                start_idx = i - 1
            else:
                start_idx = i
            break
    
    # Collect body
    body_lines = []
    for line in lines[start_idx:]:
        s = line.strip()
        # Skip from __future__ imports inside modules
        if s.startswith('from __future__'):
            continue
        body_lines.append(line)
    
    body = ''.join(body_lines).strip()
    
    if body:
        output.append(f'\n# ========== {fname} ==========\n')
        output.append(body)

# Write
with open(ROOT / 'doninha_standalone.py', 'w', encoding='ascii') as f:
    f.write(''.join(output))

print('OK')
