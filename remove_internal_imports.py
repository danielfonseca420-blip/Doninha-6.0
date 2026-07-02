#!/usr/bin/env python3
content = open('doninha_standalone.py').read()

# Lista de imports internos a remover
internal_patterns = [
    'from layer_titles import',
    'from l4_synthesis import',
    'from l3_paraconsistent import',
    'from l4_chain_verification import',
    'from l4_russell_equivalence import',
    'from l1_concept_table import',
    'from l2_kantian_judgments import',
    'from l5_generation import',
    'from l6_final_response import',
    'from l7_final_text import',
    'from knowledge_base import',
    'from config_loader import',
    'from chat_session import',
    'from metrics import',
    'from paraconsistent_rules import',
    'from syllogism_module import',
]

lines = content.split('\n')
new_lines = []
skip_next = 0

for i, line in enumerate(lines):
    if skip_next > 0:
        skip_next -= 1
        continue
    
    # Check if line should be removed
    remove = False
    for pattern in internal_patterns:
        if line.strip().startswith(pattern):
            remove = True
            if line.rstrip().endswith('\\'):
                skip_next = 1
            break
    
    if not remove:
        new_lines.append(line)

new_content = '\n'.join(new_lines)

with open('doninha_standalone.py', 'w') as f:
    f.write(new_content)

print('OK: Imports internos removidos')

# Verify
import py_compile
try:
    py_compile.compile('doninha_standalone.py', doraise=True)
    print('OK: Sintaxe valida')
except Exception as e:
    print(f'ERRO: {e}')
