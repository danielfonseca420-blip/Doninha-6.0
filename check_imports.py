#!/usr/bin/env python3
content = open('doninha_standalone.py').read()
lines = content.split('\n')

in_docstring = False
count = 0

for i, line in enumerate(lines, 1):
    if '"""' in line or "'''" in line:
        in_docstring = not in_docstring
    
    if not in_docstring:
        if any(x in line for x in ['from layer_titles', 'from config_loader', 'from l1_', 'from l2_', 'from l3_', 'from l4_', 'from l5_', 'from l6_', 'from l7_']):
            if line.strip().startswith('from'):
                count += 1
                if count <= 5:
                    print(f'Linha {i}: {line[:70]}')

print(f'\nTotal de imports internos reais: {count}')
if count == 0:
    print('OK: Nenhum import interno executavel encontrado')
