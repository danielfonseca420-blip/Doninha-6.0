#!/usr/bin/env python3
import re

content = open('doninha_standalone.py').read()
lines = content.split('\n')
new_lines = []

i = 0
while i < len(lines):
    line = lines[i]
    
    # Check for try: without body (removed imports inside)
    if line.strip() == 'try:' and i + 1 < len(lines):
        next_line = lines[i + 1].strip()
        # If next line is not indented code, add pass
        if next_line.startswith(('except', 'finally', 'else:')) or i + 1 >= len(lines):
            new_lines.append(line)
            new_lines.append('    pass')
            i += 1
            continue
    
    # Check for except: without body
    if line.strip().startswith('except') and line.strip().endswith(':') and i + 1 < len(lines):
        next_line = lines[i + 1].strip()
        if next_line.startswith(('except', 'finally', 'else:', 'try:', '')) or i + 1 >= len(lines):
            new_lines.append(line)
            new_lines.append('    pass')
            i += 1
            continue
    
    new_lines.append(line)
    i += 1

new_content = '\n'.join(new_lines)

with open('doninha_standalone.py', 'w') as f:
    f.write(new_content)

print('Indentations corrigidas')

# Verify
import py_compile
try:
    py_compile.compile('doninha_standalone.py', doraise=True)
    print('OK: Sintaxe valida')
except Exception as e:
    print(f'ERRO: {e}')
