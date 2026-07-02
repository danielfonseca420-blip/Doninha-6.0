#!/usr/bin/env python3
import ast
import sys

content = open('doninha_standalone.py').read()
lines = content.split('\n')

# Find all try/except blocks that are empty
new_lines = []
i = 0

while i < len(lines):
    line = lines[i]
    stripped = line.strip()
    
    # Handle try blocks
    if stripped == 'try:':
        indent = len(line) - len(line.lstrip())
        new_lines.append(line)
        
        # Check next lines
        has_body = False
        j = i + 1
        while j < len(lines):
            next_line = lines[j]
            next_stripped = next_line.strip()
            next_indent = len(next_line) - len(next_line.lstrip())
            
            # If next line has same or less indent and is control flow, body is empty
            if next_indent <= indent and (next_stripped.startswith(('except', 'finally', 'else:', 'try:', 'def ', 'class ')) or not next_stripped):
                break
            
            if next_stripped and next_indent > indent:
                has_body = True
                break
            
            j += 1
        
        if not has_body:
            new_lines.append(' ' * (indent + 4) + 'pass')
        
        i += 1
        continue
    
    # Handle except blocks
    if stripped.startswith('except') and stripped.endswith(':'):
        indent = len(line) - len(line.lstrip())
        new_lines.append(line)
        
        # Check next lines
        has_body = False
        j = i + 1
        while j < len(lines):
            next_line = lines[j]
            next_stripped = next_line.strip()
            next_indent = len(next_line) - len(next_line.lstrip())
            
            if next_indent <= indent and (next_stripped.startswith(('except', 'finally', 'else:', 'try:', 'def ', 'class ')) or not next_stripped):
                break
            
            if next_stripped and next_indent > indent:
                has_body = True
                break
            
            j += 1
        
        if not has_body:
            new_lines.append(' ' * (indent + 4) + 'pass')
        
        i += 1
        continue
    
    new_lines.append(line)
    i += 1

new_content = '\n'.join(new_lines)

with open('doninha_standalone.py', 'w') as f:
    f.write(new_content)

# Validate
import py_compile
try:
    py_compile.compile('doninha_standalone.py', doraise=True)
    print('OK: Arquivo corrigido e valido')
except SyntaxError as e:
    print(f'ERRO: {e}')
    sys.exit(1)
