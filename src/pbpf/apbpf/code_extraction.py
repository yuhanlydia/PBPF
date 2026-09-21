"""Select the requested implementation from public model completion text."""
import ast
import re


def extract_solution(text):
    blocks = re.findall(r'```(?:python3?|py)?\s*\n(.*?)```', text, flags=re.S | re.I)
    if not blocks:
        return text.strip()
    imports = []
    solution = None
    for block in blocks:
        try:
            tree = ast.parse(block)
        except SyntaxError:
            continue
        if tree.body and all(isinstance(node, (ast.Import, ast.ImportFrom)) for node in tree.body):
            if block.strip() not in imports:
                imports.append(block.strip())
        if solution is None and any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                                    and node.name == 'solution' for node in tree.body):
            solution = block.strip()
    if solution is not None:
        return '\n\n'.join([*imports, solution])
    # A truncated implementation remains an implementation failure. Do not
    # replace it with a longer usage example just because parsing failed.
    for block in blocks:
        if re.search(r'^\s*(?:async\s+)?def\s+solution\s*\(', block, re.M):
            return block.strip()
    return max(blocks, key=len).strip()
