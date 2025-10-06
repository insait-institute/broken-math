import pandas as pd
import re
def are_brackets_balanced(text):
        stack = []
        bracket_map = {')': '(', '}': '{', ']': '['}
        for char in text:
            if char in bracket_map.values(): 
                stack.append(char)
            elif char in bracket_map.keys(): 
                if not stack or bracket_map[char] != stack.pop():
                    return False
        return not stack 

def is_valid_numerical_latex(answer: str) -> bool:

    if not isinstance(answer, str) or not answer.strip():
        return False

    s = answer.strip()

    if '=' in s or '...' in s or not are_brackets_balanced(s):
        return False

    valid_tokens_regex = re.compile(r"""
        \d+(\.\d+)? 
        | \\(pi|frac|sqrt)
        | [ei]            
        | [\+\-\*/\^]     
        | [,\(\)\{\}\[\]] 
        | \s+              
    """, re.VERBOSE)

    remaining_chars = re.sub(valid_tokens_regex, '', s)
    if remaining_chars:
        return False

    return True
