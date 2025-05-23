from typing import List, Dict, Optional, Sequence, Tuple
import numpy as np
import random
# import torch
import os
import re

def softmax(x: np.ndarray) -> np.ndarray:
    '''
    Compute the softmax of vector x.
    '''
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def seed_all(seed=1029, others: Optional[list] = None):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def random_select_and_remove(lst: List):
    if not lst:
        return None
    index = random.randrange(len(lst))
    lst[index], lst[-1] = lst[-1], lst[index]
    return lst.pop()

def simplify_equation(equation: str) -> str:
    # Remove extra whitespace
    expr = equation.strip()
    steps = [expr]

    def is_number(s: str) -> bool:
        try:
            float(s)
            return True
        except ValueError:
            return False

    # Regex patterns for operations
    paren_pattern = re.compile(r'\([^()]+\)')  # innermost parentheses
    mult_pattern = re.compile(r'(?P<a>\-?\d+)\s*\*\s*(?P<b>\-?\d+)')
    add_sub_pattern = re.compile(r'(?P<a>\-?\d+)\s*(?P<op>[+\-])\s*(?P<b>\-?\d+)')

    while not is_number(expr):
        # First, handle parentheses
        paren_match = paren_pattern.search(expr)
        if paren_match:
            paren_str = paren_match.group(0)      # e.g., "(1 + 2 + 3)"
            inner_expr = paren_str[1:-1]            # e.g., "1 + 2 + 3"
            # Count addition/subtraction operators in the inner expression
            ops = re.findall(r'[+\-]', inner_expr)
            # If more than one operator, do a partial simplification (step by step)
            if len(ops) > 1:
                sub_match = add_sub_pattern.search(inner_expr)
                if sub_match:
                    a = sub_match.group('a')
                    op = sub_match.group('op')
                    b = sub_match.group('b')
                    partial_result = str(int(eval(f"{a}{op}{b}")))
                    # Replace the leftmost operation within the parentheses with its result
                    new_inner = inner_expr[:sub_match.start()] + partial_result + inner_expr[sub_match.end():]
                    new_paren = f"({new_inner})"
                    expr = expr[:paren_match.start()] + new_paren + expr[paren_match.end():]
                    steps.append(expr)
                    continue
            else:
                # When only one operator exists inside, evaluate the whole parenthesis
                inner_value = str(int(eval(inner_expr)))
                expr = expr[:paren_match.start()] + inner_value + expr[paren_match.end():]
                steps.append(expr)
                continue

        # Next, handle multiplication outside of parentheses
        mult_match = mult_pattern.search(expr)
        if mult_match:
            a = mult_match.group('a')
            b = mult_match.group('b')
            result = str(int(eval(f"{a}*{b}")))
            expr = expr[:mult_match.start()] + result + expr[mult_match.end():]
            steps.append(expr)
            continue

        # Finally, handle addition/subtraction outside of parentheses.
        add_sub_match = add_sub_pattern.search(expr)
        if add_sub_match:
            a = add_sub_match.group('a')
            op = add_sub_match.group('op')
            b = add_sub_match.group('b')
            result = str(int(eval(f"{a}{op}{b}")))
            expr = expr[:add_sub_match.start()] + result + expr[add_sub_match.end():]
            steps.append(expr)
            continue

        # Break to avoid an infinite loop if nothing matches.
        break

    return steps


def _strip_markers(text: str) -> Tuple[str, List[bool]]:
    """
    Remove << … >> markers.

    Returns
    -------
    cleaned_text : str
        The sentence with markers removed.
    char_mask : List[bool]
        Per-character mask (True ⇔ the character originates from inside a marker).
    """
    cleaned_chars: List[str] = []
    char_mask: List[bool] = []

    i, inside = 0, False
    while i < len(text):
        two = text[i : i + 2]
        if two == "<<" and not inside:
            inside = True
            i += 2
            continue
        if two == ">>" and inside:
            inside = False
            i += 2
            continue

        cleaned_chars.append(text[i])
        char_mask.append(inside)
        i += 1

    return "".join(cleaned_chars), char_mask


def tokenize_with_content_mask(
    text: str,
    tokenizer,
) -> Tuple[str, List[bool]]:
    """
    Parameters
    ----------
    text : str
        Original sentence containing << … >> markers.
    tokenizer : PreTrainedTokenizerBase
        Any **fast** Hugging Face tokenizer.

    Returns
    -------
    cleaned_text : str
    tokens : List[str]
        Tokens produced by `tokenizer`.
    token_mask : List[bool]
        True ⇔ corresponding token (even partially) belongs to content.
    """
    cleaned_text, char_mask = _strip_markers(text)

    enc = tokenizer(
        cleaned_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    # tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"])
    offsets = enc["offset_mapping"]

    # token is content iff any character in its span is content
    token_mask = [any(char_mask[s:e]) for (s, e) in offsets]

    return cleaned_text, token_mask