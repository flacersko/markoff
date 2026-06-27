import argparse
import random
import math
import re
import json
import os
import sys
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

@dataclass
class StyleConfig:
    camel: bool = False
    alternate: bool = False
    xx: bool = False
    add_digits: bool = False
    prefix: str = ""
    suffix: str = ""

def load_names(filepath: str, alphabet: Optional[str] = None, keep_chars: Optional[str] = None) -> List[str]:
    names = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                name = line.strip()
                if not name:
                    continue
                proc = []
                for ch in name:
                    if ch.isalpha():
                        if alphabet == 'latin':
                            if 'a' <= ch.lower() <= 'z':
                                proc.append(ch)
                        else:
                            proc.append(ch)
                    elif keep_chars and ch in keep_chars:
                        proc.append(ch)
                name_filtered = ''.join(proc)
                if name_filtered:
                    names.append(name_filtered)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)
    return names

def build_model_counts(names: List[str], order: int, smooth: float) -> Dict[str, Counter]:
    model_counts = defaultdict(Counter)
    start = '^' * order
    for name in names:
        seq = start + name.lower() + '$'
        for i in range(len(name) + 1):
            ctx = seq[i:i + order]
            nxt = seq[i + order]
            model_counts[ctx][nxt] += 1

    if smooth > 0:
        full = set()
        for cnt in model_counts.values():
            full.update(cnt.keys())
        for ctx, cnt in model_counts.items():
            for symb in full:
                cnt[symb] = cnt.get(symb, 0) + smooth

    return model_counts

def get_next_char(model_counts: Dict[str, Counter], ctx: str, order: int, temp: float) -> Optional[str]:
    while True:
        if ctx in model_counts:
            cnt = model_counts[ctx]
            syms = list(cnt.keys())
            weights = [cnt[s] for s in syms]
            if temp != 1.0:
                inv_temp = 1.0 / temp
                weights = [w ** inv_temp for w in weights]
            tot = sum(weights)
            if tot <= 0:
                ctx = ctx[1:] if ctx else ''
                continue
            r = random.random() * tot
            c = 0.0
            for s, w in zip(syms, weights):
                c += w
                if r <= c:
                    return s
            return syms[-1] if syms else None
        if not ctx:
            break
        ctx = ctx[1:]
    return None

def has_char_repeat(text: str, limit: int) -> bool:
    if limit <= 0:
        return False
    run = 1
    for i in range(1, len(text)):
        if text[i] == text[i - 1]:
            run += 1
            if run > limit:
                return True
        else:
            run = 1
    return False

def has_ngram_loop(text: str, max_repeat: int, max_size: int) -> bool:
    if max_repeat <= 0:
        return False
    n = len(text)
    for size in range(2, max_size + 1):
        if n < size * (max_repeat + 1):
            continue
        chunk = text[-size:]
        repeats = 1

        for i in range(n - size * 2, -1, -size):
            if text[i:i + size] != chunk:
                break
            repeats += 1
            if repeats > max_repeat:
                return True
    
    return False

def beam_search(model_counts: Dict[str, Counter], order: int, start_letter: Optional[str], end_letter: Optional[str], min_len: int, max_len: int, beam_width: int, temp: float, max_repeat: int, max_ngram_repeat: int, max_ngram_size: int) -> List[str]:
    if max_len is None:
        max_len = 100
    beams = [("", 0.0)]
    completed = []
    for step in range(max_len):
        new_beams = []
        for prefix, score in beams:
            if prefix.endswith('$'):
                completed.append((prefix[:-1], score))
                continue
            ctx = ('^' * order + prefix)[-order:]
            cnt = model_counts.get(ctx, {})
            if not cnt:
                continue
            if step == 0 and start_letter:
                cnt = {s: cnt[s] for s in cnt if s.lower() == start_letter}
            if not cnt:
                continue
            for sym, w in cnt.items():
                if sym == '$':
                    if end_letter and not prefix.endswith(end_letter):
                        continue
                    if len(prefix) < min_len:
                        continue
                    completed.append((prefix, score))
                    continue
                if len(prefix) + 1 > max_len:
                    continue
                candidate = prefix + sym
                if has_char_repeat(candidate, max_repeat):
                    continue
                if has_ngram_loop(candidate, max_ngram_repeat, max_ngram_size):
                    continue
                new_score = score + (math.log(w) if w > 0 else -1000)
                new_beams.append((candidate, new_score))
        if not new_beams:
            break
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_width]
    for prefix, score in beams:
        name = prefix[:-1] if prefix.endswith('$') else prefix
        if len(name) >= min_len and (not end_letter or name.endswith(end_letter)):
            completed.append((name, score))
    seen = set()
    result = []
    completed.sort(key=lambda x: x[1], reverse=True)
    for n, _ in completed:
        if n not in seen:
            seen.add(n)
            result.append(n)
    return result

def apply_style(name: str, style: StyleConfig) -> str:
    result = name
    if style.camel:
        result = result.capitalize()
    if style.alternate:
        result = ''.join(ch.upper() if i % 2 == 0 else ch.lower() for i, ch in enumerate(result))
    if style.xx:
        result = 'xX' + result + 'Xx'
    if style.add_digits:
        digits = ''.join(str(random.randint(0, 9)) for _ in range(random.randint(1, 3)))
        result = result + digits
    if style.prefix:
        result = style.prefix + result
    if style.suffix:
        result = result + style.suffix
    return result

def save_model(model_counts: Dict[str, Counter], filepath: str) -> None:
    try:
        with open(filepath, 'w', encoding='utf-8') as jf:
            json.dump({ctx: dict(cnt) for ctx, cnt in model_counts.items()}, jf, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving model: {e}", file=sys.stderr)

def load_model(filepath: str) -> Optional[Dict[str, Counter]]:
    if not filepath or not os.path.isfile(filepath):
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as jf:
            data = json.load(jf)
        return {ctx: Counter(cnt) for ctx, cnt in data.items()}
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        return None

def validate_args(args) -> None:
    if args.order < 1:
        print("Error: --order must be >= 1", file=sys.stderr)
        sys.exit(1)
    if args.beam < 1:
        print("Error: --beam must be >= 1", file=sys.stderr)
        sys.exit(1)
    if args.temp <= 0:
        print("Error: --temp must be > 0", file=sys.stderr)
        sys.exit(1)
    if args.min_length < 0:
        print("Error: --min-length must be >= 0", file=sys.stderr)
        sys.exit(1)
    if args.max_length is not None and args.max_length < args.min_length:
        print("Error: --max-length must be >= --min-length", file=sys.stderr)
        sys.exit(1)
    if args.count < 1:
        print("Error: --count must be >= 1", file=sys.stderr)
        sys.exit(1)
    if args.max_attempts < 1:
        print("Error: --max-attempts must be >= 1", file=sys.stderr)
        sys.exit(1)
    if args.start and len(args.start) != 1:
        print("Error: --start must be a single character", file=sys.stderr)
        sys.exit(1)
    if args.end and len(args.end) != 1:
        print("Error: --end must be a single character", file=sys.stderr)
        sys.exit(1)

def check_constraints_feasible(model_counts: Dict[str, Counter], order: int, start_letter: Optional[str], end_letter: Optional[str], min_len: int, max_len: Optional[int]) -> bool:
    if not start_letter:
        return True
    ctx = '^' * order
    if ctx not in model_counts:
        return False
    cnt = model_counts[ctx]
    valid_starts = {s for s in cnt if s.lower() == start_letter and s != '$'}
    return len(valid_starts) > 0

def main():
    parser = argparse.ArgumentParser(description='Markov chain-based nickname generator with flexible constraints and styling')
    parser.add_argument('--names-file', required=True, help='File with names list (UTF-8)')
    parser.add_argument('--order', type=int, default=3, help='Markov chain order (default: 3)')
    parser.add_argument('--smooth', type=float, default=0.0, help='Laplace smoothing (default: 0.0)')
    parser.add_argument('--beam', type=int, default=1, help='Beam search width (default: 1)')
    parser.add_argument('--temp', type=float, default=1.0, help='Generation temperature (default: 1.0)')
    parser.add_argument('--max-attempts', type=int, default=1000, help='Max generation attempts (default: 1000)')
    parser.add_argument('--min-length', type=int, default=0, help='Minimum name length (default: 0)')
    parser.add_argument('--max-length', type=int, help='Maximum name length')
    parser.add_argument('--length', type=int, help='Exact name length')
    parser.add_argument('--start', type=str, help='Starting letter')
    parser.add_argument('--end', type=str, help='Ending letter')
    parser.add_argument('--forbid-pattern', type=str, help='Regex pattern of forbidden strings')
    parser.add_argument('--forbid-letters', type=str, help='Forbidden characters')
    parser.add_argument('--style-camel', action='store_true', help='CamelCase style')
    parser.add_argument('--style-alternate', action='store_true', help='aLtErNaTiNg case')
    parser.add_argument('--style-xx', action='store_true', help='xX...Xx style')
    parser.add_argument('--add-digits', action='store_true', help='Add random digits')
    parser.add_argument('--prefix', type=str, default='', help='Name prefix')
    parser.add_argument('--suffix', type=str, default='', help='Name suffix')
    parser.add_argument('--count', type=int, default=1, help='Number of names to generate (default: 1)')
    parser.add_argument('--unique', action='store_true', help='Ensure uniqueness in output')
    parser.add_argument('--save-model', type=str, help='Save model to JSON file')
    parser.add_argument('--load-model', type=str, help='Load model from JSON file')
    parser.add_argument('--max-repeat', type=int, default=2)
    parser.add_argument('--max-ngram-repeat', type=int, default=2)
    parser.add_argument('--max-ngram-size', type=int, default=6)
    parser.add_argument('--beam-random-top', type=int, default=10)
    parser.add_argument('--alphabet', type=str, default=None, help="'latin' for a-z only, otherwise any letters")
    parser.add_argument('--keep-chars', type=str, default=None, help='Characters to keep (e.g., "_-")')
    args = parser.parse_args()

    validate_args(args)

    names = load_names(args.names_file, alphabet=args.alphabet, keep_chars=args.keep_chars)
    if not names:
        print('Error: No names loaded from file.', file=sys.stderr)
        sys.exit(1)

    if args.length is not None:
        args.min_length = args.max_length = args.length

    start_letter = args.start.lower() if args.start else None
    end_letter = args.end.lower() if args.end else None

    model_counts = load_model(args.load_model)
    if model_counts is None:
        model_counts = build_model_counts(names, args.order, args.smooth)

    if args.save_model:
        save_model(model_counts, args.save_model)

    if not check_constraints_feasible(model_counts, args.order, start_letter, end_letter, args.min_length, args.max_length):
        print("Warning: Constraints may be too restrictive. No valid starting transitions found.", file=sys.stderr)

    style = StyleConfig(
        camel=args.style_camel,
        alternate=args.style_alternate,
        xx=args.style_xx,
        add_digits=args.add_digits,
        prefix=args.prefix,
        suffix=args.suffix
    )

    results = []
    attempts = 0
    existing: Set[str] = {n.lower() for n in names}
    generated_set: Set[str] = set()

    while len(results) < args.count and attempts < args.max_attempts:
        attempts += 1

        if args.beam > 1:
            cands = beam_search(
                model_counts,
                args.order,
                start_letter,
                end_letter,
                args.min_length,
                args.max_length or 100,
                args.beam,
                args.temp,
                args.max_repeat,
                args.max_ngram_repeat,
                args.max_ngram_size
            )
            if not cands:
                continue
            top = min(args.beam_random_top, len(cands))
            base_name = random.choice(cands[:top])
        else:
            ctx = '^' * args.order
            base_name = ''
            while True:
                ch = get_next_char(model_counts, ctx, args.order, args.temp)
                if not ch or ch == '$':
                    break
                candidate = base_name + ch
                if has_char_repeat(candidate, args.max_repeat):
                    continue
                if has_ngram_loop(candidate, args.max_ngram_repeat, args.max_ngram_size):
                    continue
                base_name = candidate
                ctx = (ctx + ch)[-args.order:]
                if args.max_length and len(base_name) >= args.max_length:
                    break

        name_lower = base_name.lower()

        if len(base_name) < args.min_length:
            continue
        if args.max_length and len(base_name) > args.max_length:
            continue
        if start_letter and not name_lower.startswith(start_letter):
            continue
        if end_letter and not name_lower.endswith(end_letter):
            continue
        if args.forbid_letters and any(ch in name_lower for ch in args.forbid_letters.lower()):
            continue
        if args.forbid_pattern and re.search(args.forbid_pattern, base_name, re.IGNORECASE):
            continue
        if name_lower in existing:
            continue

        styled = apply_style(base_name, style)

        if args.unique and styled in generated_set:
            continue

        generated_set.add(styled)
        results.append(styled)

    for result in results:
        print(result)

    if len(results) < args.count:
        print(f"Warning: Generated only {len(results)} out of {args.count} requested names. Try adjusting constraints.", file=sys.stderr)

if __name__ == '__main__':
    main()
