"""Name-invariant C++ lexical relationships and conservative structural features.

These are learned classifier covariates, not a formal C++ parser or Big-O proof.
"""
from collections import Counter
import math
import re
from sklearn.base import BaseEstimator, TransformerMixin
from src.features.build_features import TOKEN, KEYWORDS, normalized

LIBRARY = set(('unordered_map unordered_set priority_queue deque list multiset '
               'multimap pair tuple string swap max min abs sqrt log pow reverse '
               'accumulate count find binary_search nth_element next_permutation '
               'push pop top front back insert erase reserve resize clear empty '
               'emplace emplace_back int64_t uint64_t signed unsigned char short '
               'static inline template typename constexpr nullptr double float '
               'try catch throw virtual override operator typedef enum union').split())
RESERVED = KEYWORDS | LIBRARY
IDENTIFIER = re.compile(r'[A-Za-z_]\w*\Z')


def lex(code):
    if not isinstance(code, str):
        raise ValueError('C++ source must be text')
    tokens = []
    for token in TOKEN.findall(code):
        if token.startswith(('//', '/*')):
            continue
        tokens.append('STR' if token.startswith(('"', "'", 'R"')) else token)
    return tokens


def canonical(code):
    names, output = {}, []
    for token in lex(code):
        if token == 'STR':
            output.append(token)
        elif token[0].isdigit():
            output.append(token if token in {'0', '1', '2', '3'} else 'NUM')
        elif IDENTIFIER.fullmatch(token) and token not in RESERVED:
            if token not in names:
                names[token] = 'ID' + str(len(names))
            output.append(names[token])
        else:
            output.append(token)
    return ' '.join(output)


class CodeText(BaseEstimator, TransformerMixin):
    def __init__(self, representation='canonical'):
        self.representation = representation

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        if self.representation not in {'canonical', 'legacy'}:
            raise ValueError('Unknown code representation')
        function = canonical if self.representation == 'canonical' else normalized
        return [function(code) for code in x]


def structural(code):
    tokens = lex(code)
    counts = Counter(tokens)
    pairs, stack = {}, []
    closing = {')': '(', ']': '[', '}': '{'}
    for i, token in enumerate(tokens):
        if token in {'(', '[', '{'}:
            stack.append((token, i))
        elif token in closing and stack and stack[-1][0] == closing[token]:
            _, start = stack.pop()
            pairs[start] = i

    def statement_end(i, depth=0):
        if i >= len(tokens) or depth > 100:
            return min(i, len(tokens)-1)
        if tokens[i] == '{':
            return pairs.get(i, i)
        if tokens[i] in {'for', 'while', 'if', 'switch'} and i+1 < len(tokens) and tokens[i+1] == '(':
            return statement_end(pairs.get(i+1, i+1)+1, depth+1)
        for j in range(i, len(tokens)):
            if tokens[j] == ';':
                return j
            if tokens[j] == '{':
                return pairs.get(j, j)
        return len(tokens)-1

    loops = []
    for i, token in enumerate(tokens[:-1]):
        if token in {'for', 'while'} and tokens[i+1] == '(' and i+1 in pairs:
            end_condition = pairs[i+1]
            loops.append((i, statement_end(end_condition+1)))
    maximum_nesting = max((sum(start <= i <= end for start, end in loops) for i, _ in loops), default=0)
    recursive_functions, maximum_self_calls, functions = 0, 0, 0
    controls = {'if', 'for', 'while', 'switch', 'catch'}
    for i in range(len(tokens)-2):
        if not IDENTIFIER.fullmatch(tokens[i]) or tokens[i] in controls or tokens[i+1] != '(':
            continue
        end = pairs.get(i+1)
        if end is None or end+1 >= len(tokens) or tokens[end+1] != '{':
            continue
        body_end = pairs.get(end+1, end+1)
        self_calls = sum(tokens[j] == tokens[i] and j+1 < body_end and tokens[j+1] == '('
                         for j in range(end+2, body_end))
        functions += 1
        recursive_functions += int(self_calls > 0)
        maximum_self_calls = max(maximum_self_calls, self_calls)
    depth, brace_depth = 0, 0
    for token in tokens:
        depth += int(token == '{') - int(token == '}')
        brace_depth = max(brace_depth, depth)
    result = {'log_tokens': math.log1p(len(tokens)), 'max_loop_nesting': maximum_nesting,
              'loops': len(loops), 'brace_depth': brace_depth, 'functions': functions,
              'recursive_functions': recursive_functions, 'maximum_self_calls': maximum_self_calls,
              'small_constant_2': math.log1p(counts['2']), 'small_constant_3': math.log1p(counts['3'])}
    for token in sorted(KEYWORDS | LIBRARY | {'[', '*', '/', '%', '<<', '>>', '++', '--'}):
        result['token:' + token] = math.log1p(counts[token])
    return result


class CodeStructure(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return [structural(code) for code in x]
