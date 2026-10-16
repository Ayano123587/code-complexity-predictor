"""Conservative name-invariant loop/recursion mechanics; covariates, not Big-O proofs."""
from sklearn.base import BaseEstimator, TransformerMixin
from src.features.code_v2 import lex, structural, IDENTIFIER, RESERVED


def mechanics(code):
    tokens=lex(code);pairs={};stack=[]
    for i,token in enumerate(tokens):
        if token in ['(','{']:stack.append((token,i))
        elif token in [')','}'] and stack and stack[-1][0]=={')':'(','}':'{'}[token]:
            _,start=stack.pop();pairs[start]=i
    result={key:0 for key in ['for_multiplicative_updates','for_additive_updates',
        'for_constant_bounds','for_dependent_nested_bounds','while_multiplicative_updates',
        'recursive_division_calls','recursive_subtraction_calls','sort_inside_loop']}
    loops=[]
    def statement_end(i,depth=0):
        if i>=len(tokens) or depth>100:return min(i,len(tokens)-1)
        if tokens[i]=='{':return pairs.get(i,i)
        if tokens[i] in ['for','while','if'] and i+1 in pairs and tokens[i+1]=='(':
            return statement_end(pairs[i+1]+1,depth+1)
        j=i
        while j<len(tokens):
            if tokens[j]==';':return j
            if tokens[j] in ['(','{'] and j in pairs:j=pairs[j]+1
            else:j+=1
        return len(tokens)-1
    def identifier(token):return bool(IDENTIFIER.fullmatch(token)) and token not in RESERVED
    def multiplicative(part):
        return any(part[j] in ['*','/','<<','>>'] and part[j+1]=='=' for j in range(len(part)-1)) or any(part[j]=='=' and any(t in ['*','/','<<','>>'] for t in part[j+1:]) for j in range(len(part)))
    for i,token in enumerate(tokens):
        if token not in ['for','while'] or i+1 not in pairs or tokens[i+1]!='(':continue
        end=pairs[i+1];body_start=end+1
        body_end=statement_end(body_start)
        part=tokens[i+2:end];body=tokens[body_start:body_end+1]
        counter=None
        if token=='for' and part.count(';')==2:
            left,right=[j for j,t in enumerate(part) if t==';']
            init,condition,update=part[:left],part[left+1:right],part[right+1:]
            eq=next((j for j,t in enumerate(init) if t=='='),len(init))
            counter=next((t for t in reversed(init[:eq]) if identifier(t)),None)
            result['for_multiplicative_updates']+=int(multiplicative(update))
            result['for_additive_updates']+=int(any(t in ['++','--','+','-'] for t in update))
            bounds=[t for t in condition if identifier(t) and t!=counter]
            result['for_constant_bounds']+=int(any(t[0].isdigit() for t in condition) and not bounds)
            result['for_dependent_nested_bounds']+=int(any(start<i<finish and outer in bounds for start,finish,outer in loops if outer))
        elif token=='while':
            result['while_multiplicative_updates']+=int(multiplicative(body))
        result['sort_inside_loop']+=int('sort' in body)
        loops.append((i,body_end,counter))
    for i in range(len(tokens)-2):
        if not identifier(tokens[i]) or tokens[i+1]!='(':continue
        end=pairs.get(i+1)
        if end is None or end+1>=len(tokens) or tokens[end+1]!='{':continue
        finish=pairs.get(end+1,end+1)
        for j in range(end+2,finish):
            if tokens[j]!=tokens[i] or j+1>=finish or tokens[j+1]!='(' or j+1 not in pairs:continue
            args=tokens[j+2:pairs[j+1]]
            result['recursive_division_calls']+=int(any(args[k] in ['/','>>'] and args[k+1][0].isdigit() for k in range(len(args)-1)))
            result['recursive_subtraction_calls']+=int(any(args[k]=='-' and args[k+1][0].isdigit() for k in range(len(args)-1)))
    return {'mechanics:'+key:float(value) for key,value in result.items()}


class CodeMechanics(BaseEstimator,TransformerMixin):
    def fit(self,x,y=None):return self
    def transform(self,x):return [{**structural(code),**mechanics(code)} for code in x]
