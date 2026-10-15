"""Conservative lexical C++ normalization; never compiles untrusted source."""
import re
TOKEN = re.compile(r'R"[^\s()\\]{0,16}\(.*?\)[^\s()\\]{0,16}"|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\n]*|/\*.*?\*/|[A-Za-z_]\w*|\d+(?:\.\d+)?|<<|>>|<=|>=|==|!=|\+\+|--|&&|\|\||[^\s]', re.S)
KEYWORDS = set('for while do if else return int long float double void auto const bool vector map set queue stack sort begin end size push_back lower_bound upper_bound break continue switch case struct class public private include using namespace std new delete true false sizeof'.split())
def tokenize(code):
    result=[]
    for token in TOKEN.findall(code):
        if token.startswith(('//','/*')): continue
        if token.startswith(('"',"'",'R"')): token='STR'
        elif token[0].isdigit(): token='NUM'
        elif re.fullmatch(r'[A-Za-z_]\w*',token) and token not in KEYWORDS: token='ID'
        result.append(token)
    return result

def normalized(code): return ' '.join(tokenize(code))
