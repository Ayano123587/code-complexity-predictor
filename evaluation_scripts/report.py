"""Read aggregate histories without loading data, checkpoints or scoring a test."""
import argparse,json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path);args=parser.parse_args()
    r=json.loads((args.output/'selection.json').read_text())
    print(json.dumps({'selected':r['selected'],'candidates':[{'candidate':x['candidate'],'oof_macro_f1':x['oof']['macro_f1'],'validation_macro_f1':x['validation']['macro_f1']} for x in r['runs']],'failures':r['failures'],'test_evaluated':r['test_evaluated']},indent=2))


if __name__=='__main__':main()
