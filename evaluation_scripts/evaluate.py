"""Explicit, retrospective observed-holdout evaluation after frozen selection."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study',choices=['selected','mechanics','family']);parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--observed-holdout',action='store_true',required=True,help='Acknowledge this is an already-observed retrospective cohort')
    parser.add_argument('--data',type=Path);parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    if args.study == 'selected':
        if args.data is None or args.report is None: parser.error('selected requires --data and fresh --report')
        from src.models.selected import evaluate
        print(json.dumps(evaluate(args.data,args.output,args.report,acknowledge_observed=True),indent=2))
        return
    selection=json.loads((args.output/'selection.json').read_text())
    if selection['selected']=='v2_baseline':
        print(json.dumps({'test_evaluated':False,'reason':'Existing baseline retained; no new holdout scoring'}));return
    if args.study=='mechanics':from src.models.mechanics_v3 import release
    else:from src.models.family_study import release
    print(json.dumps(release(args.output),indent=2))


if __name__=='__main__':main()
