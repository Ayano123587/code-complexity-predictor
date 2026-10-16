"""Validation-only study launcher; fresh output and copied protocol are mandatory."""
import argparse
from pathlib import Path


def create_run(output,plan):
    output=Path(output)
    if output.exists():raise ValueError('Output already exists; preserve historical artifacts and choose a fresh directory')
    content=Path(plan).read_bytes()
    output.mkdir(parents=True)
    (output/'plan.json').write_bytes(content)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study',choices=['mechanics','family']);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    default='models/mechanics-v3/plan.json' if args.study=='mechanics' else 'models/family-weight-v4/plan.json'
    output=create_run(args.output,default)
    if args.study=='mechanics':from src.models.mechanics_v3 import run
    else:from src.models.family_study import run
    import json
    print(json.dumps(run(output),indent=2))


if __name__=='__main__':main()
