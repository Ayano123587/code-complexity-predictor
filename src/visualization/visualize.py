import argparse,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('evaluation',type=Path);a=p.parse_args();r=json.loads(a.evaluation.read_text())
    print('Actual \\ predicted: '+', '.join(r['label_order']))
    for label,row in zip(r['label_order'],r['confusion_matrix']):print(label, row)
if __name__=='__main__':main()
