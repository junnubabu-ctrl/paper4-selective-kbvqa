"""Prepare official annotation metadata; image downloads are a separate step."""
from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_ROOT=_Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in _sys.path: _sys.path.insert(0, str(_ROOT / "src"))
import argparse, json, os, subprocess
from pathlib import Path

URLS={
'okvqa': {
 'train_questions':'https://okvqa.allenai.org/static/data/OpenEnded_mscoco_train2014_questions.json.zip',
 'test_questions':'https://okvqa.allenai.org/static/data/OpenEnded_mscoco_val2014_questions.json.zip',
 'train_annotations':'https://okvqa.allenai.org/static/data/mscoco_train2014_annotations.json.zip',
 'test_annotations':'https://okvqa.allenai.org/static/data/mscoco_val2014_annotations.json.zip'},
'aokvqa': {'annotations':'https://prior-datasets.s3.us-east-2.amazonaws.com/aokvqa/aokvqa_v1p0.tar.gz'}}

def run(cmd): subprocess.run(cmd,check=True)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',choices=['okvqa','aokvqa'],required=True); ap.add_argument('--out',default='datasets'); args=ap.parse_args()
    out=Path(args.out)/args.dataset; out.mkdir(parents=True,exist_ok=True)
    if args.dataset=='okvqa':
        for name,url in URLS['okvqa'].items():
            z=out/f'{name}.zip'; run(['curl','-fL',url,'-o',str(z)]); run(['unzip','-o',str(z),'-d',str(out)])
    else:
        z=out/'aokvqa_v1p0.tar.gz'; run(['curl','-fL',URLS['aokvqa']['annotations'],'-o',str(z)]); run(['tar','-xzf',str(z),'-C',str(out)])
    (out/'SOURCE_URLS.json').write_text(json.dumps(URLS[args.dataset],indent=2),encoding='utf-8')
    print(f'Prepared annotation metadata in {out}. COCO images must be obtained under the COCO terms.')
if __name__=='__main__': main()
