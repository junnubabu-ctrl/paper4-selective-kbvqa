#!/usr/bin/env python
from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_ROOT=_Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in _sys.path: _sys.path.insert(0, str(_ROOT / "src"))
import argparse, json
from pathlib import Path
from paper4_kbvqa.data.manifest import write_jsonl


def aokvqa(args):
    dataset_path=Path(args.aokvqa_dir)/f"aokvqa_v1p0_{args.split}.json"
    data=json.loads(dataset_path.read_text(encoding="utf-8"))
    image_split="test" if args.split in {"test","test_w_ans"} else args.split
    rows=[]
    for x in data:
        image_path=Path(args.coco_dir)/f"{image_split}2017"/f"{int(x['image_id']):012d}.jpg"
        rows.append({
            "question_id":str(x["question_id"]),
            "image_path":str(image_path),
            "question":x["question"],
            "answers":x.get("direct_answers",[]),
            "metadata":{"dataset":"aokvqa","split":args.split,"image_id":x["image_id"],
                        "difficult_direct_answer":x["difficult_direct_answer"]},
        })
    write_jsonl(args.out,rows)


def okvqa(args):
    q=json.loads(Path(args.questions).read_text(encoding="utf-8"))["questions"]
    anns={}
    if args.annotations:
        raw=json.loads(Path(args.annotations).read_text(encoding="utf-8"))["annotations"]
        anns={int(x["question_id"]):x for x in raw}
    rows=[]
    for x in q:
        image_path=Path(args.coco_dir)/args.image_dir/f"COCO_{args.image_dir}_{int(x['image_id']):012d}.jpg"
        rows.append({"question_id":str(x["question_id"]),"image_path":str(image_path),"question":x["question"],"answers":[a["answer"] for a in anns.get(int(x["question_id"]),{}).get("answers",[])],"metadata":{"dataset":"okvqa","image_id":x["image_id"],"split":args.image_dir,"official_annotation":anns.get(int(x["question_id"]))}})
    write_jsonl(args.out,rows)


def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest="dataset",required=True)
    a=sub.add_parser("aokvqa"); a.add_argument("--aokvqa-dir",required=True); a.add_argument("--coco-dir",required=True); a.add_argument("--split",default="val"); a.add_argument("--out",required=True)
    o=sub.add_parser("okvqa"); o.add_argument("--questions",required=True); o.add_argument("--annotations"); o.add_argument("--coco-dir",required=True); o.add_argument("--image-dir",default="val2014"); o.add_argument("--out",required=True)
    args=ap.parse_args(); aokvqa(args) if args.dataset=="aokvqa" else okvqa(args)
if __name__=="__main__": main()
