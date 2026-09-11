#!/usr/bin/env python
from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_ROOT=_Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in _sys.path:
    _sys.path.insert(0, str(_ROOT / "src"))

import argparse, urllib.request, time
from pathlib import Path
from paper4_kbvqa.data.manifest import load_jsonl


def main():
    ap=argparse.ArgumentParser(description="Download only COCO images referenced by a manifest.")
    ap.add_argument("--manifest",required=True)
    ap.add_argument("--out-dir",required=True)
    ap.add_argument("--split",choices=["train2017","val2017"],required=True)
    ap.add_argument("--base-url",default="https://images.cocodataset.org")
    args=ap.parse_args()

    rows=load_jsonl(args.manifest)
    out=Path(args.out_dir)/args.split
    out.mkdir(parents=True,exist_ok=True)
    failures=[]
    from PIL import Image
    def valid_image(path):
        try:
            with Image.open(path) as image: image.verify()
            return True
        except (OSError,ValueError): return False
    for i,s in enumerate(rows,1):
        image_id=(s.metadata or {}).get("image_id")
        if image_id is None:
            failures.append({"question_id":s.question_id,"reason":"missing image_id"})
            continue
        name=f"{int(image_id):012d}.jpg"
        dest=out/name
        if dest.exists() and valid_image(dest):
            continue
        url=f"{args.base_url}/{args.split}/{name}"
        try:
            part=dest.with_suffix('.part')
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(url,timeout=60) as response, part.open('wb') as output:
                        import shutil
                        shutil.copyfileobj(response,output)
                    if not valid_image(part): raise ValueError('Downloaded file is not a valid image')
                    part.replace(dest)
                    break
                except Exception:
                    part.unlink(missing_ok=True)
                    if attempt==2: raise
                    time.sleep(attempt+1)
        except Exception as exc:
            failures.append({"question_id":s.question_id,"url":url,"error":repr(exc)})
            if dest.exists():
                dest.unlink()
        if i % 100 == 0:
            print(f"processed={i}/{len(rows)} failures={len(failures)}")
    print(f"completed={len(rows)-len(failures)} failures={len(failures)}")
    if failures:
        raise RuntimeError(f"COCO download failures: {failures[:5]}")


if __name__=="__main__":
    main()
