#!/usr/bin/env python3
"""Prepare a frozen, train-derived 50-image/question development pilot. No inference."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paper4_kbvqa.data.restart_pilot import prepare_pilot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--annotations", type=Path, help="Existing aokvqa_v1p0_train.json; no val/test file permitted")
    source.add_argument("--archive", type=Path, help="Existing capped official-format archive; selective train-member read only")
    source.add_argument("--download-annotations", action="store_true", help="One capped official archive request; read train member only")
    parser.add_argument("--download-record", type=Path, help="Observed official download JSON matching --archive URL/SHA/size")
    parser.add_argument("--download-images", action="store_true", help="Download only the 50 selected COCO train2017 images")
    parser.add_argument("--timeout", type=float, default=30.0, help="Socket timeout per request, seconds")
    parser.add_argument("--max-consecutive-failures", type=int, default=3,
                        help="Stop image requests after this many consecutive failures; ledger retains all 50 identities")
    args = parser.parse_args()
    if not 0 < args.timeout <= 120:
        parser.error("timeout must be in (0,120] seconds")
    if args.download_record is not None and args.archive is None:
        parser.error("download-record requires archive")
    if args.max_consecutive_failures < 1:
        parser.error("max-consecutive-failures must be positive")
    def progress(event):
        print(json.dumps({"progress": event}), file=sys.stderr, flush=True)
    try:
        report = prepare_pilot(args.out_dir, args.annotations, args.download_annotations, args.download_images, args.timeout,
                               archive_path=args.archive, download_record=args.download_record,
                               max_consecutive_failures=args.max_consecutive_failures, progress=progress)
    except Exception as error:
        report = {"status": "BLOCKED", "error": f"{type(error).__name__}: {error}",
                  "benchmark_results": False, "calibration_performed": False}
    print(json.dumps(report, indent=2))
    return 2 if report["status"] == "BLOCKED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
