"""Bounded, train-only development preparation. No model loading or result fabrication."""
from __future__ import annotations

import hashlib
import json
import os
import random
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Callable

ANNOTATION_URL = "https://prior-datasets.s3.us-east-2.amazonaws.com/aokvqa/aokvqa_v1p0.tar.gz"
TRAIN_NAME = "aokvqa_v1p0_train.json"
SEED = 2026
PILOT_SIZE = 50
TRAIN_QUESTION_COUNT = 17056
TRAIN_IMAGE_COUNT = 16540
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_ANNOTATION_BYTES = 64 * 1024 * 1024
MAX_IMAGE_BYTES = 20 * 1024 * 1024
INFERENCE_KEYS = frozenset({"question_id", "image_path", "question"})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def jsonl_bytes(rows) -> bytes:
    return b"".join((json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8") for row in rows)


def freeze_bytes(path: Path, content: bytes) -> None:
    """Create exclusively and reject changed existing content; never overwrite a record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Frozen content differs: {path}")
        return
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".part", dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)  # Exclusive publish; no overwriting a concurrent writer.
        except FileExistsError:
            if path.read_bytes() != content:
                raise ValueError(f"Frozen content differs: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def download_atomic(url: str, destination: Path, timeout: float, cap: int, validator=None) -> bool:
    """Bounded stream, validate before publishing. Existing files must validate."""
    destination = Path(destination)
    if destination.exists():
        if destination.stat().st_size > cap:
            raise ValueError(f"Existing file exceeds byte cap: {destination}")
        if validator:
            validator(destination)
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".part", dir=destination.parent)
    temporary = Path(temporary)
    try:
        started = time.monotonic()
        request = urllib.request.Request(url, headers={"User-Agent": "JB-PhD-Research-Pilot/1.0"})
        with urllib.request.urlopen(request, timeout=timeout) as response, os.fdopen(fd, "wb") as stream:
            fd = None
            if not response.geturl().startswith("https://"):
                raise ValueError("Download redirected away from HTTPS")
            length = response.headers.get("Content-Length")
            if length and int(length) > cap:
                raise ValueError("Download Content-Length exceeds byte cap")
            total = 0
            while True:
                if time.monotonic() - started > timeout:
                    raise TimeoutError("Download exceeded elapsed-time budget")
                chunk = response.read(min(1024 * 1024, cap + 1 - total))
                if not chunk:
                    break
                total += len(chunk)
                if total > cap:
                    raise ValueError("Download exceeds byte cap")
                stream.write(chunk)
            if total == 0:
                raise ValueError("Empty download")
            stream.flush()
            os.fsync(stream.fileno())
        if validator:
            validator(temporary)
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if sha256_file(destination) != sha256_file(temporary):
                raise ValueError(f"Concurrent file differs: {destination}")
        return True
    finally:
        if fd is not None:
            os.close(fd)
        temporary.unlink(missing_ok=True)


def validate_train_annotations(content: bytes, require_official_counts: bool = True) -> list[dict]:
    if not content or len(content) > MAX_ANNOTATION_BYTES:
        raise ValueError("Training annotations empty or exceed byte cap")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate annotation JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"Nonfinite annotation JSON constant: {value}")

    rows = json.loads(content.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    if not isinstance(rows, list) or len(rows) < PILOT_SIZE:
        raise ValueError("Training annotations must contain at least 50 records")
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Annotation must be an object")
        if row.get("split", "train") != "train":
            raise ValueError("Only training annotations are permitted")
        question_id = row.get("question_id")
        image_id = row.get("image_id")
        if not isinstance(question_id, (str, int)) or isinstance(question_id, bool) or not str(question_id).strip():
            raise ValueError("Missing or invalid question identity")
        if str(question_id) in seen:
            raise ValueError("Duplicate question identity in training annotations")
        seen.add(str(question_id))
        if not isinstance(image_id, int) or isinstance(image_id, bool) or not 0 <= image_id <= 999999999999:
            raise ValueError("Image identity must be an integer representable by a COCO filename")
        if not isinstance(row.get("question"), str) or not row["question"].strip():
            raise ValueError("Missing question text")
        answers = row.get("direct_answers")
        if not isinstance(answers, list) or not answers or any(not isinstance(x, str) for x in answers):
            raise ValueError("Training references require direct_answers")
        if not isinstance(row.get("difficult_direct_answer"), bool):
            raise ValueError("Missing A-OKVQA difficulty flag")
    if require_official_counts and (len(rows) != TRAIN_QUESTION_COUNT or len({r["image_id"] for r in rows}) != TRAIN_IMAGE_COUNT):
        raise ValueError("A-OKVQA v1.0 train identity requires exactly 17,056 questions and 16,540 images; held-out/subset files are not accepted")
    return rows


def train_from_archive(archive: Path) -> bytes:
    """Read only the expected regular training member; never call extract/extractall."""
    with tarfile.open(archive, "r:gz") as package:
        candidates = []
        for count, member in enumerate(package, 1):
            if count > 100:
                raise ValueError("Too many archive members")
            normalized = PurePosixPath(member.name)
            if normalized.name != TRAIN_NAME:
                continue
            if member.name not in {TRAIN_NAME, "aokvqa_v1p0/" + TRAIN_NAME}:
                raise ValueError("Unexpected training member path")
            if not member.isfile() or member.size > MAX_ANNOTATION_BYTES:
                raise ValueError("Training member is not a bounded regular file")
            candidates.append(member)
        if len(candidates) != 1:
            raise ValueError("Archive must contain exactly one expected training JSON")
        reader = package.extractfile(candidates[0])
        if reader is None:
            raise ValueError("Cannot read training member")
        content = reader.read(MAX_ANNOTATION_BYTES + 1)
    validate_train_annotations(content)
    return content


def choose_pilot(rows: list[dict]) -> tuple[list[dict], list[dict], list[int]]:
    groups = {}
    for row in rows:
        groups.setdefault(row["image_id"], []).append(row)
    image_ids = sorted(groups)
    if len(image_ids) < PILOT_SIZE:
        raise ValueError("At least 50 independent training image groups required")
    random.Random(SEED).shuffle(image_ids)
    selected_images = image_ids[:PILOT_SIZE]
    selected = [min(groups[image_id], key=lambda row: str(row["question_id"])) for image_id in selected_images]
    inference = [{"question_id": str(row["question_id"]), "question": row["question"],
                  "image_path": f"images/train2017/{row['image_id']:012d}.jpg"} for row in selected]
    references = [{"question_id": str(row["question_id"]), "image_id": row["image_id"],
                   "answers": row["direct_answers"], "difficult_direct_answer": row["difficult_direct_answer"],
                   "role": "offline_reference_only"} for row in selected]
    return inference, references, selected_images


def image_info(path: Path) -> dict:
    from PIL import Image
    with Image.open(path) as image:
        if image.format != "JPEG":
            raise ValueError("COCO image must decode as JPEG")
        width, height = image.size
        if min(width, height) < 1 or width * height > 100_000_000:
            raise ValueError("Invalid or excessive original image dimensions")
        image.verify()
    with Image.open(path) as image:
        image.load()  # Also reject truncated streams that a header-only check misses.
    return {"sha256": sha256_file(path), "bytes": path.stat().st_size,
            "original_width": width, "original_height": height, "format": "JPEG"}


def load_pilot_manifest(pilot_dir: Path, require_images: bool = True) -> list[dict]:
    """Validate label-free identity, return paths resolved relative to the portable folder."""
    pilot_dir = Path(pilot_dir).resolve()
    source = json.loads((pilot_dir / "source_manifest.json").read_text(encoding="utf-8"))
    if source.get("source_split") != "train" or source.get("partition_role") != "development_only":
        raise ValueError("Expected train-derived development-only pilot")
    if source.get("source_annotation_question_count") != TRAIN_QUESTION_COUNT or source.get("source_annotation_distinct_image_count") != TRAIN_IMAGE_COUNT:
        raise ValueError("Full official training cardinalities missing from source identity")
    result = []
    for line in (pilot_dir / "inference_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if not isinstance(row, dict) or set(row) != INFERENCE_KEYS:
            raise ValueError("Inference fields must be exactly question_id, image_path, question; labels forbidden")
        if not all(isinstance(row[key], str) and row[key].strip() for key in INFERENCE_KEYS):
            raise ValueError("Invalid inference field")
        relative = PurePosixPath(row["image_path"])
        if relative.is_absolute() or ".." in relative.parts or "\\" in row["image_path"] or len(relative.parts) != 3:
            raise ValueError("Invalid portable image path")
        if relative.parts[:2] != ("images", "train2017"):
            raise ValueError("Expected train2017 image path")
        path = pilot_dir.joinpath(*relative.parts)
        if not path.resolve().is_relative_to(pilot_dir):
            raise ValueError("Image symlink escapes pilot directory")
        if require_images and not path.is_file():
            raise ValueError(f"Image unavailable: {path}")
        result.append({**row, "image_path": str(path)})
    if len(result) != PILOT_SIZE or len({row["question_id"] for row in result}) != PILOT_SIZE or len({row["image_path"] for row in result}) != PILOT_SIZE:
        raise ValueError("Pilot must have exactly 50 distinct question/image identities")
    expected = source.get("inference_manifest_sha256")
    if expected != sha256_file(pilot_dir / "inference_manifest.jsonl"):
        raise ValueError("Frozen inference manifest hash changed")
    if source.get("reference_manifest_sha256") != sha256_file(pilot_dir / "offline_reference_manifest.jsonl"):
        raise ValueError("Frozen offline reference manifest hash changed")
    if source.get("annotation_sha256") != sha256_file(pilot_dir / "annotations" / TRAIN_NAME):
        raise ValueError("Source training annotation hash changed")
    if source.get("download_record_sha256") is not None and source["download_record_sha256"] != sha256_file(pilot_dir / "annotations" / "annotation_download_record.json"):
        raise ValueError("Observed download record hash changed")
    return result


def validate_prepared_pilot(pilot_dir: Path, require_images: bool = True) -> dict:
    """Shared entry point for inference: validates source, frozen files and actual images."""
    pilot_dir = Path(pilot_dir).resolve()
    rows = load_pilot_manifest(pilot_dir, require_images=require_images)
    source = json.loads((pilot_dir / "source_manifest.json").read_text(encoding="utf-8"))
    inventory = json.loads((pilot_dir / "image_inventory.json").read_text(encoding="utf-8"))
    records = inventory.get("images", [])
    image_ids = source["development_image_ids"]
    if len(records) != PILOT_SIZE or {r["image_id"] for r in records} != set(image_ids):
        raise ValueError("Inventory must map exactly the 50 reserved image identities")
    by_image = {r["image_id"]: r for r in records}
    reference_rows = [json.loads(line) for line in (pilot_dir / "offline_reference_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(reference_rows) != PILOT_SIZE or {r["question_id"] for r in reference_rows} != {r["question_id"] for r in rows}:
        raise ValueError("Offline reference identities do not match inference questions")
    reference_by_question = {r["question_id"]: r for r in reference_rows}
    for row in rows:
        image_id = reference_by_question[row["question_id"]]["image_id"]
        if image_id not in by_image:
            raise ValueError("Reference image missing from reserved development groups")
        expected_relative = f"images/train2017/{image_id:012d}.jpg"
        if by_image[image_id].get("relative_path") != expected_relative or Path(row["image_path"]).name != f"{image_id:012d}.jpg":
            raise ValueError("Question/image mapping changed")
        if require_images:
            details = image_info(Path(row["image_path"]))
            recorded = by_image[image_id]
            if recorded.get("status") != "verified_local" or any(recorded.get(key) != value for key, value in details.items()):
                raise ValueError("Image content or dimensions changed from verified inventory")
        row["metadata"] = {"dataset": "aokvqa", "split": "train", "source_split": "train",
                           "partition_role": "development_only", "image_id": image_id}
    return {"rows": rows, "source": source, "source_manifest_sha256": sha256_file(pilot_dir / "source_manifest.json"),
            "image_inventory_sha256": sha256_file(pilot_dir / "image_inventory.json")}


def prepare_pilot(out_dir: Path, annotations: Path | None, download_annotations: bool,
                  download_images: bool, timeout: float, archive_path: Path | None = None,
                  download_record: Path | None = None, max_consecutive_failures: int = 3,
                  progress: Callable[[dict], None] | None = None) -> dict:
    if max_consecutive_failures < 1:
        raise ValueError("max_consecutive_failures must be positive")
    out_dir = Path(out_dir).resolve()
    target = out_dir / "annotations" / TRAIN_NAME
    archive = out_dir / "annotations" / "aokvqa_v1p0.tar.gz"
    provenance_sha = None
    if archive_path is not None:
        archive_path = Path(archive_path)
        if archive_path.stat().st_size > MAX_ARCHIVE_BYTES:
            raise ValueError("Supplied archive exceeds byte cap")
        content = train_from_archive(archive_path)
        rows = validate_train_annotations(content)
        archive_sha = sha256_file(archive_path)
        source_kind, source_url = "user_supplied_local_archive", None
        if download_record is not None:
            download_record = Path(download_record)
            record = json.loads(download_record.read_text(encoding="utf-8"))
            if (record.get("url") != ANNOTATION_URL or record.get("status") != "downloaded"
                    or record.get("synthetic") is not False or record.get("sha256") != archive_sha
                    or record.get("size_bytes") != archive_path.stat().st_size):
                raise ValueError("Download record does not prove the supplied official archive identity")
            source_kind, source_url = "official_archive_download_with_record", ANNOTATION_URL
            provenance_sha = sha256_file(download_record)
            freeze_bytes(out_dir / "annotations" / "annotation_download_record.json", download_record.read_bytes())
        freeze_bytes(archive, archive_path.read_bytes())
    elif annotations is not None:
        if download_record is not None:
            raise ValueError("--download-record is supported only with --archive")
        annotations = Path(annotations)
        if annotations.name != TRAIN_NAME:
            raise ValueError(f"--annotations must name {TRAIN_NAME}; held-out annotations are not allowed")
        if annotations.stat().st_size > MAX_ANNOTATION_BYTES:
            raise ValueError("Supplied annotations exceed byte cap")
        content = annotations.read_bytes()
        rows = validate_train_annotations(content)
        source_kind, source_url = "user_supplied_local_training_annotations", None
        archive_sha = None
    elif target.exists():
        if target.stat().st_size > MAX_ANNOTATION_BYTES:
            raise ValueError("Existing annotations exceed byte cap")
        content = target.read_bytes()
        rows = validate_train_annotations(content)
        existing_source = out_dir / "source_manifest.json"
        if existing_source.exists():
            prior = json.loads(existing_source.read_text(encoding="utf-8"))
            source_kind, source_url, archive_sha = prior["source_kind"], prior["download_url"], prior["archive_sha256"]
            provenance_sha = prior.get("download_record_sha256")
        elif archive.exists():
            source_kind, source_url, archive_sha = "official_archive_download", ANNOTATION_URL, sha256_file(archive)
        else:
            source_kind, source_url, archive_sha = "preexisting_local_training_annotations", None, None
    elif download_annotations:
        download_atomic(ANNOTATION_URL, archive, timeout, MAX_ARCHIVE_BYTES, train_from_archive)
        content = train_from_archive(archive)
        rows = validate_train_annotations(content)
        source_kind, source_url, archive_sha = "official_archive_download", ANNOTATION_URL, sha256_file(archive)
    else:
        raise ValueError("Training annotations missing; supply --annotations or --download-annotations")
    inference, references, image_ids = choose_pilot(rows)
    inference_content, reference_content = jsonl_bytes(inference), jsonl_bytes(references)
    source = {"schema_version": 1, "dataset": "A-OKVQA v1.0", "source_split": "train",
              "partition_role": "development_only", "not_for": ["calibration_fit", "threshold_selection", "held_out_evaluation"],
              "seed": SEED, "question_count": PILOT_SIZE, "distinct_image_count": PILOT_SIZE,
              "source_annotation_question_count": len(rows), "source_annotation_distinct_image_count": len({r["image_id"] for r in rows}),
              "source_kind": source_kind, "download_url": source_url, "archive_sha256": archive_sha,
              "download_record_sha256": provenance_sha,
              "annotation_sha256": hashlib.sha256(content).hexdigest(),
              "inference_manifest_sha256": hashlib.sha256(inference_content).hexdigest(),
              "reference_manifest_sha256": hashlib.sha256(reference_content).hexdigest(),
              "development_image_ids": image_ids, "development_question_ids": [r["question_id"] for r in inference],
              "allocation_rule": "sorted image identities shuffled with Python Random(2026); first 50; lexicographically smallest question ID per image",
              "reserve_rule": "Exclude every development_image_id and all its questions from subsequent calibration-fit and threshold sets",
              "gold_in_inference": False, "references_role": "offline evaluator only; never query, prompt or retrieval evidence",
              "source_authenticity": ("Downloaded URL and SHA recorded; no upstream signed checksum authentication"
                                      if source_kind in {"official_archive_download", "official_archive_download_with_record"}
                                      else "Local supplied origin not independently verified; schema/cardinalities and SHA recorded"),
              "raw_images_status": "separate image_inventory.json records actual verified files; URLs alone are metadata"}
    # Check source identity before publishing a new run's supporting files.
    frozen_source = out_dir / "source_manifest.json"
    if frozen_source.exists() and frozen_source.read_bytes() != json_bytes(source):
        raise ValueError("Frozen source identity changed; use a distinct directory, never overwrite the pilot")
    freeze_bytes(target, content)
    freeze_bytes(out_dir / "inference_manifest.jsonl", inference_content)
    freeze_bytes(out_dir / "offline_reference_manifest.jsonl", reference_content)
    freeze_bytes(frozen_source, json_bytes(source))
    previous_inventory_path = out_dir / "image_inventory.json"
    previous = json.loads(previous_inventory_path.read_text(encoding="utf-8")) if previous_inventory_path.exists() else {"images": []}
    prior_images = {r["image_id"]: r for r in previous["images"]}
    inventory = []
    failures = []
    consecutive_failures = 0
    network_attempts = 0
    for ordinal, (image_id, row) in enumerate(zip(image_ids, inference), 1):
        image_path = out_dir.joinpath(*PurePosixPath(row["image_path"]).parts)
        url = f"https://images.cocodataset.org/train2017/{image_id:012d}.jpg"
        requested_this_file = download_images and not image_path.exists()
        if requested_this_file and consecutive_failures >= max_consecutive_failures:
            inventory.append({**prior_images.get(image_id, {}), "image_id": image_id,
                              "relative_path": row["image_path"], "url": url,
                              "status": "not_attempted_due_to_network_failures"})
            continue
        try:
            if download_images:
                if requested_this_file:
                    network_attempts += 1
                    if progress:
                        progress({"stage": "image_download_attempt", "image_id": image_id,
                                  "ordinal": ordinal, "total": PILOT_SIZE, "attempt": network_attempts})
                download_atomic(url, image_path, timeout, MAX_IMAGE_BYTES, image_info)
                if requested_this_file:
                    consecutive_failures = 0
            entry = {"image_id": image_id, "relative_path": row["image_path"], "url": url,
                     "status": "verified_local" if image_path.is_file() else "metadata_only_not_downloaded"}
            if image_path.is_file():
                entry.update(image_info(image_path))
                old = prior_images.get(image_id, {})
                if old.get("sha256") and old["sha256"] != entry["sha256"]:
                    raise ValueError("Existing verified image hash changed")
            elif prior_images.get(image_id, {}).get("sha256"):
                raise ValueError("Previously verified image is now missing")
            inventory.append(entry)
            if requested_this_file and progress:
                progress({"stage": "image_download_verified", "image_id": image_id, "ordinal": ordinal})
        except Exception as error:
            if requested_this_file:
                consecutive_failures += 1
                if progress:
                    progress({"stage": "image_download_failed", "image_id": image_id,
                              "ordinal": ordinal, "consecutive_failures": consecutive_failures,
                              "stop_after": max_consecutive_failures, "error": f"{type(error).__name__}: {error}"})
            failures.append({"image_id": image_id, "error": f"{type(error).__name__}: {error}"})
            inventory.append({**prior_images.get(image_id, {}), "image_id": image_id, "relative_path": row["image_path"], "url": url,
                              "status": "blocked", "error": str(error)})
    # Inventory is a status ledger, intentionally refreshable; frozen experiment identity is separate.
    report = {"dataset": "A-OKVQA / COCO train2017", "requested_images": PILOT_SIZE,
              "verified_images": sum(r["status"] == "verified_local" for r in inventory),
              "downloads_requested": download_images, "network_attempts": network_attempts,
              "max_consecutive_failures": max_consecutive_failures,
              "not_attempted_images": sum(r["status"] == "not_attempted_due_to_network_failures" for r in inventory),
              "images": inventory, "failures": failures}
    inventory_bytes = json_bytes(report)
    previous_inventory_path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="image_inventory.", suffix=".part", dir=out_dir)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(inventory_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, previous_inventory_path)
    finally:
        Path(name).unlink(missing_ok=True)
    load_pilot_manifest(out_dir, require_images=False)
    status = "BLOCKED" if failures or (download_images and report["verified_images"] != PILOT_SIZE) else (
        "PREPARED" if report["verified_images"] == PILOT_SIZE else "METADATA_PREPARED")
    return {"status": status, "pilot_dir": str(out_dir),
            "partition_role": "development_only", "questions": PILOT_SIZE, "distinct_images": PILOT_SIZE,
            "verified_images": report["verified_images"], "failures": failures,
            "network_attempts": network_attempts, "not_attempted_images": report["not_attempted_images"],
            "benchmark_results": False, "calibration_performed": False}
