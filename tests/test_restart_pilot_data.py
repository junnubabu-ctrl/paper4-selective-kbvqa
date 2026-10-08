"""Preparation integrity tests use declared local fixtures; never benchmark results."""
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paper4_kbvqa.data.restart_pilot import (
    TRAIN_NAME, choose_pilot, download_atomic, freeze_bytes, image_urls, json_bytes, load_pilot_manifest,
    prepare_pilot, train_from_archive, validate_prepared_pilot, validate_train_annotations,
)


def fixture_rows():
    return [{"question_id": f"fixture-{index:05d}", "image_id": index % 16540,
             "question": "Declared unit-test fixture question", "direct_answers": ["fixture label"],
             "rationales": ["never include me in inference"], "choices": ["never include me"],
             "difficult_direct_answer": False} for index in range(17056)]


class RestartPilotDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = fixture_rows()
        cls.content = json_bytes(cls.rows)

    def test_fixed_seed_distinct_images_and_label_separation(self):
        inference, references, groups = choose_pilot(self.rows)
        reordered = choose_pilot(list(reversed(self.rows)))
        self.assertEqual((inference, references, groups), reordered)
        self.assertEqual(len(inference), 50)
        self.assertEqual(len(set(groups)), 50)
        self.assertTrue(all(set(row) == {"question_id", "image_path", "question"} for row in inference))
        self.assertTrue(all("answers" in row and "question" not in row for row in references))

    def test_rejects_heldout_and_subsets(self):
        changed = [dict(row) for row in self.rows]
        changed[0]["split"] = "val"
        with self.assertRaisesRegex(ValueError, "Only training"):
            validate_train_annotations(json_bytes(changed))
        with self.assertRaisesRegex(ValueError, "17,056"):
            validate_train_annotations(json_bytes(self.rows[:50]))

    def test_freeze_rejects_drift_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "file.json"
            freeze_bytes(path, b"first")
            freeze_bytes(path, b"first")
            with self.assertRaises(ValueError):
                freeze_bytes(path, b"second")
            self.assertEqual(path.read_bytes(), b"first")

    def test_safe_archive_does_not_extract_other_members(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "annotations.tar.gz"
            with tarfile.open(archive, "w:gz") as package:
                for name, content in [(TRAIN_NAME, self.content), ("../../escape", b"never extract")]:
                    member = tarfile.TarInfo(name)
                    member.size = len(content)
                    package.addfile(member, io.BytesIO(content))
            self.assertEqual(train_from_archive(archive), self.content)
            self.assertEqual(list(Path(directory).iterdir()), [archive])

    def test_rejects_training_symlink_member(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "annotations.tar.gz"
            with tarfile.open(archive, "w:gz") as package:
                member = tarfile.TarInfo(TRAIN_NAME)
                member.type = tarfile.SYMTYPE
                member.linkname = "external.json"
                package.addfile(member)
            with self.assertRaisesRegex(ValueError, "regular file"):
                train_from_archive(archive)

    def test_stable_resume_and_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / TRAIN_NAME
            source.write_bytes(self.content)
            out = Path(directory) / "pilot"
            report = prepare_pilot(out, source, False, False, 5)
            self.assertEqual(report["questions"], 50)
            self.assertEqual(report["verified_images"], 0)
            self.assertEqual(report["status"], "METADATA_PREPARED")
            frozen = (out / "source_manifest.json").read_bytes()
            self.assertEqual(prepare_pilot(out, None, False, False, 5)["questions"], 50)
            self.assertEqual((out / "source_manifest.json").read_bytes(), frozen)
            validated = validate_prepared_pilot(out, require_images=False)
            self.assertEqual(validated["rows"][0]["metadata"]["source_split"], "train")
            with self.assertRaisesRegex(ValueError, "Image unavailable"):
                load_pilot_manifest(out, require_images=True)
            inference = out / "inference_manifest.jsonl"
            rows = [json.loads(line) for line in inference.read_text().splitlines()]
            rows[0]["answers"] = ["leaked"]
            inference.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            with self.assertRaisesRegex(ValueError, "labels forbidden"):
                load_pilot_manifest(out, require_images=False)

    def test_rejects_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / TRAIN_NAME
            source.write_bytes(self.content)
            out = Path(directory) / "pilot"
            prepare_pilot(out, source, False, False, 5)
            inference = out / "inference_manifest.jsonl"
            rows = [json.loads(line) for line in inference.read_text().splitlines()]
            rows[0]["image_path"] = "../outside.jpg"
            inference.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            with self.assertRaisesRegex(ValueError, "portable image path"):
                load_pilot_manifest(out, require_images=False)

    def test_download_byte_cap_leaves_no_final_or_partial_file(self):
        class Response(io.BytesIO):
            headers = {"Content-Length": "100"}
            def geturl(self):
                return "https://example.test/file"
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "data"
            with patch("urllib.request.urlopen", return_value=Response(b"x" * 100)):
                with self.assertRaisesRegex(ValueError, "byte cap"):
                    download_atomic("https://example.test/file", target, 5, 10)
            self.assertFalse(target.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_decoded_image_dimensions_and_drift_never_accepted_on_resume(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / TRAIN_NAME
            source.write_bytes(self.content)
            out = Path(directory) / "pilot"
            prepare_pilot(out, source, False, False, 5)
            selected = json.loads((out / "source_manifest.json").read_text())["development_image_ids"]
            for image_id in selected:
                path = out / "images" / "train2017" / f"{image_id:012d}.jpg"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (11, 13), "red").save(path)
            self.assertEqual(prepare_pilot(out, None, False, False, 5)["verified_images"], 50)
            validated = validate_prepared_pilot(out)
            self.assertEqual(len(validated["rows"]), 50)
            inventory = json.loads((out / "image_inventory.json").read_text())
            first = inventory["images"][0]
            self.assertEqual((first["original_width"], first["original_height"]), (11, 13))
            altered = out / first["relative_path"]
            Image.new("RGB", (11, 13), "blue").save(altered)
            first_block = prepare_pilot(out, None, False, False, 5)
            self.assertEqual(first_block["status"], "BLOCKED")
            second_block = prepare_pilot(out, None, False, False, 5)
            self.assertEqual(second_block["status"], "BLOCKED")
            with self.assertRaisesRegex(ValueError, "changed"):
                validate_prepared_pilot(out)
            altered.unlink()
            missing_report = prepare_pilot(out, None, False, False, 5)
            self.assertEqual(missing_report["status"], "BLOCKED")
            preserved = json.loads((out / "image_inventory.json").read_text())["images"][0]
            self.assertEqual(preserved["sha256"], first["sha256"])

    def test_invalid_download_never_published(self):
        class Response(io.BytesIO):
            headers = {}
            def geturl(self):
                return "https://example.test/file"
        def reject(_):
            raise ValueError("Invalid downloaded content")
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "data"
            with patch("urllib.request.urlopen", return_value=Response(b"not a JPEG")):
                with self.assertRaisesRegex(ValueError, "Invalid downloaded"):
                    download_atomic("https://example.test/file", target, 5, 100, reject)
            self.assertFalse(target.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_consecutive_network_failures_bound_attempts_and_preserve_full_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / TRAIN_NAME
            source.write_bytes(self.content)
            out = Path(directory) / "pilot"
            events = []
            with patch("paper4_kbvqa.data.restart_pilot.download_atomic", side_effect=TimeoutError("Declared test timeout")) as downloader:
                report = prepare_pilot(out, source, False, True, 5, progress=events.append)
            self.assertEqual(report["status"], "BLOCKED")
            self.assertEqual(downloader.call_count, 3)
            self.assertEqual(report["network_attempts"], 3)
            self.assertEqual(report["not_attempted_images"], 47)
            inventory = json.loads((out / "image_inventory.json").read_text())
            self.assertEqual(len(inventory["images"]), 50)
            self.assertEqual(len({row["image_id"] for row in inventory["images"]}), 50)
            self.assertEqual([row["status"] for row in inventory["images"][:3]], ["blocked"] * 3)
            self.assertTrue(all(row["status"] == "not_attempted_due_to_network_failures" for row in inventory["images"][3:]))
            self.assertEqual(len([event for event in events if event["stage"] == "image_download_attempt"]), 3)
            self.assertEqual(len([event for event in events if event["stage"] == "image_download_failed"]), 3)
            validated = validate_prepared_pilot(out, require_images=False)
            self.assertEqual(len(validated["rows"]), 50)

    def test_same_bucket_https_routes_and_unknown_transport_rejection(self):
        canonical, selected = image_urls(287900, "coco-s3-path")
        self.assertEqual(canonical, "https://images.cocodataset.org/train2017/000000287900.jpg")
        self.assertEqual(selected, "https://s3.amazonaws.com/images.cocodataset.org/train2017/000000287900.jpg")
        self.assertEqual(image_urls(287900, "coco-host"), (canonical, canonical))
        with self.assertRaisesRegex(ValueError, "Unknown image transport"):
            image_urls(287900, "unverified-mirror")

    def test_selected_route_ledger_retains_actual_origin_across_resume(self):
        from PIL import Image
        def declared_fixture_download(url, destination, timeout, cap, validator, response_metadata=None):
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (11, 13), "red").save(destination)
            validator(destination)
            if response_metadata is not None:
                response_metadata["resolved_url"] = url
            return True
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / TRAIN_NAME
            source.write_bytes(self.content)
            out = Path(directory) / "pilot"
            prepare_pilot(out, source, False, False, 5)
            frozen_before = {name: (out / name).read_bytes() for name in
                             ["source_manifest.json", "inference_manifest.jsonl", "offline_reference_manifest.jsonl"]}
            with patch("paper4_kbvqa.data.restart_pilot.download_atomic", side_effect=declared_fixture_download) as downloader:
                result = prepare_pilot(out, None, False, True, 5, image_transport="coco-s3-path")
            self.assertEqual(result["verified_images"], 50)
            self.assertEqual(downloader.call_count, 50)
            ledger = json.loads((out / "image_inventory.json").read_text())
            first = ledger["images"][0]
            self.assertEqual(first["image_transport"], "coco-s3-path")
            self.assertEqual(first["verified_download_transport"], "coco-s3-path")
            self.assertEqual(first["verified_download_url"], first["selected_url"])
            self.assertTrue(first["canonical_url"].startswith("https://images.cocodataset.org/train2017/"))
            prepare_pilot(out, None, False, False, 5, image_transport="coco-host")
            resumed = json.loads((out / "image_inventory.json").read_text())["images"][0]
            self.assertEqual(resumed["image_transport"], "coco-host")
            self.assertEqual(resumed["verified_download_transport"], "coco-s3-path")
            self.assertEqual(resumed["verified_download_url"], first["verified_download_url"])
            self.assertEqual(resumed["sha256"], first["sha256"])
            self.assertEqual(frozen_before, {name: (out / name).read_bytes() for name in frozen_before})
            self.assertEqual(len(validate_prepared_pilot(out)["rows"]), 50)


if __name__ == "__main__":
    unittest.main()
