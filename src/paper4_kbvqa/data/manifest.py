from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

@dataclass(frozen=True)
class VQASample:
    question_id: str
    image_path: str
    question: str
    answers: tuple[str, ...] = ()
    visual_entities: tuple[str, ...] = ()
    metadata: dict = field(default_factory=dict)

    def inference_view(self) -> dict:
        return {
            "question_id": self.question_id,
            "image_path": self.image_path,
            "question": self.question,
            "visual_entities": list(self.visual_entities),
        }


def load_jsonl(path: str | Path) -> list[VQASample]:
    path = Path(path)
    out: list[VQASample] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            for key in ("question_id", "image_path", "question"):
                if key not in obj:
                    raise ValueError(f"{path}:{line_no}: missing required field {key!r}")
            out.append(VQASample(
                question_id=str(obj["question_id"]),
                image_path=str(obj["image_path"]),
                question=str(obj["question"]),
                answers=tuple(str(x) for x in obj.get("answers", [])),
                visual_entities=tuple(str(x) for x in obj.get("visual_entities", [])),
                metadata=dict(obj.get("metadata", {})),
            ))
    return out


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
