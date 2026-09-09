from __future__ import annotations
import json, os, tempfile
from pathlib import Path
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone

@dataclass
class CheckpointState:
    run_id: str
    completed_question_ids: list[str] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)
    updated_at: str = ""

class CheckpointManager:
    def __init__(self, path: str | Path): self.path=Path(path)
    def load(self) -> CheckpointState | None:
        if not self.path.exists(): return None
        data=json.loads(self.path.read_text(encoding='utf-8'))
        return CheckpointState(**data)
    def save(self, state: CheckpointState) -> None:
        state.updated_at=datetime.now(timezone.utc).isoformat()
        self.path.parent.mkdir(parents=True,exist_ok=True)
        fd,tmp=tempfile.mkstemp(dir=self.path.parent,prefix=self.path.name+'.',suffix='.tmp')
        try:
            with os.fdopen(fd,'w',encoding='utf-8') as f:
                json.dump(asdict(state),f,indent=2); f.flush(); os.fsync(f.fileno())
            os.replace(tmp,self.path)
        finally:
            if os.path.exists(tmp): os.unlink(tmp)
