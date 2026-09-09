from __future__ import annotations
import csv
from pathlib import Path
FIELDS=['run_id','timestamp','git_commit','dataset','split','sample_count','model','model_revision','precision','quantization','retrieval_method','knowledge_sources','top_k','filtering_method','verifier','calibration','threshold','seed','accuracy','coverage','selective_accuracy','selective_risk','ECE','Brier','AURC','latency','peak_vram','status','error_message']
def append_experiment(path: str | Path, row: dict) -> None:
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); exists=path.exists() and path.stat().st_size>0
    with path.open('a',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=FIELDS,extrasaction='ignore')
        if not exists: wr.writeheader()
        wr.writerow({k:row.get(k,'') for k in FIELDS})
