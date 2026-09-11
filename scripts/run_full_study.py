#!/usr/bin/env python
"""Resumable A-OKVQA study. --plan performs no inference or downloads."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from paper4_kbvqa.execution.study import write_json, freeze_json, read_rows, validate_predictions

MODELS={'generator':'Qwen/Qwen2.5-VL-3B-Instruct','qwen':'Qwen/Qwen2.5-1.5B-Instruct',
        'deberta':'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli',
        'reranker':'sentence-transformers/all-MiniLM-L6-v2'}


def experiment_matrix():
    runs=[{'name':v,'variant':v,'options':[]} for v in ['B0','B1','B2','B3']]
    for source in ['wikipedia','wikidata','conceptnet']:
        runs.append({'name':'source_'+source,'variant':'B3','options':['--sources',source]})
    for k in [1,3,10]:
        runs.append({'name':f'topk_{k}','variant':'B3','options':['--filtered-top-k',str(k)]})
    for name,options in [('nli',['--critic-backend','deberta']),('semantic',['--filter-backend','semantic']),
                         ('no_provenance',['--provenance-weight','0']),
                         ('no_contradiction_penalty',['--contradiction-penalty','0'])]:
        runs.append({'name':name,'variant':'B3','options':options})
    for spec in ['drop:wikipedia','drop:wikidata','drop:conceptnet','scarcity:0','scarcity:1','scarcity:3',
                 'ranking:0.5','ranking:1.0','contradiction:1','contradiction:3','cross_question:1','cross_question:3']:
        for v in ['B1','B3']:
            runs.append({'name':v+'_'+spec.replace(':','_').replace('.','p'),
                         'variant':v,'options':['--corruption',spec],'corruption':True})
    return runs


class Study:
    def __init__(self,args):
        self.args=args; self.out=Path(args.out).resolve(); self.out.mkdir(parents=True,exist_ok=True)
        self.state_path=self.out/'progress.json'
        self.state=json.loads(self.state_path.read_text()) if self.state_path.exists() else {'steps':{}}
        self.model_lock={}

    def command(self,name,script,*args,outputs=()):
        command=[sys.executable,str(ROOT/'scripts'/script),*map(str,args)]
        inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                for arg in command[1:] if (p:=Path(arg)).is_file() and p not in map(Path,outputs)}
        digest=hashlib.sha256(json.dumps({'command':command,'inputs':inputs},sort_keys=True).encode()).hexdigest()
        old=self.state['steps'].get(name,{})
        if outputs and old.get('status')=='COMPLETED' and old.get('command_sha256')==digest:
            if all(Path(p).exists() and hashlib.sha256(Path(p).read_bytes()).hexdigest()==old.get('outputs',{}).get(str(p)) for p in outputs):
                print('[RESUME]',name,flush=True); return
        self.state['steps'][name]={'status':'RUNNING','command':command,'command_sha256':digest}
        write_json(self.state_path,self.state)
        log=self.out/'logs'/f'{name}.log'; log.parent.mkdir(exist_ok=True)
        print('[RUN]',name,flush=True)
        with log.open('a') as f:
            proc=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,
                                  env={**os.environ,'PYTHONUNBUFFERED':'1','PYTHONHASHSEED':str(self.args.seed)})
            try:
                for line in proc.stdout:
                    print(line,end='',flush=True); f.write(line); f.flush()
                code=proc.wait()
            except BaseException:
                proc.terminate(); proc.wait(); raise
        step=self.state['steps'][name]; step['returncode']=code
        step['status']='COMPLETED' if code==0 and all(Path(p).exists() for p in outputs) else 'FAILED'
        if step['status']=='COMPLETED': step['outputs']={str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in outputs}
        write_json(self.state_path,self.state)
        if step['status']!='COMPLETED': raise RuntimeError(f'{name} failed. See {log}. Rerun after correction.')

    def predict(self,name,manifest,variant='B3',options=()):
        path=self.out/'predictions'/f'{name}.jsonl'
        backend='deberta' if 'deberta' in options else 'qwen'
        args=['--manifest',manifest,'--variant',variant,'--out',path,
              '--checkpoint',self.out/'checkpoints'/f'{name}.json',
              '--cache-dir',self.out/'cache/knowledge','--seed',self.args.seed,
              '--max-pixels',self.args.max_pixels,
              '--generator-revision',self.model_lock['generator']['revision'],
              '--critic-model',MODELS[backend],'--critic-revision',self.model_lock[backend]['revision'],
              '--reranker-revision',self.model_lock['reranker']['revision'],*options]
        if '--corruption' in options:
            args+=['--distractor-predictions',self.out/'predictions/aokvqa_cal_B3.jsonl']
        self.command('predict_'+name,'run_variants.py',*args,
                     outputs=[path,path.with_suffix('.identity.json'),path.with_suffix('.sessions.jsonl')])
        validate_predictions(manifest,path)
        return path

    def policy(self,name,manifest,predictions,risk=.05):
        path=self.out/'policies'/f'{name}.json'
        self.command('policy_'+name,'fit_selective_policy.py','--manifest',manifest,'--predictions',predictions,
                     '--target-risk',risk,'--out',path,outputs=[path])
        return path

    def evaluate(self,name,manifest,predictions,policy=None,selective=False):
        out=self.out/'metrics'/f'{name}.json'
        extra=[]; field='raw_confidence'
        if policy:
            applied=self.out/'predictions'/(f'{name}.jsonl' if name in {'aokvqa_val_B4','aokvqa_val_B5'} else f'{name}_calibrated.jsonl')
            self.command('apply_'+name,'apply_selective_policy.py','--policy',policy,'--predictions',predictions,
                         '--out',applied,outputs=[applied])
            predictions=applied; field='calibrated_confidence'
            if selective: extra=['--threshold',json.loads(policy.read_text())['threshold']]
        self.command('evaluate_'+name,'evaluate.py','--manifest',manifest,'--predictions',predictions,
                     '--dataset','aokvqa','--confidence-field',field,*extra,'--out',out,outputs=[out])
        return out

    def run(self):
        import torch
        if not torch.cuda.is_available(): raise RuntimeError('Enable a CUDA GPU in Colab before running. No results generated.')
        identity={'cal_n':self.args.cal_n,'eval_max':self.args.eval_max,'seed':self.args.seed,
                  'max_pixels':self.args.max_pixels,'scope':self.args.scope,'matrix':experiment_matrix(),
                  'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
        freeze_json(self.out/'study_identity.json',identity)
        lock=self.out/'model_revisions.json'
        if lock.exists(): self.model_lock=json.loads(lock.read_text())
        else:
            from huggingface_hub import HfApi
            self.model_lock={k:{'model_id':v,'revision':HfApi().model_info(v).sha} for k,v in MODELS.items()}
            write_json(lock,self.model_lock)
        (self.out/'requirements-resolved.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
        data=self.out/'datasets'; aok=data/'aokvqa'; coco=data/'coco'; man=self.out/'manifests'
        man.mkdir(exist_ok=True)
        self.command('annotations','prepare_data.py','--dataset','aokvqa','--out',data,
                     outputs=[aok/'aokvqa_v1p0_train.json',aok/'aokvqa_v1p0_val.json'])
        for split in ['train','val']:
            self.command('manifest_'+split,'build_manifest.py','aokvqa','--aokvqa-dir',aok,'--coco-dir',coco,
                         '--split',split,'--out',man/f'{split}.jsonl',outputs=[man/f'{split}.jsonl'])
        train=read_rows(man/'train.jsonl'); val=read_rows(man/'val.jsonl')
        random.Random(self.args.seed).shuffle(train)
        if len(train)<self.args.cal_n: raise ValueError('Insufficient calibration samples')
        cal_rows=train[:self.args.cal_n]
        if self.args.eval_max: val=val[:self.args.eval_max]
        if {x['question_id'] for x in cal_rows}&{x['question_id'] for x in val}: raise ValueError('Split leakage')
        if {x['metadata']['image_id'] for x in cal_rows}&{x['metadata']['image_id'] for x in val}: raise ValueError('Image leakage')
        cal=man/'calibration.jsonl'; evaluation=man/'evaluation.jsonl'; dev=man/'development.jsonl'
        for path,rows in [(cal,cal_rows),(evaluation,val),(dev,cal_rows[:50])]:
            content=''.join(json.dumps(x)+'\n' for x in rows)
            if path.exists() and path.read_text()!=content: raise ValueError('Frozen manifest changed')
            path.write_text(content)
        write_json(self.out/'dataset_audit.json',{'dataset':'aokvqa','calibration_split':'train',
            'evaluation_split':'val','calibration_n':len(cal_rows),'evaluation_n':len(val),
            'development':self.args.eval_max is not None,'seed':self.args.seed})
        # Downloads validate every image; do not skip just because a directory exists.
        self.command('images_cal','download_coco_images.py','--manifest',cal,'--out-dir',coco,'--split','train2017')
        self.command('images_eval','download_coco_images.py','--manifest',evaluation,'--out-dir',coco,'--split','val2017')
        for rows in [cal_rows,val]:
            from PIL import Image
            for row in rows:
                with Image.open(row['image_path']) as image: image.verify()
        for v in ['B0','B3']: self.predict('development_'+v,dev,v)
        cal_pred=self.predict('aokvqa_cal_B3',cal)
        policy=self.policy('aokvqa_policy_5pct',cal,cal_pred)
        main_preds={}
        for v in ['B0','B1','B2','B3']:
            main_preds[v]=self.predict('aokvqa_val_'+v,evaluation,v)
            self.evaluate('aokvqa_val_'+v,evaluation,main_preds[v])
        self.evaluate('aokvqa_val_B4',evaluation,main_preds['B3'],policy)
        self.evaluate('aokvqa_val_B5',evaluation,main_preds['B3'],policy,True)
        summary=self.out/'metrics/aokvqa_B0_B5_summary.json'
        arguments=[]
        for v in ['B0','B1','B2','B3','B4','B5']: arguments+=['--'+v.lower(),self.out/f'metrics/aokvqa_val_{v}.json']
        self.command('summary','summarize_variants.py',*arguments,'--out',summary,outputs=[summary])
        for risk in [.01,.10,.20]:
            p=self.policy(f'risk_{risk}',cal,cal_pred,risk)
            self.evaluate(f'target_risk_{risk}',evaluation,main_preds['B3'],p,True)
        if self.args.scope=='extended':
            # Baseline corruption uses its own clean-data calibration; never
            # apply the B3 calibrator to a different confidence definition.
            cal_b1=self.predict('aokvqa_cal_B1',cal,'B1')
            b1_policy=self.policy('aokvqa_B1_policy_5pct',cal,cal_b1)
            for spec in experiment_matrix()[4:]:
                name=spec['name']; v=spec['variant']; options=spec['options']
                p=policy if v=='B3' else b1_policy
                if not spec.get('corruption'):
                    cp=self.predict('cal_'+name,cal,v,options)
                    p=self.policy(name,cal,cp)
                pred=self.predict(name,evaluation,v,options)
                self.evaluate(name+'_raw',evaluation,pred)
                self.evaluate(name,evaluation,pred,p,True)
        report=self.out/'report/study_report.json'
        self.command('report','report_study.py','--results',self.out,outputs=[report])
        write_json(self.out/'completion.json',{'status':'COMPLETED','scope':self.args.scope,
            'dataset':'aokvqa','development':self.args.eval_max is not None,
            'manuscript_ready':False,'remaining':['Official OK-VQA evaluation','External protocol-matched baselines',
            'Independent evidence-support assessment','Scientific review of realised results'],
            'note':'5% target is an empirical calibration operating point, not a held-out risk guarantee.'})
        print('Completed requested A-OKVQA scope. See completion.json for remaining paper requirements.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default='results/full_study')
    p.add_argument('--scope',choices=['main','extended'],default='extended')
    p.add_argument('--cal-n',type=int,default=1000)
    p.add_argument('--eval-max',type=int)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--max-pixels',type=int,default=1003520)
    p.add_argument('--plan',action='store_true')
    args=p.parse_args()
    if args.cal_n<50 or (args.eval_max is not None and args.eval_max<1): p.error('cal-n >= 50; eval-max must be positive')
    if args.plan:
        print(json.dumps({'scope':args.scope,'experiments':experiment_matrix() if args.scope=='extended' else experiment_matrix()[:4],
                         'additional':'development gate, calibration, risk sensitivity, statistics and figures'},indent=2)); return
    Study(args).run()

if __name__=='__main__': main()
