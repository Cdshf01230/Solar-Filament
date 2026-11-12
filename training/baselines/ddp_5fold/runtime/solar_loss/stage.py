"""Stage ALL707 sources; generate only authorized fold crops, preserving annotators."""
import argparse,json,os,shutil,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2,numpy as np
from .common import cpu_budget,atomic

def stage(source,dest,fold):
    source,dest=Path(source),Path(dest);runtime=Path(__file__).resolve().parents[1]
    assignment=json.loads((runtime/'expected_folds.json').read_text());info=assignment['folds'][fold]
    names=sorted(s for f in assignment['folds'] for s in f['train_sources']);assert len(names)==len(set(names))==707
    dest.mkdir(parents=True,exist_ok=True);files=['data/train_images/'+s for s in names]+['data/MAGFiLO_1.0_Annotations_kaggle2026_train.json']
    assert all((source/f).is_file() for f in files)
    def copy(name):
        target=dest/name;target.parent.mkdir(parents=True,exist_ok=True)
        temp=target.with_name(target.name+'.tmp');shutil.copyfile(source/name,temp);os.replace(temp,target)
    with ThreadPoolExecutor(max_workers=min(16,cpu_budget()*4)) as pool:list(pool.map(copy,files))
    from u0_multihead.evaluation import Context
    context=Context(dest);assert sorted(context.records)==names
    records=[json.loads(l) for l in (runtime/f'train_fold{fold}.jsonl').read_text().splitlines()]
    assert {r['source_file'] for r in records}==set(info['train_sources']) and len(records)==info['crops']
    cv2.setNumThreads(1);cropbase=dest/'magfilo_crops_768'
    for name in ('images','masks'):(cropbase/name).mkdir(parents=True,exist_ok=True)
    def generate(name):
        image=cv2.imread(str(dest/'data/train_images'/name),0);assert image is not None and image.shape==(2048,2048)
        own=[r for r in records if r['source_file']==name];targets={}
        for rec in context.records[name]:
            mask=np.zeros((2048,2048),np.uint8)
            for instance in context.masks(rec):mask|=instance.astype(np.uint8)
            targets[str(rec['id'])]=mask
        written=set()
        for r in own:
            y,x=r['y'],r['x'];img=image[y:y+768,x:x+768];mask=targets[str(r['record_id'])][y:y+768,x:x+768]
            assert img.shape==mask.shape==(768,768) and mask.max()<=1
            if r['image'] not in written:assert cv2.imwrite(str(cropbase/r['image']),img);written.add(r['image'])
            assert cv2.imwrite(str(cropbase/r['semantic_mask']),mask)
        return len(own)
    count=0
    with ThreadPoolExecutor(max_workers=min(4,cpu_budget())) as pool:
        for n,number in enumerate(pool.map(generate,info['train_sources']),1):
            count+=number
            if n%20==0:print('CROP707_STAGE',fold,n,len(info['train_sources']),count,flush=True)
    assert count==info['crops']
    shutil.copyfile(runtime/f'train_fold{fold}.jsonl',dest/'train.jsonl')
    atomic(assignment,dest/'fold_assignment.json')
    atomic(dict(passed=True,sources=707,train_sources=len(info['train_sources']),val_sources=len(info['unseen_train_sources']),fold=fold,crops=count,annotation_records_preserved=True),dest/'staging_verification.json')
    print('ALL707_FOLD_STAGE_VERIFIED',fold,len(info['train_sources']),len(info['unseen_train_sources']),count,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--dest',required=True);p.add_argument('--fold',type=int,required=True);a=p.parse_args();stage(a.source,a.dest,a.fold)
