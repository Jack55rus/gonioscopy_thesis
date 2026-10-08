from itertools import combinations,product
from pathlib import Path
import csv,json
import numpy as np
from skimage.morphology import remove_small_objects,binary_erosion,disk
import config


RAW_DIR=config.OUTPUT_DIR / "raw_predictions" / config.TEST_SPLIT
OUTPUT_DIR=config.OUTPUT_DIR / "ensemble_research"
TM_CLASS_ID=1
THRESHOLDS=[.3,.4,.5,.6,.7,.8]
AGGREGATIONS=["mean","max","median"]
TTA_SETS=[("none",),("none","hflip"),("none","vflip"),("none","hflip","vflip"),("none","hflip","vflip","hvflip")]

MIN_OBJECT_SIZES=[0,config.POSTPROCESS_MIN_OBJECT_SIZE]
EROSION_RADII=[0,1]

def softmax(x):
    x=x.astype(np.float32); x-=x.max(0,keepdims=True); e=np.exp(x); return e/e.sum(0,keepdims=True)

def agg(xs,method):
    x=np.stack(xs)
    return x.mean(0) if method=="mean" else x.max(0) if method=="max" else np.median(x,0)

def post(pred,min_size,erosion):
    pred=pred.astype(bool)
    if min_size:
        pred=remove_small_objects(pred,min_size=min_size, connectivity=2)
    if erosion:
        pred=binary_erosion(pred,footprint=disk(erosion))
    return pred
def counts(pred,target):

    valid=target!=config.IGNORE_INDEX; gt=(target==TM_CLASS_ID)&valid; pred=pred&valid
    return int((pred&gt).sum()),int((pred&~gt&valid).sum()),int((~pred&gt&valid).sum()),int((~pred&~gt&valid).sum())

def metrics(tp,fp,fn,tn):
    return {"dice":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,
            "iou":tp/(tp+fp+fn) if tp+fp+fn else 1.,
            "precision":tp/(tp+fp) if tp+fp else 0.,
            "recall":tp/(tp+fn) if tp+fn else 0.,
            "specificity":tn/(tn+fp) if tn+fp else 0.}

def main():
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    names=(RAW_DIR/"names.txt").read_text().splitlines()
    models=sorted(p.name for p in RAW_DIR.iterdir() if p.is_dir() and p.name!="targets")
    targets={n:np.load(RAW_DIR/"targets"/f"{n}.npy") for n in names}
    cache={}
    def prob(m,n,ttas):
        k=(m,n,ttas)
        if k not in cache: cache[k]=np.mean([softmax(np.load(RAW_DIR/m/t/f"{n}.npy")) for t in ttas],0)
        return cache[k]
    combos=[c for r in range(1,len(models)+1) for c in combinations(models,r)]
    rows=[]
    configs=list(product(combos,AGGREGATIONS,THRESHOLDS,MIN_OBJECT_SIZES,EROSION_RADII,TTA_SETS))
    for i,(members,a,thr,ms,er,ttas) in enumerate(configs,1):
        total=np.zeros(4,dtype=np.int64); image_dice=[]
        for n in names:
            pred=post(agg([prob(m,n,ttas) for m in members],a)[TM_CLASS_ID]>thr,ms,er)
            c=counts(pred,targets[n]); total+=c; image_dice.append(metrics(*c)["dice"])
        met=metrics(*total)
        rows.append({"models":"+".join(members),"n_models":len(members),"aggregation":a,"threshold":thr,"min_size":ms,"erosion":er,"tta":"+".join(ttas),**met,"mean_image_dice":float(np.mean(image_dice)),"std_image_dice":float(np.std(image_dice)),"tp":int(total[0]),"fp":int(total[1]),"fn":int(total[2]),"tn":int(total[3])})
        print(f"[{i}/{len(configs)}] Dice={met['dice']:.4f}")
    rows.sort(key=lambda r:(r["dice"],r["mean_image_dice"]),reverse=True)
    with open(OUTPUT_DIR/"results.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    best=rows[0]; singles=[r for r in rows if r["n_models"]==1]; ensembles=[r for r in rows if r["n_models"]>1]
    best_single=max(singles,key=lambda r:r["dice"]); best_ensemble=max(ensembles,key=lambda r:r["dice"]) if ensembles else None
    lines=["# Ensemble research report",f"Samples: {len(names)} | Models: {len(models)} | Experiments: {len(rows)}","## Best overall","```json",json.dumps(best,indent=2),"```","## Best individual networks"]

    for m in models:
        r=max((x for x in singles if x["models"]==m),key=lambda x:x["dice"])
        lines.append(f"- {m}: Dice={r['dice']:.4f}, IoU={r['iou']:.4f}, P={r['precision']:.4f}, R={r['recall']:.4f}, threshold={r['threshold']}, TTA={r['tta']}")
    lines+=["## Best ensemble","```json",json.dumps(best_ensemble,indent=2) if best_ensemble else "N/A","```"]
    if best_ensemble:
        lines.append(f"Gain over best single: {best_ensemble['dice']-best_single['dice']:+.4f} Dice.")
    lines+=["## Error analysis",f"Best: TP={best['tp']}, FP={best['fp']}, FN={best['fn']}, TN={best['tn']}.",("Errors are FP-heavy; investigate stricter thresholds/post-processing." if best["fp"]>best["fn"] else "Errors are FN-heavy; investigate lower thresholds/max aggregation."),"## Top 25 configurations"]
    for i,r in enumerate(rows[:25],1):
        lines.append(f"{i}. Dice={r['dice']:.4f}, image Dice={r['mean_image_dice']:.4f}±{r['std_image_dice']:.4f}, IoU={r['iou']:.4f}, P={r['precision']:.4f}, R={r['recall']:.4f} | {r['models']} | {r['aggregation']} | t={r['threshold']} | TTA={r['tta']} | min={r['min_size']} | erosion={r['erosion']}")
    lines+=["## Methodological note","Select networks, aggregation, threshold, TTA and post-processing only on validation data. Evaluate the chosen final configuration once on the untouched test set."]
    (OUTPUT_DIR/"report.md").write_text("\n".join(lines))
    print(f"Best Dice: {best['dice']:.4f}\nResults: {OUTPUT_DIR/'results.csv'}\nReport: {OUTPUT_DIR/'report.md'}")


if __name__=="__main__":
    main()
