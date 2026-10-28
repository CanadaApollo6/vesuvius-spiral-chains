"""W1: band-to-band pitch (vox) on Kaggle surface-label patches in data/eval/surface_labels_2026_07_sample/kaggle/labels_only/.

  uv run python scripts/w1_kaggle_pitch.py
"""
import json, numpy as np, os, glob, tifffile, cc3d
from scipy import ndimage as ndi
from multiprocessing import Pool
CAP=10
def one(p):
    lab=tifffile.imread(p); fg=lab==1
    e=ndi.distance_transform_edt(fg); ridge=fg&(e>=ndi.maximum_filter(e,3))
    cc=cc3d.connected_components(fg,connectivity=26); ids,sz=np.unique(cc[cc>0],return_counts=True); big=set(ids[sz>2000].tolist())
    v=np.bincount(lab.ravel(),minlength=3); r={"name":os.path.basename(p),"fg":float(v[1]/lab.size),"ignore":float(v[2]/lab.size),"n_sheets":len(big)}
    if len(big)<2: return r, None
    bb=ndi.find_objects(cc); out=[]
    for i in big:
        s=bb[i-1]; s=tuple(slice(max(0,a.start-CAP),min(n,a.stop+CAP)) for a,n in zip(s,cc.shape))
        sub=cc[s]; other=(sub>0)&(sub!=i)&np.isin(sub,list(big))
        if not other.any(): out.append(np.full(int((ridge[s]&(sub==i)).sum()),np.inf)); continue
        d=ndi.distance_transform_edt(~other)
        out.append(d[ridge[s]&(sub==i)])
    P=np.concatenate(out); r|={"n_ridge":int(len(P)),"frac_lt4":float((P<4).mean()),"frac_lt6":float((P<6).mean())}
    return r, P
if __name__=="__main__":
    paths=sorted(glob.glob("data/eval/surface_labels_2026_07_sample/kaggle/labels_only/*.tif"))
    with Pool(6) as pool: res=pool.map(one,paths)
    rows=[r for r,_ in res]; A=np.concatenate([P for _,P in res if P is not None])
    for r in rows: print(r)
    summ={"n_patches":len(rows),"n_with_2plus_sheets":sum(P is not None for _,P in res),"frac_ridge_pitch_lt4vox":float((A<4).mean()),
          "frac_ridge_pitch_lt6vox":float((A<6).mean()),"patches_with_lt4_ge5pct":sum(r.get("frac_lt4",0)>=0.05 for r in rows),
          "patches_with_lt4_ge1pct":sum(r.get("frac_lt4",0)>=0.01 for r in rows),
          "mean_fg":float(np.mean([r["fg"] for r in rows])),"mean_ignore":float(np.mean([r["ignore"] for r in rows])),
          "note":"pitch = distance from a band ridge voxel to nearest voxel of another label component (sheet), capped at 10 vox; voxel units"}
    print(summ); json.dump({"rows":rows,"summary":summ},open("data/eval/surface_labels_2026_07_sample/kaggle_pitch40.json","w"),indent=1)
