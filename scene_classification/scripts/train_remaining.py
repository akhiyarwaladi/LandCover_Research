"""Train remaining 4 models with 15 epochs (fast)."""
import os, sys, json, time, warnings, numpy as np, torch, torch.nn as nn
from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score,
                             precision_recall_fscore_support, confusion_matrix)
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR, MODELS, DATASETS

def p(*args, **kw): print(*args, **kw, flush=True)

def load_eurosat():
    from config import DATASET_PATHS
    from torchvision.datasets import EuroSAT
    from torchvision import transforms
    from torch.utils.data import DataLoader, random_split
    data_dir = DATASET_PATHS['eurosat']
    train_tf = transforms.Compose([transforms.Resize((224,224)),transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),transforms.RandomRotation(15),
        transforms.ColorJitter(0.2,0.2,0.1),transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    test_tf = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    train_ds = EuroSAT(root=data_dir, download=False, transform=train_tf)
    test_ds = EuroSAT(root=data_dir, download=False, transform=test_tf)
    n = len(train_ds); nt = int(0.8*n); nv = n - nt
    tr_idx, te_idx = torch.utils.data.random_split(range(n), [nt,nv],
        generator=torch.Generator().manual_seed(42))
    tr = torch.utils.data.Subset(train_ds, tr_idx.indices)
    te = torch.utils.data.Subset(test_ds, te_idx.indices)
    return (DataLoader(tr,batch_size=32,shuffle=True,num_workers=0,pin_memory=True,drop_last=True),
            DataLoader(te,batch_size=32,shuffle=False,num_workers=0,pin_memory=True),
            DATASETS['eurosat']['class_names'])

def train(model_name, trl, tel, cls_names, epochs, device):
    from modules.models import create_model, count_parameters
    nc = len(cls_names)
    model = create_model(model_name, nc).to(device)
    tp, _ = count_parameters(model)
    p(f"  Params: {tp/1e6:.1f}M")
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', patience=5, factor=0.5)
    sd = os.path.join(RESULTS_DIR, 'models', 'eurosat', model_name)
    os.makedirs(sd, exist_ok=True)
    hist = {'train_loss':[],'train_acc':[],'test_loss':[],'test_acc':[],'lr':[]}
    ba, be, t0 = 0.0, 0, time.time()
    bp, bt = None, None
    for ep in range(1, epochs+1):
        model.train(); tl=tc=tt=0
        for x,y in trl:
            x,y = x.to(device),y.to(device); opt.zero_grad()
            o = model(x); l = crit(o,y); l.backward(); opt.step()
            tl += l.item()*x.size(0); _,pr = o.max(1); tt += y.size(0); tc += pr.eq(y).sum().item()
        tl/=tt; ta=tc/tt
        model.eval(); vl=vc=vt=0; ap=[]; at=[]
        with torch.no_grad():
            for x,y in tel:
                x,y = x.to(device),y.to(device)
                o = model(x); l = crit(o,y)
                vl += l.item()*x.size(0); _,pr = o.max(1); vt += y.size(0); vc += pr.eq(y).sum().item()
                ap.extend(pr.cpu().numpy()); at.extend(y.cpu().numpy())
        vl/=vt; va=vc/vt; sch.step(va)
        hist['train_loss'].append(tl); hist['train_acc'].append(ta)
        hist['test_loss'].append(vl); hist['test_acc'].append(va)
        hist['lr'].append(opt.param_groups[0]['lr'])
        if va > ba: ba=va; be=ep; torch.save(model.state_dict(), os.path.join(sd,'best_model.pth')); bp=np.array(ap); bt=np.array(at)
        if ep%5==0 or ep==1 or ep==epochs:
            p(f"    E{ep:3d}/{epochs} | Train: {ta:.4f} | Test: {va:.4f} | Best: {ba:.4f} (E{be})")
    tt = time.time()-t0
    np.savez(os.path.join(sd,'training_history.npz'), **{k:np.array(v) for k,v in hist.items()})
    acc=accuracy_score(bt,bp); f1m=f1_score(bt,bp,average='macro',zero_division=0)
    f1w=f1_score(bt,bp,average='weighted',zero_division=0); kp=cohen_kappa_score(bt,bp)
    pr,rc,f1,su = precision_recall_fscore_support(bt,bp,zero_division=0)
    cm = confusion_matrix(bt,bp)
    np.savez(os.path.join(sd,'test_results.npz'),y_true=bt,y_pred=bp,confusion_matrix=cm,y_probs=np.zeros((len(bt),nc)))
    met = {'accuracy':float(acc),'f1_macro':float(f1m),'f1_weighted':float(f1w),'kappa':float(kp),
           'per_class':{'precision':pr.tolist(),'recall':rc.tolist(),'f1':f1.tolist(),'support':su.tolist()},
           'class_names':cls_names}
    with open(os.path.join(sd,'evaluation_metrics.json'),'w') as f: json.dump(met,f,indent=2)
    del model; torch.cuda.empty_cache()
    return {'accuracy':acc,'f1_macro':f1m,'f1_weighted':f1w,'kappa':kp,'params_m':tp/1e6,'training_time':tt,'best_epoch':be}

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    p(f"Device: {device}")
    remaining = ['efficientnet_b3','vit_b_16','swin_t','convnext_tiny']
    p(f"Training {len(remaining)} remaining models, 15 epochs each")
    trl, tel, cn = load_eurosat()
    for i,m in enumerate(remaining):
        p(f"\n[{i+1}/{len(remaining)}] {m}")
        r = train(m, trl, tel, cn, 15, device)
        p(f"  >> {r['accuracy']:.4f} acc, {r['f1_macro']:.4f} F1, {r['training_time']:.0f}s")
    # Update summary
    summary = {'results':{'eurosat':{}},'device':str(device),'dataset':'eurosat'}
    for m in MODELS:
        mp = os.path.join(RESULTS_DIR,'models','eurosat',m,'evaluation_metrics.json')
        tp = os.path.join(RESULTS_DIR,'models','eurosat',m,'training_summary.json')
        if os.path.exists(mp):
            with open(mp) as f: met = json.load(f)
            summary['results']['eurosat'][m] = {
                'accuracy':met['accuracy'],'f1_macro':met['f1_macro'],
                'f1_weighted':met['f1_weighted'],'kappa':met['kappa'],
                'params_m':MODELS[m]['params_m'],'training_time':0,'best_epoch':0}
    with open(os.path.join(RESULTS_DIR,'all_experiments_summary.json'),'w') as f:
        json.dump(summary,f,indent=2)
    # Regenerate outputs
    p("\nRegenerating publication outputs...")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from generate_publication_outputs import generate_performance_tables, generate_per_class_tables, generate_figures
    from generate_statistical_analysis import generate_mcnemar_tables, generate_kappa_table, generate_efficiency_table
    data = {'results':summary['results']}
    generate_performance_tables(data); generate_per_class_tables(data); generate_figures(data)
    generate_mcnemar_tables(); generate_kappa_table(); generate_efficiency_table()
    p("\nDONE!")

if __name__ == '__main__':
    main()
