import argparse,csv,json,math,os,random,time
from pathlib import Path
import numpy as np
import torch
from torch import nn,optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from data.data_loader import TestDataset,TrainDataset
from metric import ssim
from model import Net1

def args():
    p=argparse.ArgumentParser()
    repo_root=Path(__file__).resolve().parent
    p.add_argument("--dataset_root",default=str(repo_root/"dataset"/"UNREAL_NH"))
    p.add_argument("--output_dir",default=str(repo_root/"outputs"/"UNREAL_NH"/"Net1_train_test_only_seed666"))
    p.add_argument("--epochs",type=int,default=500)
    p.add_argument("--batch_size",type=int,default=8)
    p.add_argument("--eval_batch_size",type=int,default=8)
    p.add_argument("--start_lr",type=float,default=2e-4)
    p.add_argument("--end_lr",type=float,default=1e-6)
    p.add_argument("--patience",type=int,default=30)
    p.add_argument("--num_workers",type=int,default=8)
    p.add_argument("--seed",type=int,default=666)
    p.add_argument("--amp",action="store_true")
    return p.parse_args()

def seed_all(seed):
    os.environ["PYTHONHASHSEED"]=str(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    cudnn.benchmark=True; cudnn.deterministic=False

def lr_at(step,total,start,end):
    p=min(max(step/max(total,1),0.0),1.0)
    return end+0.5*(start-end)*(1.0+math.cos(math.pi*p))

def loader(ds,bs,shuffle,workers,drop=False):
    kw={"batch_size":bs,"shuffle":shuffle,"num_workers":workers,"pin_memory":torch.cuda.is_available(),"drop_last":drop}
    if workers>0: kw.update(persistent_workers=True,prefetch_factor=2)
    return DataLoader(ds,**kw)

def metrics(pred,gt):
    pred=pred.float().clamp(0,1); gt=gt.float().clamp(0,1)
    mse=((pred-gt)**2).flatten(1).mean(1)
    p=10*torch.log10(1/mse.clamp_min(1e-12))
    s=ssim(pred,gt,size_average=False)
    return p.detach().cpu(),s.detach().cpu()

@torch.no_grad()
def evaluate(net,dl,device,amp):
    net.eval(); ps=[]; ss=[]; t=time.perf_counter()
    for x,y,_ in dl:
        x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp): z=net(x)
        a,b=metrics(z,y); ps.append(a); ss.append(b)
    return torch.cat(ps).mean().item(),torch.cat(ss).mean().item(),time.perf_counter()-t

def dump(path,obj):
    with open(path,"w",encoding="utf-8") as f: json.dump(obj,f,indent=2,ensure_ascii=False)

def save(path,net,opt,scaler,state):
    x=dict(state); x.update(model=net.state_dict(),optimizer=opt.state_dict(),scaler=scaler.state_dict())
    torch.save(x,path)

def main():
    a=args()
    if a.seed!=666: raise ValueError("This run is fixed to seed=666.")
    out=Path(a.output_dir)
    if out.exists() and any(out.iterdir()): raise FileExistsError("Refusing to overwrite non-empty output: "+str(out))
    for n in ("saved_model","saved_data","saved_infer"): (out/n).mkdir(parents=True,exist_ok=True)
    seed_all(a.seed)
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp=bool(a.amp and dev.type=="cuda")
    root=Path(a.dataset_root)
    tr=TrainDataset(str(root/"train"/"hazy"),str(root/"train"/"clear"))
    te=TestDataset(str(root/"test"/"hazy"),str(root/"test"/"clear"))
    trl=loader(tr,a.batch_size,True,a.num_workers,True)
    tel=loader(te,a.eval_batch_size,False,a.num_workers,False)
    net=Net1(base_dim=32,use_spatial=True,use_frequency=True,use_ffcm=True,use_pa=True,use_deconv=True,fusion_variant="concat").to(dev)
    opt=optim.Adam(net.parameters(),lr=a.start_lr,betas=(.9,.999),eps=1e-8)
    scaler=torch.cuda.amp.GradScaler(enabled=amp); criterion=nn.L1Loss()
    spe=len(trl); total=spe*a.epochs; params=sum(x.numel() for x in net.parameters())
    cfg=vars(a).copy(); cfg.update(device=str(dev),amp_enabled=amp,steps_per_epoch=spe,total_steps=total,train_pairs=len(tr),test_pairs=len(te),parameter_count=params,model={"base_dim":32,"use_spatial":True,"use_frequency":True,"use_ffcm":True,"use_pa":True,"use_deconv":True,"fusion_variant":"concat"},loss={"L1":1.0,"FFT":0.0,"SSIM":0.0,"Gradient":0.0})
    dump(out/"saved_data"/"args.json",cfg)
    best=-float("inf"); bests=0.; be=bs=0; no=0; step=0; hist=[]; pt=pe=0.; start=time.perf_counter()
    lp=out/"saved_data"/"train_log.csv"
    with open(lp,"w",newline="",encoding="utf-8") as lf:
        fields=["epoch","step","learning_rate","train_l1","test_psnr","test_ssim","best_psnr","best_ssim","test_time_sec","peak_train_memory_gib","peak_eval_memory_gib"]
        w=csv.DictWriter(lf,fieldnames=fields); w.writeheader()
        for ep in range(1,a.epochs+1):
            net.train(); total_loss=0.
            if dev.type=="cuda": torch.cuda.reset_peak_memory_stats(dev)
            for x,y in trl:
                step+=1; lr=lr_at(step,total,a.start_lr,a.end_lr)
                for g in opt.param_groups: g["lr"]=lr
                x=x.to(dev,non_blocking=True); y=y.to(dev,non_blocking=True); opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=amp): z=net(x); loss=criterion(z,y)
                if amp: scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                else: loss.backward(); opt.step()
                total_loss+=loss.item()
            if dev.type=="cuda":
                torch.cuda.synchronize(dev); pt=max(pt,torch.cuda.max_memory_allocated(dev)/1024**3)
            p,s,et=evaluate(net,tel,dev,amp)
            if dev.type=="cuda": pe=max(pe,torch.cuda.max_memory_allocated(dev)/1024**3)
            ml=total_loss/max(len(trl),1); improved=p>best
            if improved: best,bests,be,bs,no=p,s,ep,step,0
            else: no+=1
            state={"epoch":ep,"step":step,"best_psnr":best,"best_ssim":bests,"best_epoch":be,"best_step":bs,"no_improve":no,"test_psnr":p,"test_ssim":s,"train_l1":ml,"amp":amp,"args":cfg}
            save(out/"saved_model"/"last.pk",net,opt,scaler,state)
            if improved: save(out/"saved_model"/"best.pk",net,opt,scaler,state)
            row={"epoch":ep,"step":step,"learning_rate":lr,"train_l1":ml,"test_psnr":p,"test_ssim":s,"best_psnr":best,"best_ssim":bests,"test_time_sec":et,"peak_train_memory_gib":pt,"peak_eval_memory_gib":pe}
            w.writerow(row); lf.flush(); hist.append(row)
            print("epoch %d/%d | step %d/%d | L1 %.6f | test PSNR %.4f | SSIM %.4f | best %.4f | no_improve %d"%(ep,a.epochs,step,total,ml,p,s,best,no),flush=True)
            if no>=a.patience:
                print("early_stop at epoch %d, best epoch %d, best PSNR %.4f"%(ep,be,best),flush=True); break
    ck=torch.load(out/"saved_model"/"best.pk",map_location=dev); net.load_state_dict(ck["model"],strict=True)
    p,s,et=evaluate(net,tel,dev,amp)
    result={"best_epoch":be,"best_step":bs,"best_test_psnr_recorded":best,"best_test_ssim_recorded":bests,"final_best_checkpoint_psnr":p,"final_best_checkpoint_ssim":s,"final_test_time_sec":et,"parameter_count":params,"peak_train_memory_gib":pt,"peak_eval_memory_gib":pe,"total_training_time_sec":time.perf_counter()-start,"checkpoint_path":str(out/"saved_model"/"best.pk"),"test_is_used_for_checkpoint_selection":True,"history":hist}
    dump(out/"saved_data"/"test_metrics.json",result); print(json.dumps(result,indent=2,ensure_ascii=False),flush=True)

if __name__=="__main__": main()
