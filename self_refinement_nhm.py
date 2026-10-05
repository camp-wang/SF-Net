import argparse
import copy
import csv
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import RandomCrop, ToTensor
from torchvision.transforms.functional import crop


def parse_args():
    repo_root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument('--code_root', default=str(repo_root))
    p.add_argument('--nhm_root', default=str(repo_root / 'dataset' / 'NHM'))
    p.add_argument('--rwnhc_root', default=str(repo_root / 'dataset' / 'RWNHC_MM23'))
    p.add_argument('--nhrw_root', default=str(repo_root / 'dataset' / 'NHRW' / 'NHRW'))
    p.add_argument('--real_nh_root', default=str(repo_root / 'dataset' / 'REAL-NH_50'))
    p.add_argument('--nhm_checkpoint', default='')
    p.add_argument('--musiq_weights', default=str(repo_root / 'weights' / 'musiq_ava_ckpt-e8d3f067.pth'))
    p.add_argument('--niqe_weights', default=str(repo_root / 'weights' / 'niqe_modelparameters.mat'))
    p.add_argument('--output_dir', default=str(repo_root / 'outputs' / 'self-refinement'))
    p.add_argument('--source_batch', type=int, default=8)
    p.add_argument('--target_batch', type=int, default=1)
    p.add_argument('--patch_micro_batch', type=int, default=1)
    p.add_argument('--grad_accum', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--pseudo_weight', type=float, default=0.2)
    p.add_argument('--consistency_weight', type=float, default=0.05)
    p.add_argument('--ema', type=float, default=0.999)
    p.add_argument('--max_steps', type=int, default=3000)
    p.add_argument('--eval_interval', type=int, default=250)
    p.add_argument('--patience', type=int, default=3)
    p.add_argument('--patch_size', type=int, default=224)
    p.add_argument('--stride', type=int, default=16)
    p.add_argument('--confidence_threshold', type=float, default=0.005)
    p.add_argument('--seed', type=int, default=666)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--smoke_test', action='store_true')
    return p.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def import_project(code_root):
    if code_root not in sys.path:
        sys.path.insert(0, code_root)
    from data.data_loader import TrainDataset
    from model import Net1
    return TrainDataset, Net1


class TargetDataset(Dataset):
    def __init__(self, root, indices=None, crop_size=256, random_crop=True):
        self.files = sorted(p for p in Path(root).iterdir() if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'})
        self.indices = list(range(len(self.files))) if indices is None else list(indices)
        self.crop_size = crop_size
        self.random_crop = random_crop
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        path = self.files[self.indices[i]]
        image = Image.open(path).convert('RGB')
        if self.crop_size is not None:
            w, h = image.size
            if min(w, h) < self.crop_size:
                scale = self.crop_size / min(w, h)
                image = image.resize((int(round(w * scale)), int(round(h * scale))), Image.BICUBIC)
            if self.random_crop:
                top, left, height, width = RandomCrop.get_params(image, [self.crop_size, self.crop_size])
            else:
                top, left = (image.height - self.crop_size) // 2, (image.width - self.crop_size) // 2
                height = width = self.crop_size
            image = crop(image, top, left, height, width)
        return self.to_tensor(image), path.name


class EvalImageDataset(Dataset):
    def __init__(self, root):
        self.files = sorted(p for p in Path(root).rglob('*') if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'})
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        p = self.files[i]
        return self.to_tensor(Image.open(p).convert('RGB')), p.name


def build_net(Net1, device):
    return Net1(base_dim=32, use_spatial=True, use_frequency=True, use_ffcm=True,
                use_pa=True, use_deconv=True, fusion_variant='concat').to(device)


def find_checkpoint(args):
    if args.nhm_checkpoint:
        return Path(args.nhm_checkpoint)
    experiment_root = Path(args.code_root) / 'outputs' / 'experiment'
    candidates = sorted(experiment_root.glob('NHM/*/saved_model/best.pk'))
    preferred = [p for p in candidates if 'NHM_lr1e-4_fft0_concat' in str(p)]
    if not preferred:
        raise FileNotFoundError(f'NHM best.pk not found under {experiment_root / "NHM"}')
    return preferred[0]


def load_checkpoint(net, path, device):
    payload = torch.load(path, map_location=device)
    state = payload.get('model', payload) if isinstance(payload, dict) else payload
    state = {k[7:] if k.startswith('module.') else k: v for k, v in state.items()}
    net.load_state_dict(state, strict=True)
    return payload


def pad_multiple(x, multiple=4):
    h, w = x.shape[-2:]
    ph, pw = (multiple - h % multiple) % multiple, (multiple - w % multiple) % multiple
    return F.pad(x, (0, pw, 0, ph), mode='reflect'), h, w


def forward_original(net, x):
    padded, h, w = pad_multiple(x)
    return net(padded)[..., :h, :w]


def overlap_predictions(teacher, image, patch_size, stride, micro_batch):
    b, c, h, w = image.shape
    if b != 1 or h < patch_size or w < patch_size:
        raise ValueError('target image must be one 256x256 crop for overlap prediction')
    ys = list(range(0, h - patch_size + 1, stride))
    xs = list(range(0, w - patch_size + 1, stride))
    if ys[-1] != h - patch_size:
        ys.append(h - patch_size)
    if xs[-1] != w - patch_size:
        xs.append(w - patch_size)
    patches, locations = [], []
    for y in ys:
        for x in xs:
            patches.append(image[..., y:y + patch_size, x:x + patch_size])
            locations.append((y, x))
    mean = torch.zeros_like(image)
    second = torch.zeros_like(image)
    count = torch.zeros((1, 1, h, w), device=image.device)
    with torch.no_grad():
        for start in range(0, len(patches), micro_batch):
            pred = teacher(torch.cat(patches[start:start + micro_batch], dim=0)).clamp(0, 1)
            for j in range(pred.shape[0]):
                y, x = locations[start + j]
                q = pred[j:j + 1]
                mean[..., y:y + patch_size, x:x + patch_size] += q
                second[..., y:y + patch_size, x:x + patch_size] += q * q
                count[..., y:y + patch_size, x:x + patch_size] += 1
    mean = mean / count.clamp_min(1)
    variance = (second / count.clamp_min(1) - mean * mean).clamp_min(0).mean(1, keepdim=True)
    return mean, variance


def masked_l1(pred, target, mask):
    denom = mask.sum() * pred.shape[1]
    if denom.item() == 0:
        return pred.sum() * 0
    return (torch.abs(pred - target) * mask).sum() / denom


def clone_state(obj):
    return copy.deepcopy(obj.state_dict())


def restore_state(obj, state):
    obj.load_state_dict(state)


@torch.no_grad()
def evaluate_iqa(net, loader, device, niqe, musiq, output_dir=None):
    net.eval()
    niqe_values, musiq_values, rows = [], [], []
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    from torchvision.utils import save_image
    for image, names in loader:
        image = image.to(device)
        pred = forward_original(net, image).clamp(0, 1)
        for pred_one, name in zip(pred, names):
            one = pred_one.unsqueeze(0)
            niqe_value = float(niqe(one).mean().item())
            musiq_value = float(musiq(one).mean().item())
            niqe_values.append(niqe_value)
            musiq_values.append(musiq_value)
            if output_dir:
                save_path = output_dir / (Path(name).stem + '.png')
                save_image(pred_one.cpu(), str(save_path))
                rows.append({'filename': name, 'path': str(save_path), 'niqe': niqe_value, 'musiq_ava': musiq_value})
    result = {'niqe_mean': float(np.mean(niqe_values)), 'musiq_ava_mean': float(np.mean(musiq_values)), 'count': len(niqe_values)}
    if output_dir:
        with open(output_dir / 'per_image_metrics.json', 'w') as f:
            json.dump(rows, f, indent=2)
    return result


def quality_score(metrics, baseline):
    return 0.5 * (metrics['musiq_ava_mean'] - baseline['musiq_ava_mean']) / max(abs(baseline['musiq_ava_mean']), 1e-6) - 0.5 * (metrics['niqe_mean'] - baseline['niqe_mean']) / max(abs(baseline['niqe_mean']), 1e-6)


def save_checkpoint(path, student, teacher, optimizer, state):
    payload = dict(state)
    payload.update(model=student.state_dict(), student=student.state_dict(), teacher=teacher.state_dict(), optimizer=optimizer.state_dict())
    torch.save(payload, path)


def main():
    args = parse_args()
    if args.smoke_test:
        args.max_steps = 2
        args.eval_interval = 2
        args.patience = 1
    seed_all(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    TrainDataset, Net1 = import_project(args.code_root)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = find_checkpoint(args)
    student = build_net(Net1, device)
    load_checkpoint(student, checkpoint, device)
    teacher = build_net(Net1, device)
    teacher.load_state_dict(student.state_dict(), strict=True)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    optimizer = optim.Adam(student.parameters(), lr=args.lr)
    source = TrainDataset(str(Path(args.nhm_root) / 'train' / 'hazy'), str(Path(args.nhm_root) / 'train' / 'clear'))
    all_target = TargetDataset(args.rwnhc_root, crop_size=256)
    order = np.random.RandomState(args.seed).permutation(len(all_target))
    split = max(1, int(round(len(order) * 0.8)))
    train_target = Subset(all_target, order[:split].tolist())
    hold_target = TargetDataset(args.rwnhc_root, order[split:].tolist(), crop_size=256, random_crop=False)
    source_loader = DataLoader(source, batch_size=args.source_batch, shuffle=True, num_workers=args.num_workers, drop_last=True, pin_memory=True)
    target_loader = DataLoader(train_target, batch_size=args.target_batch, shuffle=True, num_workers=args.num_workers, drop_last=True, pin_memory=True)
    hold_loader = DataLoader(hold_target, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    import pyiqa
    # pyiqa 0.1.16 passes weights_only to torch.load, while this server's
    # PyTorch predates that keyword. Keep the compatibility shim local to this
    # runner instead of modifying the installed third-party package.
    _torch_load = torch.load
    def _compat_torch_load(*load_args, **load_kwargs):
        load_kwargs.pop('weights_only', None)
        return _torch_load(*load_args, **load_kwargs)
    torch.load = _compat_torch_load
    niqe = pyiqa.create_metric('niqe', device=device, pretrained_model_path=args.niqe_weights)
    musiq = pyiqa.create_metric('musiq-ava', device=device, num_class=10, pretrained_model_path=args.musiq_weights)
    niqe.eval(); musiq.eval()
    probe = TargetDataset(args.rwnhc_root, order[:min(50, split)].tolist(), crop_size=256, random_crop=False)
    probe_loader = DataLoader(probe, batch_size=1, shuffle=False, num_workers=args.num_workers)
    mask_ratios = []
    student.eval()
    with torch.no_grad():
        for image, _ in probe_loader:
            image = image.to(device)
            _, var = overlap_predictions(teacher, image, args.patch_size, args.stride, args.patch_micro_batch)
            mask_ratios.append(float((var < args.confidence_threshold).float().mean().item()))
    mask_ratio_mean = float(np.mean(mask_ratios))
    actual_threshold = args.confidence_threshold
    if mask_ratio_mean < 0.10 or mask_ratio_mean > 0.80:
        all_var = []
        with torch.no_grad():
            for image, _ in probe_loader:
                _, var = overlap_predictions(teacher, image.to(device), args.patch_size, args.stride, args.patch_micro_batch)
                all_var.append(var.flatten().cpu())
        actual_threshold = float(torch.cat(all_var).quantile(0.30).item())
    with open(out / 'mask_calibration.json', 'w') as f:
        json.dump({'requested_threshold': args.confidence_threshold, 'actual_threshold': actual_threshold, 'mean_mask_ratio': mask_ratio_mean}, f, indent=2)
    baseline = evaluate_iqa(student, hold_loader, device, niqe, musiq)
    best_score = quality_score(baseline, baseline)
    before_score = best_score
    save_checkpoint(out / 'best.pk', student, teacher, optimizer, {'step': 0, 'quality_score': best_score, 'metrics': baseline})
    source_iter, target_iter = iter(source_loader), iter(target_loader)
    block_student = clone_state(student)
    block_teacher = clone_state(teacher)
    block_optimizer = copy.deepcopy(optimizer.state_dict())
    rows = []
    no_improve = 0
    accum = 0
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, args.max_steps + 1):
        student.train()
        try:
            hazy, clear = next(source_iter)
        except StopIteration:
            source_iter = iter(source_loader); hazy, clear = next(source_iter)
        try:
            target, _ = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader); target, _ = next(target_iter)
        hazy, clear, target = hazy.to(device), clear.to(device), target.to(device)
        with torch.no_grad():
            pseudo, var = overlap_predictions(teacher, target, args.patch_size, args.stride, args.patch_micro_batch)
            mask = (var < actual_threshold).float()
        out_source = student(hazy)
        out_target = forward_original(student, target)
        flipped = torch.flip(target, dims=[3])
        out_flip = torch.flip(forward_original(student, flipped), dims=[3])
        loss_source = F.l1_loss(out_source, clear)
        loss_pseudo = masked_l1(out_target, pseudo, mask)
        loss_consistency = masked_l1(out_target, out_flip.detach(), mask)
        loss = loss_source + args.pseudo_weight * loss_pseudo + args.consistency_weight * loss_consistency
        if not torch.isfinite(loss):
            raise FloatingPointError('non-finite loss at step {}'.format(step))
        (loss / args.grad_accum).backward()
        accum += 1
        if accum == args.grad_accum:
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step(); optimizer.zero_grad(set_to_none=True); accum = 0
            with torch.no_grad():
                for tp, sp in zip(teacher.parameters(), student.parameters()):
                    tp.mul_(args.ema).add_(sp, alpha=1.0 - args.ema)
        if step % args.eval_interval == 0 or step == args.max_steps:
            metrics = evaluate_iqa(student, hold_loader, device, niqe, musiq)
            score = quality_score(metrics, baseline)
            accepted = score >= before_score
            rows.append({'step': step, 'quality_score': score, 'accepted': accepted, **metrics, 'loss': float(loss.item()), 'mask_ratio': float(mask.mean().item())})
            if not accepted:
                restore_state(student, block_student)
                restore_state(teacher, block_teacher)
                optimizer.load_state_dict(block_optimizer)
                current_score = before_score
                no_improve += 1
            else:
                before_score = score
                current_score = score
                if score > best_score:
                    best_score = score; no_improve = 0
                    save_checkpoint(out / 'best.pk', student, teacher, optimizer, {'step': step, 'quality_score': score, 'metrics': metrics})
                else:
                    no_improve += 1
            block_student = clone_state(student)
            block_teacher = clone_state(teacher)
            block_optimizer = copy.deepcopy(optimizer.state_dict())
            with open(out / 'train_log.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=sorted(rows[0].keys())); writer.writeheader(); writer.writerows(rows)
            save_checkpoint(out / 'last.pk', student, teacher, optimizer, {'step': step, 'quality_score': score, 'metrics': metrics})
            if no_improve >= args.patience:
                break
    final_summary = {'baseline_rwnhc_holdout': baseline, 'best_quality_score': best_score}
    if not args.smoke_test:
        best_payload = torch.load(out / 'best.pk', map_location=device)
        student.load_state_dict(best_payload['model'], strict=True)
        for name, root in [('NHRW', args.nhrw_root), ('Real-NH', args.real_nh_root)]:
            dataset = EvalImageDataset(root)
            loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
            metrics = evaluate_iqa(student, loader, device, niqe, musiq, out / 'outputs' / 'best' / name)
            final_summary[name] = metrics
        with open(out / 'evaluation_summary.json', 'w') as f:
            json.dump(final_summary, f, indent=2)
    print(json.dumps({'output_dir': str(out), 'best_score': best_score, 'baseline': baseline, 'steps': rows[-1]['step'] if rows else 0, 'final': final_summary}, indent=2))


if __name__ == '__main__':
    main()
