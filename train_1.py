import torch
import torch.nn as nn
import torch.nn.functional as F
import os, time, math
import numpy as np

import torch
import torch.nn.functional as F
from torch import optim, nn
from torch.backends import cudnn
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from logger import plot_loss_log, plot_psnr_log
from metric import psnr, ssim
from model import Net1  # 你的双域改进版网络

from option_train import opt
from data.data_loader import TrainDataset, TestDataset


class FFTLoss(nn.Module):
    """标准FFT损失 - 直接在频域计算损失

    方法：对FFT结果的实部和虚部直接计算L1/L2损失
    优点：
    - 数值稳定（无对数压缩导致的梯度问题）
    - 梯度流畅（可直接反向传播）
    - 同时约束幅度和相位
    """
    def __init__(self, loss_type='l1'):
        super().__init__()
        if loss_type == 'l1':
            self.loss = nn.L1Loss(reduction='mean')
        else:
            self.loss = nn.MSELoss(reduction='mean')

    def forward(self, pred, target):
        """
        Args:
            pred: [B, C, H, W]
            target: [B, C, H, W]

        Returns:
            loss: 标量损失值
        """
        # FFT变换
        pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
        target_fft = torch.fft.fft2(target, dim=(-2, -1))

        # 堆叠实部和虚部 [B, C, H, W, 2]
        pred_fft_ri = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)
        target_fft_ri = torch.stack([target_fft.real, target_fft.imag], dim=-1)

        # 直接计算损失（同时约束实部和虚部）
        return self.loss(pred_fft_ri, target_fft_ri)


class SSIMLoss(nn.Module):
    """Differentiable SSIM loss for images normalized to [0, 1]."""
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
        self.padding = window_size // 2
        self.c1 = 0.01 ** 2
        self.c2 = 0.03 ** 2

    def forward(self, pred, target):
        mu_pred = F.avg_pool2d(
            pred, self.window_size, stride=1, padding=self.padding
        )
        mu_target = F.avg_pool2d(
            target, self.window_size, stride=1, padding=self.padding
        )
        sigma_pred = (
            F.avg_pool2d(
                pred * pred, self.window_size, stride=1, padding=self.padding
            ) - mu_pred * mu_pred
        )
        sigma_target = (
            F.avg_pool2d(
                target * target, self.window_size, stride=1, padding=self.padding
            ) - mu_target * mu_target
        )
        sigma_cross = (
            F.avg_pool2d(
                pred * target, self.window_size, stride=1, padding=self.padding
            ) - mu_pred * mu_target
        )

        numerator = (
            (2 * mu_pred * mu_target + self.c1)
            * (2 * sigma_cross + self.c2)
        )
        denominator = (
            (mu_pred * mu_pred + mu_target * mu_target + self.c1)
            * (sigma_pred + sigma_target + self.c2)
        )
        ssim_map = numerator / denominator.clamp_min(1e-12)
        return (1.0 - ssim_map).mean()


class GradientLoss(nn.Module):
    """L1 loss on horizontal and vertical image gradients."""
    def forward(self, pred, target):
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
        return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)



import os, time, math
import numpy as np

import torch
import torch.nn.functional as F
from torch import optim, nn
from torch.backends import cudnn
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from logger import plot_loss_log, plot_psnr_log
from metric import psnr, ssim
from model import Net1  # 你的双域改进版网络

from option_train import opt
from data.data_loader import TrainDataset, TestDataset



start_time = time.time()
steps = opt.iters_per_epoch * opt.epochs
T = steps


def lr_schedule_cosdecay(t, T, init_lr=opt.start_lr, end_lr=opt.end_lr):
    lr = end_lr + 0.5 * (init_lr - end_lr) * (1 + math.cos(t * math.pi / T))
    return lr


def train(net, loader_train, loader_test, optim, criterion, resume_state=None):
    amp_enabled = bool(opt.amp and opt.device == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    if resume_state is None:
        losses = []
        start_step = 0
        max_ssim = 0
        max_psnr = 0
        ssims = []
        psnrs = []
        no_improve_epochs = 0
    else:
        losses = resume_state.get('losses', [])
        start_step = int(resume_state.get('step', 0))
        max_ssim = float(resume_state.get('max_ssim', 0))
        max_psnr = float(resume_state.get('max_psnr', 0))
        ssims = resume_state.get('ssims', [])
        psnrs = resume_state.get('psnrs', [])
        no_improve_epochs = int(resume_state.get('no_improve_epochs', 0))
        print(
            f'Resuming from step {start_step}, best PSNR {max_psnr:.4f}, '
            f'no-improvement epochs {no_improve_epochs}'
        )

    loss_log = {'L1': [], 'FFT': [], 'SSIM': [], 'Grad': [], 'total': []}
    loss_log_tmp = {'L1': [], 'FFT': [], 'SSIM': [], 'Grad': [], 'total': []}
    psnr_log = []

    loader_train_iter = iter(loader_train)

    for step in range(start_step + 1, steps + 1):
        net.train()
        lr = opt.start_lr
        if not opt.no_lr_sche:
            lr = lr_schedule_cosdecay(step, T)
            for param_group in optim.param_groups:
                param_group["lr"] = lr

        x, y = next(loader_train_iter)
        x = x.to(opt.device)
        y = y.to(opt.device)

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            out = net(x)

            loss_L1 = out.new_zeros(())
            loss_FFT = out.new_zeros(())
            loss_SSIM = out.new_zeros(())
            loss_Grad = out.new_zeros(())
            if opt.w_loss_L1 > 0:
                loss_L1 = criterion[0](out, y)
            if opt.w_loss_FFT > 0:
                loss_FFT = criterion[1](out, y)
            if opt.w_loss_SSIM > 0:
                loss_SSIM = criterion[2](out, y)
            if opt.w_loss_Grad > 0:
                loss_Grad = criterion[3](out, y)

            loss = (
                opt.w_loss_L1 * loss_L1
                + opt.w_loss_FFT * loss_FFT
                + opt.w_loss_SSIM * loss_SSIM
                + opt.w_loss_Grad * loss_Grad
            )
        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            optim.step()
        optim.zero_grad()

        losses.append(loss.item())
        loss_log_tmp['L1'].append(loss_L1.item())
        loss_log_tmp['FFT'].append(loss_FFT.item())
        loss_log_tmp['SSIM'].append(loss_SSIM.item())
        loss_log_tmp['Grad'].append(loss_Grad.item())
        loss_log_tmp['total'].append(loss.item())

        print(
            f'\rloss:{loss.item():.5f} | L1:{loss_L1.item():.5f} | '
            f'FFT:{loss_FFT.item():.5f} | SSIM:{loss_SSIM.item():.5f} | '
            f'Grad:{loss_Grad.item():.5f} | step :{step}/{steps} | '
            f'lr :{lr :.7f} | time_used :{(time.time() - start_time) / 60 :.1f}',
            end='', flush=True
        )

        if step % len(loader_train) == 0:
            loader_train_iter = iter(loader_train)
            for key in loss_log.keys():
                loss_log[key].append(np.average(np.array(loss_log_tmp[key])))
                loss_log_tmp[key] = []
            if not opt.resume:
                plot_loss_log(loss_log, int(step / len(loader_train)), opt.saved_plot_dir)
            np.save(os.path.join(opt.saved_data_dir, 'losses.npy'), losses)

        should_eval = (
            step % opt.iters_per_epoch == 0
            and step <= opt.finer_eval_step
        ) or (
            step > opt.finer_eval_step
            and (step - opt.finer_eval_step) % (5 * len(loader_train)) == 0
        )
        if should_eval:
            epoch = int(step / opt.iters_per_epoch)
            with torch.no_grad():
                ssim_eval, psnr_eval = test(net, loader_test)

            log = (
                f'\nstep :{step} | epoch: {epoch} | '
                f'ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f}'
            )
            print(log)
            with open(os.path.join(opt.saved_data_dir, 'log.txt'), 'a') as f:
                f.write(log + '\n')

            ssims.append(ssim_eval)
            psnrs.append(psnr_eval)
            psnr_log.append(psnr_eval)
            if not opt.resume:
                plot_psnr_log(psnr_log, epoch, opt.saved_plot_dir)

            improved = psnr_eval > max_psnr
            if improved:
                max_psnr = psnr_eval
                max_ssim = ssim_eval
                no_improve_epochs = 0
                print(
                    f'\n model saved at step :{step}| epoch: {epoch} | '
                    f'best_psnr:{max_psnr:.4f}| best_ssim:{max_ssim:.4f}'
                )
            else:
                no_improve_epochs += 1

            checkpoint = {
                'epoch': epoch,
                'step': step,
                'max_psnr': max_psnr,
                'max_ssim': max_ssim,
                'no_improve_epochs': no_improve_epochs,
                'ssims': ssims,
                'psnrs': psnrs,
                'losses': losses,
                'model': net.state_dict(),
                'optimizer': optim.state_dict(),
                'scaler': scaler.state_dict() if amp_enabled else None,
                'amp': amp_enabled
            }

            if improved:
                torch.save(
                    checkpoint,
                    os.path.join(opt.saved_model_dir, 'best.pk')
                )
            torch.save(
                checkpoint,
                os.path.join(opt.saved_model_dir, 'last.pk')
            )

            np.save(os.path.join(opt.saved_data_dir, 'ssims.npy'), ssims)
            np.save(os.path.join(opt.saved_data_dir, 'psnrs.npy'), psnrs)

            if no_improve_epochs >= opt.patience:
                stop_log = (
                    f'early_stop: epoch {epoch}, step {step}, '
                    f'best_psnr {max_psnr:.4f}, '
                    f'no_improve_epochs {no_improve_epochs}'
                )
                print(stop_log)
                with open(os.path.join(opt.saved_data_dir, 'log.txt'), 'a') as f:
                    f.write(stop_log + '\n')
                break

def pad_img(x, patch_size):
    _, _, h, w = x.size()
    mod_pad_h = (patch_size - h % patch_size) % patch_size
    mod_pad_w = (patch_size - w % patch_size) % patch_size
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return x

def test(net, loader_test):
    net.eval()
    torch.cuda.empty_cache()
    ssims = []
    psnrs = []

    for i, (inputs, targets, hazy_name) in enumerate(loader_test):
        inputs = inputs.to(opt.device)
        targets = targets.to(opt.device)
        with torch.no_grad():
            H, W = inputs.shape[2:]
            inputs = pad_img(inputs, 4)
            with torch.cuda.amp.autocast(enabled=opt.amp and opt.device == 'cuda'):
                pred = net(inputs).clamp(0, 1)
            pred = pred[:, :, :H, :W].float()
            targets = targets.float()
            # save_path = os.path.join(opt.saved_infer_dir, hazy_name[0])
            # save_image(pred, save_path)
        ssim_tmp = ssim(pred, targets).item()
        psnr_tmp = psnr(pred, targets)
        ssims.append(ssim_tmp)
        psnrs.append(psnr_tmp)

    return np.mean(ssims), np.mean(psnrs)


def set_seed_torch(seed=2018):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":

    set_seed_torch(666)

    train_dir = os.path.join(opt.dataset_root, opt.dataset, 'train')
    train_set = TrainDataset(os.path.join(train_dir, 'hazy'), os.path.join(train_dir, 'clear'))
    test_dir = os.path.join(opt.dataset_root, opt.dataset, 'test')
    test_set = TestDataset(os.path.join(test_dir, 'hazy'), os.path.join(test_dir, 'clear'))
    loader_train = DataLoader(dataset=train_set, batch_size=opt.bs, shuffle=True, num_workers=12)
    loader_test = DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=4)

    sfda_flags = {
        'spatial_frequency_full': (True, True),
        'spatial_only': (True, False),
        'frequency_only': (False, True),
    }
    dsrm_flags = {
        'dsrm_full': (True, True),
        'no_ffcm': (False, True),
        'no_pa': (True, False),
        'local_only': (False, False),
    }
    use_spatial, use_frequency = sfda_flags[opt.sfda_variant]
    use_ffcm, use_pa = dsrm_flags[opt.dsrm_variant]
    use_deconv = opt.deconv_variant == 'all_deconv'
    print('sfda_variant:', opt.sfda_variant)
    print('dsrm_variant:', opt.dsrm_variant)
    print('deconv_variant:', opt.deconv_variant)
    print('fusion_variant:', opt.fusion_variant)
    print('batch_size:', opt.bs)
    print('AMP:', opt.amp and opt.device == 'cuda')
    print('w_loss_L1:', opt.w_loss_L1)
    print('w_loss_FFT:', opt.w_loss_FFT)
    print('w_loss_SSIM:', opt.w_loss_SSIM)
    print('w_loss_Grad:', opt.w_loss_Grad)
    print('patience:', opt.patience)
    print('resume:', opt.resume)
    net = Net1(
        base_dim=32,
        use_spatial=use_spatial,
        use_frequency=use_frequency,
        use_ffcm=use_ffcm,
        use_pa=use_pa,
        use_deconv=use_deconv,
        fusion_variant=opt.fusion_variant,
    )
    net = net.to(opt.device)

    epoch_size = len(loader_train)
    print("epoch_size: ", epoch_size)
    if opt.device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Total_params: ==> {}".format(pytorch_total_params))

    criterion = []
    criterion.append(nn.L1Loss().to(opt.device))

    criterion.append(FFTLoss(loss_type='l1').to(opt.device))
    criterion.append(SSIMLoss().to(opt.device))
    criterion.append(GradientLoss().to(opt.device))
    optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, net.parameters()), lr=opt.start_lr, betas=(0.9, 0.999),
                           eps=1e-08)
    optimizer.zero_grad()

    resume_state = None
    if opt.resume:
        resume_path = opt.resume_model or os.path.join(
            opt.saved_model_dir, 'last.pk'
        )
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f'Resume checkpoint not found: {resume_path}')
        resume_state = torch.load(resume_path, map_location=opt.device)
        if opt.reset_patience:
            resume_state['no_improve_epochs'] = 0
            print('reset no-improvement counter for resumed training')
        net.load_state_dict(resume_state['model'], strict=True)
        optimizer.load_state_dict(resume_state['optimizer'])
        if opt.amp and opt.device == 'cuda' and resume_state.get('scaler') is not None:
            scaler.load_state_dict(resume_state['scaler'])
        print('resume checkpoint:', resume_path)

    train(
        net,
        loader_train,
        loader_test,
        optimizer,
        criterion,
        resume_state=resume_state
    )
