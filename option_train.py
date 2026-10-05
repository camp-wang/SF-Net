import torch,os,sys,torchvision,argparse
import torch,warnings
import json

# warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str,default='Automatic detection')

parser.add_argument('--epochs', type=int,default=100)
parser.add_argument('--iters_per_epoch', type=int,default=5000)
parser.add_argument('--finer_eval_step', type=int,default=400000)
parser.add_argument('--bs', type=int,default=4,help='batch size')
parser.add_argument('--start_lr', default=0.0004, type=float, help='start learning rate')
parser.add_argument('--end_lr', default=0.000001, type=float, help='end learning rate')
parser.add_argument('--no_lr_sche', action='store_true',help='no lr cos schedule')
parser.add_argument('--amp', action='store_true', help='use CUDA automatic mixed precision')
parser.add_argument('--use_warm_up', type=bool, default=False, help='using warm up in learning rate')

parser.add_argument('--w_loss_L1', default=1., type=float, help='weight of loss L1')
parser.add_argument('--w_loss_CR', default=0.1, type=float, help='weight of loss CR')
parser.add_argument('--w_loss_FFT', default=0.1, type=float, help='weight of loss FFT')
parser.add_argument('--w_loss_SSIM', default=0.0, type=float, help='weight of SSIM loss')
parser.add_argument('--w_loss_Grad', default=0.0, type=float, help='weight of gradient loss')


parser.add_argument('--exp_dir', type=str, default='outputs/experiment')
parser.add_argument('--model_name', type=str, default='MDCTDN')
parser.add_argument('--saved_model_dir', type=str, default='saved_model')
parser.add_argument('--saved_data_dir', type=str, default='saved_data')
parser.add_argument('--saved_plot_dir', type=str, default='saved_plot')
parser.add_argument('--saved_infer_dir', type=str, default='saved_infer_dir')

parser.add_argument('--dataset', type=str, default='ITS')
parser.add_argument('--dataset_root', type=str, default='dataset')
parser.add_argument('--sfda_variant', type=str,
                    choices=['spatial_frequency_full', 'spatial_only', 'frequency_only'],
                    default='spatial_frequency_full')
parser.add_argument('--dsrm_variant', type=str,
                    choices=['dsrm_full', 'no_ffcm', 'no_pa', 'local_only'],
                    default='dsrm_full')
parser.add_argument('--deconv_variant', type=str,
                    choices=['all_deconv', 'all_conv'],
                    default='all_deconv')
parser.add_argument('--fusion_variant', type=str,
                    choices=['cga', 'concat'],
                    default='cga')

# only need for resume
parser.add_argument('--resume', action='store_true', help='resume from a saved checkpoint')
parser.add_argument('--resume_model', type=str, default='', help='checkpoint path; defaults to saved_model/last.pk')
parser.add_argument('--patience', type=int, default=5, help='early-stop epochs without PSNR improvement')
parser.add_argument('--reset_patience', action='store_true', help='reset no-improvement counter when resuming')
parser.add_argument('--pre_trained_model', type=str,default='null')

opt=parser.parse_args()
opt.device='cuda' if torch.cuda.is_available() else 'cpu'

dataset_dir = os.path.join(opt.exp_dir, opt.dataset)
model_dir = os.path.join(dataset_dir, opt.model_name)

if not os.path.exists(opt.exp_dir):
    os.makedirs(opt.exp_dir)
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

opt.saved_model_dir = os.path.join(model_dir, 'saved_model')
opt.saved_data_dir = os.path.join(model_dir, 'saved_data')
opt.saved_plot_dir = os.path.join(model_dir, 'saved_plot')
opt.saved_infer_dir = os.path.join(model_dir, 'saved_infer')

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    os.makedirs(opt.saved_model_dir)
    os.makedirs(opt.saved_data_dir)
    os.makedirs(opt.saved_plot_dir)
    os.makedirs(opt.saved_infer_dir)
elif not opt.resume:
    print(f'{model_dir} has already existed!')
    exit()
else:
    os.makedirs(opt.saved_model_dir, exist_ok=True)
    os.makedirs(opt.saved_data_dir, exist_ok=True)
    os.makedirs(opt.saved_plot_dir, exist_ok=True)
    os.makedirs(opt.saved_infer_dir, exist_ok=True)

print(opt)
print('model_dir:', model_dir)

args_name = 'args_resume.json' if opt.resume else 'args.txt'
with open(os.path.join(model_dir, args_name), 'w') as f:
    json.dump(opt.__dict__, f, indent=2)
