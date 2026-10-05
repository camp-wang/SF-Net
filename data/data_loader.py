import os, random
import torch.utils.data as data
from PIL import Image
from torchvision.transforms.functional import hflip
from torchvision.transforms import ToTensor, Resize
from torchvision.transforms.functional import hflip, rotate, crop
from torchvision.transforms import ToTensor, RandomCrop, Resize
import torch  # 导入torch库，用于生成随机数,一定要加否则报错
import cv2

cv2.setNumThreads(0)  # 避免 DataLoader num_workers>0 时 cv2 内部线程争用

# clear_image_name = hazy_image_name.split('_')[0] + '.jpg' #NHR
def resolve_clear_image_name(hazy_image_name, clear_path):
    candidates = [hazy_image_name, hazy_image_name.replace("NighttimeHazy", "lowLight"), hazy_image_name.replace("_nightHazy", ""), hazy_image_name.replace("nightHazy", "")]
    for candidate in candidates:
        if os.path.exists(os.path.join(clear_path, candidate)):
            return candidate
    raise FileNotFoundError("No clear image matched %s in %s" % (hazy_image_name, clear_path))

# clear_image_name = hazy_image_name #GTA5 UNREAL_NH
# clear_image_name = hazy_image_name.rsplit('_', 1)[0] + '.jpg' #NightHaze
# # clear_image_name = resolve_clear_image_name(hazy_image_name, self.clear_path)  # NHM


class TrainDataset(data.Dataset):
    def __init__(self, hazy_path, clear_path):
        super(TrainDataset, self).__init__()
        self.hazy_path = hazy_path
        self.clear_path = clear_path
        self.hazy_image_list = os.listdir(hazy_path)
        self.clear_image_list = os.listdir(clear_path)

        # 目标裁剪尺寸
        self.target_size = 258

    def __getitem__(self, index):
        hazy_image_name = self.hazy_image_list[index]
        clear_image_name = resolve_clear_image_name(hazy_image_name, self.clear_path)  # NHM


        hazy_image_path = os.path.join(self.hazy_path, hazy_image_name)
        clear_image_path = os.path.join(self.clear_path, clear_image_name)

        hazy = Image.open(hazy_image_path).convert('RGB')
        clear = Image.open(clear_image_path).convert('RGB')

        # 第一步：检查并调整图像尺寸
        w, h = hazy.size

        # 如果短边小于256，等比例放大到短边为256
        if min(w, h) < self.target_size:
            # 计算放大比例
            scale_factor = self.target_size / min(w, h)
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)

            # 使用高质量插值放大图像Image.BILINEAR；Image.BICUBIC
            hazy = hazy.resize((new_w, new_h), Image.BICUBIC)
            clear = clear.resize((new_w, new_h), Image.BICUBIC)

        # 第二步：随机裁剪到256×256
        crop_params = RandomCrop.get_params(hazy, [256, 256])
        hazy = crop(hazy, *crop_params)
        clear = crop(clear, *crop_params)

        # 随机水平翻转（50%概率）
        if torch.rand(1).item() > 0.5:
            hazy = hflip(hazy)
            clear = hflip(clear)

        to_tensor = ToTensor()

        hazy = to_tensor(hazy)
        clear = to_tensor(clear)

        return hazy, clear

    def __len__(self):
        return len(self.hazy_image_list)


class TestDataset(data.Dataset):
    def __init__(self, hazy_path, clear_path):
        super(TestDataset, self).__init__()
        self.hazy_path = hazy_path
        self.clear_path = clear_path
        self.hazy_image_list = os.listdir(hazy_path)
        self.clear_image_list = os.listdir(clear_path)
        self.hazy_image_list.sort()
        self.clear_image_list.sort()

    def __getitem__(self, index):
        # data shape: C*H*W

        hazy_image_name = self.hazy_image_list[index]
        clear_image_name = resolve_clear_image_name(hazy_image_name, self.clear_path)  # NHM

        hazy_image_path = os.path.join(self.hazy_path, hazy_image_name)
        clear_image_path = os.path.join(self.clear_path, clear_image_name)

        hazy = Image.open(hazy_image_path).convert('RGB')
        clear = Image.open(clear_image_path).convert('RGB')

        to_tensor = ToTensor()

        hazy = to_tensor(hazy)
        clear = to_tensor(clear)

        return hazy, clear, hazy_image_name

    def __len__(self):
        return len(self.hazy_image_list)


class ValDataset(data.Dataset):
    def __init__(self, hazy_path, clear_path):
        super(ValDataset, self).__init__()
        self.hazy_path = hazy_path
        self.clear_path = clear_path
        self.hazy_image_list = os.listdir(hazy_path)
        self.clear_image_list = os.listdir(clear_path)
        self.hazy_image_list.sort()
        self.clear_image_list.sort()

    def __getitem__(self, index):
        hazy_image_name = self.hazy_image_list[index]
        clear_image_name = resolve_clear_image_name(hazy_image_name, self.clear_path)  # NHM

        hazy_image_path = os.path.join(self.hazy_path, hazy_image_name)
        clear_image_path = os.path.join(self.clear_path, clear_image_name)

        hazy = Image.open(hazy_image_path).convert('RGB')
        clear = Image.open(clear_image_path).convert('RGB')

        to_tensor = ToTensor()

        hazy = to_tensor(hazy)
        clear = to_tensor(clear)

        return {'hazy': hazy, 'clear': clear, 'filename': hazy_image_name}

    def __len__(self):
        return len(self.hazy_image_list)