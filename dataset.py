from torch.utils.data import Dataset
import PIL.Image as Image
import os


def make_dataset(root,mode):
    # root = "./data/train"
    imgs = []
    if mode == "train":
        ori_path = os.path.join(root, "image")
        ground_path = os.path.join(root, "mask")
    elif mode == "test":
        ori_path = os.path.join(root, "image")
        ground_path = os.path.join(root, "mask")        
    names = os.listdir(ori_path)
    n = len(names)
    
    for i in range(n):
        img = os.path.join(ori_path, names[i])
        mask = os.path.join(ground_path, names[i])
        imgs.append((img, mask))
    return imgs


class LiverDataset(Dataset):
    def __init__(self,root,mode='train',transform=None, target_transform=None):
        imgs = make_dataset(root,mode)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = Image.open(x_path).convert('L')
        img_y = Image.open(y_path).convert('L')
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_y

    def __len__(self):
        return len(self.imgs)
