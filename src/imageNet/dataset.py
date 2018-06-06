import torch
import os
import joblib
from torch.utils.data import Dataset
from skimage import io


class Data(Dataset):

    def __init__(self, data, is_train, transforms=None):
        self.data = data
        if is_train:
            self.img_dir = '../../data/img/train_jpg/'
        else:
            self.img_dir = '../../data/img/test_jpg/'

        self.is_train = is_train
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name = self.data.image.iloc[idx]
        if (type(image_name) == str):
            image_name = image_name + '.jpg'
            image_path = self.img_dir + image_name
            if os.path.exists(image_path):
                try:
                    img = io.imread(image_path)
                except:
                    img = torch.zeros(224, 224, 3).float()
            else:
                img = torch.zeros(224, 224, 3).float()

            if self.transforms:
                img = self.transforms(img).float()
        else:
            img = torch.zeros(3, 224, 224).float()

        if self.is_train:
            target = self.data.deal_probability.iloc[idx]
            target = torch.FloatTensor([target])
            return {"image": img, "target": target}

        return {"image": img}
