import torch
import os
import pickle
from torch.utils.data import Dataset
from skimage import io


class Data(Dataset):

    def __init__(self, data, is_train, transforms=None):
        self.data = data
        if is_train:
            self.img_dir = '../../data/img/train_jpg/'
        else:
            self.img_dir = '../../data/img/test_jpg/'
            with open('../../data/pickle/item_id_dict.pkl', mode='rb') as f:
                self.item_id_dict = pickle.load(f)

        self.is_train = is_train
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name = self.data.image.iloc[idx]
        if (type(image_name) == str):
            image_name = str(self.data.image.iloc[idx]) + '.jpg'
            image_path = self.img_dir + image_name
            if os.path.exists(image_path):
                try:
                    img = io.imread(image_path)
                except:
                    img = torch.zeros(224, 224, 3).float()
                    print('aa')
            else:
                img = torch.zeros(224, 224, 3).float()

            if self.transforms:
                img = self.transforms(img).float()
        else:
            img = torch.zeros(3, 224, 224).float()

        target = self.data.deal_probability.iloc[idx]
        target = torch.FloatTensor([target])

        if not self.is_train:
            item_id = self.data.item_id.iloc[idx]
            item_id = self.item_id_dict[item_id]
            return {"item_id": item_id, "image": img, "target": target}

        return {"image": img, "target": target}
