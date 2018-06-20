import torch
import os
import joblib
from torch.utils.data import Dataset
from skimage import io
from torchvisoin import transforms
from image_preprocess import RandomCrop, Rescale, ToTensor


class Data(Dataset):

    def __init__(self, base_data, title_feature, description_feature,
                 is_train):

        self.base_data = base_data
        self.title_data = title_feature
        self.description_data = description_feature
        self.is_train = is_train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name = self.base_data.image[idx]
        image = self.get_image(image_name)
        title = self.title_feature[idx]
        description = self.description_data[idx]
        user_id = self.base_data.user_id[idx]
        user_type = self.base_data.user_type[idx]
        region = self.base_data.region[idx]
        city = self.base_data.city[idx]
        category = self.base_data.category[idx]
        image_top = self.base_data.image_top_1[idx]
        price = self.base_data.price[idx]
        item_seq_num = self.base_data.item_seq_number[idx]

        feature = {
            "image": image,
            "title": title,
            "description": description,
            "user_id": user_id,
            "user_type": user_type,
            "region": region,
            "city": city,
            "image_top": image_top,
            "category": category,
            "item_seq_num": item_seq_num,
            "price": price
        }

        if self.is_train:
            feature['target'] = self.base_data.deal_probability[idx]

        return feature

    def get_image(self, image_name):
        if self.is_train:
            img_dir = '../../data/img/train_jpg/'
        else:
            img_dir = '../../data/img/test_jpg/'

        if (type(image_name) == str):
            image_name = image_name + '.jpg'
            image_path = img_dir + image_name
            if os.path.exists(image_path):
                try:
                    img = io.imread(image_path)
                except:
                    img = torch.zeros(224, 224, 3).float()
            else:
                img = torch.zeros(224, 224, 3).float()

        else:
            img = torch.zeros(224, 224, 3).float()

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        img = transforms.Compose(
            [Rescale(256), RandomCrop(224), ToTensor(), normalize])

        return img
