import torch
import os
import joblib
from torch.utils.data import Dataset
from skimage import io
from torchvision import transforms
from image_preprocess import RandomCrop, Rescale, ToTensor


class Data(Dataset):

    def __init__(self, base_data, title_feature, description_feature, count_feature, price_feature,
                 is_train):

        self.base_data = base_data
        self.title_data = title_feature
        self.description_data = description_feature
        self.count_feature = count_feature
        self.price_feature = price_feature
        self.is_train = is_train

    def __len__(self):
        return len(self.base_data)

    def __getitem__(self, idx):
        image_name = self.base_data.image[idx]
        image = self.get_image(image_name)

        price_feature = torch.FloatTensor(self.price_feature[idx])
        count_feature = torch.FloatTensor(self.count_feature[idx])

        title = torch.FloatTensor(self.title_data[idx])
        title = self.text_padding(title, 21)
        description = torch.FloatTensor(self.description_data[idx])
        description = self.text_padding(description, 716)

        user_id = torch.LongTensor([self.base_data.user_id[idx]])
        user_type = torch.LongTensor([self.base_data.user_type[idx]])
        region = torch.LongTensor([self.base_data.region[idx]])
        city = torch.LongTensor([self.base_data.city[idx]])
        category = torch.LongTensor([self.base_data.category_name[idx]])
        image_top = torch.LongTensor([self.base_data.image_top_1[idx]])

        price = torch.FloatTensor([self.base_data.price[idx]])
        item_seq_num = torch.FloatTensor([self.base_data.item_seq_number[idx]])

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
            "price": price,
            "count_feature": count_feature,
            "price_feature": price_feature
        }

        if self.is_train:
            feature['target'] = self.base_data.deal_probability[idx]

        return feature

    def text_padding(self, text_data, max_len):
        padded = torch.zeros(max_len, text_data.size(1))
        padded[:len(text_data), :] = text_data
        return padded

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
        transform = transforms.Compose(
            [Rescale(256), RandomCrop(224), ToTensor(), normalize])
        img = transform(img).float()

        return img
