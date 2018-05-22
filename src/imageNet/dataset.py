from torch.utils.data import Dataset
from torchvision import transforms
import torch
import pillow


class Data(Dataset):

    def __init__(self, data, max_len, feature_dim, is_train):
        self.data = data
        self.is_train = is_train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item_id = torch.LongTensor([self.item_id_dict[self.data.item_id[idx]]])
        feature = self._getPreprocessed(idx).long()
        if self.is_train:
            label = torch.FloatTensor(
                [self.data.deal_probability[idx]])
            datasets = {"feature": feature, "label": label}
        else:
            datasets = {"feature": feature, "item_id": item_id}
        return datasets
