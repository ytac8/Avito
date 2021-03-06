from torch.utils.data import Dataset
import pickle
import torch
import joblib


class Data(Dataset):

    def __init__(self, data, max_len, feature_dim, is_train):
        self.data = data
        self._fillna()
        self.max_len = max_len          # 最大系列長
        self.feature_dim = feature_dim  # 文字の種類数
        self.is_train = is_train
        # with open('../../data/pickle/char_onehot.pkl', mode='rb') as f:
        #     self.ohe = pickle.load(f)
        with open('../../data/pickle/item_id_dict.pkl', mode='rb') as f:
            self.item_id_dict = pickle.load(f)
        with open('../../data/pickle/label_encoded.pkl', mode='rb') as f:
            self.label_encoded = joblib.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item_id = torch.LongTensor([self.item_id_dict[self.data.item_id[idx]]])
        # description = self.data.description[idx]
        # feature = self._preprocess(description)
        feature = self._getPreprocessed(idx).long()
        if self.is_train:
            label = torch.FloatTensor(
                [self.data.deal_probability[idx]])
            datasets = {"feature": feature, "label": label}
        else:
            datasets = {"feature": feature, "item_id": item_id}
        return datasets

    def _preprocess(self, description):
        feature = torch.zeros(self.max_len)
        if not description == 0:
            char = list(description)
            length = len(char)
            feature[:length] = torch.from_numpy()
        return feature.unsqueeze(0)

    def _getPreprocessed(self, idx):
        feature = torch.zeros(self.max_len)
        data = self.label_encoded[idx]['encoded']
        feature[:len(data)] = torch.LongTensor(data)
        return feature

    def _fillna(self):
        self.data.description.fillna(0, inplace=True)
