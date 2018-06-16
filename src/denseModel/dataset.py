import torch
from torch.utils.data import Dataset


class Data(Dataset):

    def __init__(self, x, y=None, is_train=True):
        self.x = x
        self.y = y
        self.is_train = is_train

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        feature = self.x[idx, :].toarray()
        feature = torch.FloatTensor(feature)

        if self.is_train:
            target = self.y.iloc[idx]
            target = 1 if target >= 0.5 else 0
            target = torch.LongTensor([target])

            return {"feature": feature, "target": target}

        return {"feature": feature}
