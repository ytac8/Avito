import torch
import torch.nn as nn
import joblib
import pandas as pd
from torch.utils.data import DataLoader
from dataset import Data
from feature_extractor import VGG16FeatureExtractor
from torchvision import transforms
from image_preprocess import Rescale, RandomCrop, ToTensor


def main():
    # initialize
    use_cuda = torch.cuda.is_available()
    batch_size = 128

    model = VGG16FeatureExtractor()

    item_id_dict = joblib.load('../../data/pickle/label_dict.pkl')
    item_id_invdict = {v: k for k, v in item_id_dict.items()}
    device = "cuda" if use_cuda else "cpu"
    model.to(device)
    # model = nn.DataParallel(model)
    model.eval()
    # for t in ['train', 'test']:
    for t in ['test']:
        if t == 'train':
            train_df = pd.read_csv('../../data/unzipped/train.csv')
            train_df = train_df.sample(
                frac=1, random_state=114514).reset_index(drop=True)
            data_loader = set_data_loader(train_df, batch_size, True)
        else:
            test_df = pd.read_csv('../../data/unzipped/test.csv')
            test_df = test_df.sample(
                frac=1, random_state=114514).reset_index(drop=True)
            data_loader = set_data_loader(test_df, batch_size, False)

        feature_list = []
        item_id_list = []

        with torch.no_grad():
            for batch_i, batch in enumerate(data_loader):
                item_id = batch['item_id'].view(-1).detach().tolist()
                item_id = [item_id_invdict[x] for x in item_id]
                input = batch['image'].to(device)
                feature = model(input).to('cpu').detach().tolist()
                item_id_list += item_id
                feature_list += feature

        item_id_series = pd.Series(item_id_list)
        feature_df = pd.DataFrame(feature_list)

        feature_df['item_id'] = item_id_series
        feature_df.to_hdf(t + "_image_feature.h5", 'table',
                          complib='blosc', complevel=9)
        print('finish ' + t)


def set_data_loader(data, batch_size, is_train):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    dataset = Data(data, is_train=is_train, transforms=transforms.Compose(
        [Rescale(256), RandomCrop(224), ToTensor(), normalize]))
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=16)
    return data_loader


if __name__ == '__main__':
    main()
