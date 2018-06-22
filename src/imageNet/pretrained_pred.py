import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from dataset import Data
from torchvision import transforms
from image_preprocess import Rescale, RandomCrop, ToTensor
from torchvision.models import resnet152, vgg16_bn, inception_v3


def main():

    # initialize
    use_cuda = torch.cuda.is_available()
    batch_size = 1024

    device = "cuda" if use_cuda else "cpu"
    model_list = [inception_v3(pretrained=True)]
    model_name_list = ['inception']

    for t in ['train', 'test']:
        # for t in ['test']:
        if t == 'train':
            train_df = pd.read_csv('../../data/unzipped/train.csv')
            data_loader = set_data_loader(train_df, batch_size, True)
            item_id_series = train_df.item_id
        else:
            test_df = pd.read_csv('../../data/unzipped/test.csv')
            data_loader = set_data_loader(test_df, batch_size, False)
            item_id_series = test_df.item_id

        for model, model_name in zip(model_list, model_name_list):
            feature_list = []
            model.to(device)
            model = nn.DataParallel(model)
            model.eval()

            with torch.no_grad():
                for batch_i, batch in enumerate(data_loader):
                    input = batch['image'].to(device)
                    feature = model(input).to('cpu').detach().tolist()
                    feature_list += feature

            feature_df = pd.DataFrame(feature_list).astype('float32')
            feature_df.rename(columns=lambda s: str(s), inplace=True)
            feature_df['item_id'] = item_id_series
            feature_df.to_feather(
                t + '_' + model_name + "_image_feature.feather")
        print('finish ' + t)


def set_data_loader(data, batch_size, is_train):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    dataset = Data(data, is_train=is_train, transforms=transforms.Compose(
        [Rescale(334), RandomCrop(299), ToTensor(), normalize]))
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=16)
    return data_loader


if __name__ == '__main__':
    main()
