from trainer import Trainer
from optimizer import Optimizer
import torch.nn as nn
import pandas as pd
from model import CountFeatureNN
import torch
from preprocessor import Preprocessor
from dataset import Data
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


def main():
    is_train = True
    batch_size = 1024
    leainrning_rate = 0.001
    use_cuda = True
    n_epochs = 1000
    save_epoch = 15

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CountFeatureNN(41331, 2)
    model.to(device)

    train_image_feature = pd.read_hdf(
        '../../data/features/train2_image_feature.h5', 'table')
    test_image_feature = pd.read_hdf(
        '../../data/features/test2_image_feature.h5', 'table')

    preprocessor = Preprocessor('../../data/features/default_feature.pkl')
    # preprocessor = Preprocessor()
    preprocessor.add_feture(train_image_feature, test_image_feature, 'image')
    preprocessed = preprocessor.get_feature_vec()

    x_train = preprocessed['x_train']
    y_train = preprocessed['y_train']
    x_test = preprocessed['x_test']

    train_features, val_features, train_label, val_label = train_test_split(
        x_train, y_train, test_size=0.2, random_state=114514)

    if is_train:
        train_dataset = Data(train_features, train_label, is_train)
        train_loader = DataLoader(train_dataset, batch_size,
                                  shuffle=False, num_workers=16)
        val_dataset = Data(val_features, val_label, is_train)
        val_loader = DataLoader(val_dataset, batch_size,
                                shuffle=False, num_workers=16)
        optimizer = Optimizer(
            model, model.parameters(), lr=leainrning_rate)
        criterion = nn.NLLLoss().to(device)
        # model = nn.DataParallel(model)
        trainer = Trainer(train_loader, val_loader, model, criterion,
                          optimizer, use_cuda, n_epochs, save_epoch)
        print('now training!')
        trainer.train()
    else:
        print('test prediction')
        print(len(x_test))


if __name__ == '__main__':
    main()
