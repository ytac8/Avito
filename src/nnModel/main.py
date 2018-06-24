import gc
import torch
import torch.nn as nn
import argparse
import numpy as np
import pandas as pd
import joblib
from torch.utils.data import DataLoader
from trainer import Trainer
from predictor import Predictor
from dataset import Data
from optimizer import Optimizer
from model import NNModel
from sklearn.utils import shuffle


def main(epochs, is_train=1):
    # initialize
    is_train = True if is_train == 1 else False
    n_epochs = epochs
    use_cuda = torch.cuda.is_available()
    checkpoint_path = None

    # learning parameters
    save_epoch = 5
    batch_size = 512
    learning_rate = 0.001
    val_ratio = 0.2
    model = NNModel()

    if torch.cuda.is_available():
        print('use cuda')
        device = "cuda" if use_cuda else "cpu"
        model.to(device)

    if is_train:
        base_data = joblib.load('../../data/features/train_base_feature.gz')
        description_data = joblib.load(
            '../../data/features/train_description_vec.gz')
        title_data = joblib.load(
            '../../data/features/train_title_vec.gz')
        description_data = np.asarray(description_data)
        title_data = np.asarray(title_data)

        base_data, train_description_data, train_title_data = shuffle(
            base_data, description_data, title_data)

        train_base_data = base_data[int(
            len(base_data) * val_ratio):].reset_index(drop=True)
        val_base_data = base_data[:int(
            len(base_data) * val_ratio)].reset_index(drop=True)
        train_description_data = description_data[int(
            len(base_data) * val_ratio):]
        val_description_data = description_data[:int(
            len(base_data) * val_ratio)]
        train_title_data = title_data[int(len(base_data) * val_ratio):]
        val_title_data = title_data[:int(len(base_data) * val_ratio)]

        # debug
        # train_base_data = base_data[3000:5000].reset_index(drop=True)
        # train_description_data = description_data[3000:5000]
        # train_title_data = title_data[3000:5000]
        # val_base_data = base_data[:3000].reset_index(drop=True)
        # val_description_data = description_data[:3000]
        # val_title_data = title_data[:3000]

        del base_data, description_data, title_data
        gc.collect()

        train_dataset = Data(train_base_data, train_title_data,
                             train_description_data, is_train)
        val_dataset = Data(val_base_data, val_title_data,
                           val_description_data, is_train)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, num_workers=8)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, num_workers=8)
        print('data loaded!')

        optimizer = Optimizer(
            model, lr=learning_rate)
        criterion = nn.MSELoss().to(device)
        model = nn.DataParallel(model)
        trainer = Trainer(train_loader, val_loader, model, criterion,
                          optimizer, use_cuda, n_epochs, save_epoch)

        # training
        print('now training')
        trainer.train()
        print('finished training!!!')

    else:
        print('test prediction')
        item_id_dict = joblib.load('../../data/pickle/label_dict.pkl')
        test_df = pd.read_csv('../../data/unzipped/test.csv')

        test_loader = Data(
            test_df,  batch_size, is_train)
        predictor = Predictor(test_loader, model,
                              use_cuda, item_id_dict, is_train)

        checkpoint_path = '../output/save_point/model_5epochs.pth.tar'
        pred_list, item_id_list, target_list = predictor.predict()
        item_id = pd.Series(item_id_list)
        deal_probability = pd.Series(pred_list)
        submission_df = pd.DataFrame()
        submission_df['item_id'] = item_id
        submission_df['deal_probability'] = deal_probability
        submission_df.to_csv('../output/predictions/predict.csv', index=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Setting model and dataset')
    parser.add_argument('-ep', '--epochs', default=10, metavar='epochs',
                        type=int, help='epochs')
    parser.add_argument('-t', '--train', default=1, metavar='train_mode',
                        type=int, help='train_mode')

    args = parser.parse_args()
    epochs = args.epochs
    is_train = args.train
    main(epochs, is_train)
