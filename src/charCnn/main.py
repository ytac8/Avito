import os
import gc
import torch
import torch.nn as nn
import argparse
import datetime
import pickle
import pandas as pd
from torch.utils.data import DataLoader
from trainer import Trainer
from predictor import Predictor
from charcnn import CharCNN
from dataset import Data
from optimizer import Optimizer
from sklearn.metrics import mean_squared_error


def main(epochs, is_train=1):
    # initialize
    is_train = True if is_train == 1 else False
    n_epochs = epochs
    save_epoch = 1
    use_cuda = torch.cuda.is_available()
    checkpoint_path = None

    # learning parameters
    batch_size = 32
    max_length = 3212
    feature_dim = 1749
    dropout_p = 0.6
    learning_rate = 0.2
    filter_num = 256
    output_size = 2
    val_ratio = 0.2

    model = CharCNN(feature_dim=feature_dim, dropout_p=dropout_p,
                    output_size=output_size, filter_num=filter_num)

    model = nn.DataParallel(model)
    with open('../../data/pickle/item_id_dict.pkl', mode='rb') as f:
        item_id_dict = pickle.load(f)

    if torch.cuda.is_available():
        print('use cuda')
        device = "cuda" if use_cuda else "cpu"
        model.to(device)

    if is_train:
        df = pd.read_csv('../../data/unzipped/train.csv')
        df = df.sample(frac=1, random_state=114514).reset_index(drop=True)
        # train_df = df[int(len(df) * val_ratio):].reset_index(drop=True)
        # val_df = df[:int(len(df) * val_ratio)].reset_index(drop=True)

        # debug
        train_df = df[512:1024].reset_index(drop=True)
        val_df = df[:512].reset_index(drop=True)

        del df
        gc.collect()
        train_loader = dataset(
            train_df, batch_size, max_length, feature_dim, is_train)
        val_loader = dataset(
            val_df, batch_size, max_length, feature_dim, is_train)

        optimizer = Optimizer(model, lr=learning_rate)
        criterion = nn.MSELoss()
        predictor = Predictor(val_loader, model, use_cuda, item_id_dict)

        trainer = Trainer(train_loader, model, criterion,
                          optimizer, use_cuda=use_cuda)

        # training
        for epoch in range(1, n_epochs + 1):
            loss = trainer.train()

            # validation
            if epoch % 1 == 0:
                pred_and_print(predictor, model, val_loader,
                               checkpoint_path, epoch)
                # save_model(model, optimizer, epoch, save_epoch)
            print(epoch, loss)
        print('finished training')

    else:
        print('test_prediction')
        test_df = pd.read_csv('../../data/unzipped/test.csv')
        with open('../../data/pickle/item_id_dict.pkl', mode='rb') as f:
            item_id_dict = pickle.load(f)

        test_loader = dataset(
            test_df,  batch_size, max_length, feature_dim, is_train)
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


def pred_and_print(predictor, model, data_loader, checkpoint, epochs=None):
    epochs = '' if epochs is None else ' ' + str(epochs) + ' epochs '
    pred_list, item_id_list, target_list = predictor.predict()
    if target_list is not None:
        rmse = mean_squared_error(pred_list, target_list)
        print('valid score' + str(epochs) + ': ' + str(rmse))
    return pred_list, item_id_list


def save_model(model, optimizer, epoch, save_epoch):
    if epoch % save_epoch == 0:
        model_filename = '../output/save_point/' + \
            'model_' + str(epoch) + 'epochs.pth.tar'

        state = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.model_optimizer.state_dict(),
        }
        torch.save(state, model_filename)


def dataset(data, batch_size, max_length, feature_dim, is_train):
    dataset = Data(data, max_length, feature_dim, is_train)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    return data_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Setting model and dataset')
    parser.add_argument('-ep', '--epochs', default=10, metavar='epochs', type=int,
                        help='epochs')
    parser.add_argument('-t', '--train', default=1, metavar='train_mode', type=int,
                        help='train_mode')

    args = parser.parse_args()
    now = datetime.datetime.now().strftime('%s')
    output_dir_name = 'log/'
    epochs = args.epochs
    file_name = now + '.csv'
    is_train = args.train

    try:
        os.makedirs('../log/' + output_dir_name)
    except FileExistsError as e:
        pass

    output_path = '../' + output_dir_name + file_name
    main(epochs, is_train)
    sum_aucs = 0

    with open(output_path, 'w') as f:
        f.write(str(sum_aucs))
