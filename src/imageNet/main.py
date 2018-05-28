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
from dataset import Data
from optimizer import Optimizer
from torchvision.models import resnet152, resnet18, resnet50, vgg16_bn
from torchvision import transforms
from image_preprocess import RandomCrop, Rescale, ToTensor


def main(epochs, is_train=1):
    # initialize
    is_train = True if is_train == 1 else False
    n_epochs = epochs
    use_cuda = torch.cuda.is_available()
    checkpoint_path = None

    # learning parameters
    save_epoch = 1
    batch_size = 256
    learning_rate = 0.01
    output_size = 1
    val_ratio = 0.2

    model = resnet152(pretrained=True)
    # model = vgg16_bn(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, output_size)
    with open('../../data/pickle/item_id_dict.pkl', mode='rb') as f:
        item_id_dict = pickle.load(f)

    if torch.cuda.is_available():
        print('use cuda')
        device = "cuda" if use_cuda else "cpu"
        model.to(device)

    if is_train:
        df = pd.read_csv('../../data/unzipped/train.csv')
        df = df.sample(frac=1, random_state=114514).reset_index(drop=True)
        train_df = df[int(len(df) * val_ratio):].reset_index(drop=True)
        val_df = df[:int(len(df) * val_ratio)].reset_index(drop=True)

        # debug
        # train_df = df[600:3000].reset_index(drop=True)
        # val_df = df[:600].reset_index(drop=True)

        del df
        gc.collect()
        print('data loaded!')

        train_loader = dataset(train_df, batch_size, is_train)
        val_loader = dataset(val_df, batch_size, is_train)

        optimizer = Optimizer(model, model.fc.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()
        model = nn.DataParallel(model)
        trainer = Trainer(train_loader, val_loader,  model, criterion,
                          optimizer, use_cuda, n_epochs, save_epoch)
        # training
        print('now training')
        trainer.train()
        print('finished training!!!')

    else:
        print('test prediction')
        test_df = pd.read_csv('../../data/unzipped/test.csv')
        with open('../../data/pickle/item_id_dict.pkl', mode='rb') as f:
            item_id_dict = pickle.load(f)

        test_loader = dataset(
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


def dataset(data, batch_size, is_train):
    dataset = Data(data, is_train, transforms=transforms.Compose(
        [Rescale(256), RandomCrop(224), ToTensor()]))
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=16)

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

    # try:
    #     os.makedirs('../log/' + output_dir_name)
    # except FileExistsError as e:
    #     pass

    output_path = '../' + output_dir_name + file_name
    main(epochs, is_train)
    sum_aucs = 0

    # with open(output_path, 'w') as f:
    #     f.write(str(sum_aucs))
