import gc
import torch
import torch.nn as nn
import argparse
import pickle
import pandas as pd
import joblib
from torch.utils.data import DataLoader
from trainer import Trainer
from predictor import Predictor
from dataset import Data
from optimizer import Optimizer
from torchvision.models import resnet152, resnet18, resnet50, vgg16_bn
from feature_extractor import VGG16FeatureExtractor
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
    learning_rate = 0.1
    output_size = 1
    val_ratio = 0.2

    # model = VGG16FeatureExtractor()
    model = vgg16_bn(pretrained=True)

    # for param in model.parameters():
    #    param.requires_grad = False
    # num_features = model.fc.in_features
    # model.fc = nn.Linear(num_features, output_size)
    model.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 512),
        nn.Linear(512, 128),
        nn.Linear(128, 1),
        nn.Sigmoid()
    )
    item_id_dict = joblib.load('../../data/pickle/label_dict.pkl')

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
        # train_df = df[3000:10000].reset_index(drop=True)
        # val_df = df[:3000].reset_index(drop=True)

        del df
        gc.collect()
        print('data loaded!')

        train_loader = dataset(train_df, batch_size, is_train)
        val_loader = dataset(val_df, batch_size, is_train)

        optimizer = Optimizer(
            model, model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss().to(device)
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
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    dataset = Data(data, is_train, transforms=transforms.Compose(
        [Rescale(256), RandomCrop(224), ToTensor(), normalize]))
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=16)
    return data_loader


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
