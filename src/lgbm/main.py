import numpy as np
import pandas as pd
import lightgbm as lgb
from preprocessor import Preprocessor
from sklearn.model_selection import train_test_split


def main():

    train_image_feature = pd.read_hdf(
        '../../data/features/train2_image_feature.h5', 'table')
    test_image_feature = pd.read_hdf(
        '../../data/features/test2_image_feature.h5', 'table')
    preprocessor = Preprocessor('../../data/features/default_feature.pkl')
    preprocessor.add_feture(train_image_feature, test_image_feature, 'image')
    features = preprocessor.get_feature_vec()

    train_and_predict(features)


def train_and_predict(features):

    train_feature = features['x_train']
    test_feature = features['x_test']
    train_target = features['y_train']
    feature_names = features['feature_names']
    categorical = features['categorical']

    # parameters
    rounds = 50000
    early_stop_rounds = 500
    num_leaves = 1500
    learning_rate = 0.01

    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': num_leaves,
        'max_depth': -1,
        'learning_rate': learning_rate,
        'feature_fraction': 0.4,
        'bagging_fraction': 0.7,
        'bagging_freq': 3,
        'verbosity': -1,
        'reg_alpha': 2,
        'reg_lambda': 5,
        'max_bin': 255,
    }

    print('Number of features:', len(feature_names))

    x_train, x_val, y_train, y_val = train_test_split(
        train_feature, train_target, test_size=0.2, random_state=114514)

    dtrain = lgb.Dataset(x_train, label=y_train,
                         feature_name=list(feature_names),
                         categorical_feature=categorical)
    dvalid = lgb.Dataset(x_val, label=y_val,
                         feature_name=list(feature_names),
                         categorical_feature=categorical)

    evals_result = {}
    model = lgb.train(params, dtrain,
                      valid_sets=[dvalid],
                      valid_names=['valid'],
                      num_boost_round=rounds,
                      evals_result=evals_result,
                      early_stopping_rounds=early_stop_rounds,
                      verbose_eval=250)

    sub = pd.read_csv('../../data/unzipped/sample_submission.csv')
    valid_score = evals_result['valid']['rmse'][model.best_iteration - 1]
    sub['deal_probability'] = np.clip(model.predict(
        test_feature, num_iteration=model.best_iteration), 0, 1)

    sub.to_csv(f'val_rmse_{valid_score}_it_{model.best_iteration}_lr_{learning_rate}_num_leaves_{num_leaves}.csv.gz',
               index=False, compression='gzip')


if __name__ == '__main__':
    main()
