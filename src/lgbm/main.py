import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import gc
import scipy.sparse
from preprocessor import Preprocessor
from sklearn.model_selection import train_test_split


def main():

    # train_image_feature = pd.read_hdf(
    #     '../../data/features/train2_image_feature.h5', 'table')
    # test_image_feature = pd.read_hdf(
    #     '../../data/features/test2_image_feature.h5', 'table')
    # train_diff_day = joblib.load('../../data/features/train_diff_days')
    # test_diff_day = joblib.load('../../data/features/test_diff_days')

    # train_description, train_title, test_description, test_title = get_fasttext_feature()

    # preprocessor = Preprocessor('../../data/features/default_feature.pkl')
    # # preprocessor = Preprocessor()
    # preprocessor.add_feture(train_image_feature, test_image_feature, 'image')
    # preprocessor.add_feture(train_title, test_title, 'title')
    # preprocessor.add_feture(train_description, test_description, 'description')
    # preprocessor.add_feture(train_diff_day, test_diff_day, 'diff_day')
    # features = preprocessor.get_feature_vec()

    # features = joblib.dump(
    #     features, '../../data/features/fasttext_vgg16.joblib', compress=3)

    features = joblib.load('../../data/features/fasttext_vgg16.joblib')
    print('complete data loading')
    print('training start...!')
    train_and_predict(features)


def get_fasttext_feature():
    train_item_id = pd.read_csv('../../data/unzipped/train.csv').item_id
    test_item_id = pd.read_csv('../../data/unzipped/train.csv').item_id
    train_description = joblib.load(
        '../../data/features/train_description_sentence_vec.gz')
    train_title = joblib.load(
        '../../data/features/train_title_sentence_vec.gz')
    test_description = joblib.load(
        '../../data/features/test_description_sentence_vec.gz')
    test_title = joblib.load('../../data/features/test_title_sentence_vec.gz')

    train_description = pd.DataFrame(np.asarray(train_description))
    train_title = pd.DataFrame(np.asarray(train_title))
    test_description = pd.DataFrame(np.asarray(test_description))
    test_title = pd.DataFrame(np.asarray(test_title))

    train_description = pd.concat([train_description, train_item_id], axis=1)
    test_description = pd.concat([test_description, test_item_id], axis=1)
    train_title = pd.concat([train_title, train_item_id], axis=1)
    test_title = pd.concat([test_title, test_item_id], axis=1)

    return train_description, train_title, test_description, test_title


def train_and_predict(features):
    count_feature = joblib.load(
        '../../data/tamaki_feature/count_feature_2.joblib')
    price_feature = joblib.load(
        '../../data/tamaki_feature/price_feature_bigram.joblib')

    price_feature.drop('item_id', axis=1, inplace=True)
    count_feature.drop('item_id', axis=1, inplace=True)

    train_feature = features['x_train']
    test_feature = features['x_test']
    train_target = features['y_train']
    feature_names = features['feature_names']
    categorical = features['categorical']

    train_count_feature = count_feature[:train_feature.shape[0]]
    train_price_feature = price_feature[:train_feature.shape[0]]
    test_count_feature = count_feature[train_feature.shape[0]:]
    test_price_feature = price_feature[train_feature.shape[0]:]

    train_feature = scipy.sparse.hstack([
        train_feature,
        train_count_feature,
        train_price_feature
    ])

    test_feature = scipy.sparse.hstack([
        test_feature,
        test_count_feature,
        test_price_feature
    ])

    feature_names = np.hstack([
        feature_names,
        count_feature.columns,
        price_feature.columns,
    ])

    del test_count_feature, train_count_feature, train_price_feature,
    del test_price_feature, count_feature, price_feature
    gc.collect()

    # parameters
    rounds = 50000
    early_stop_rounds = 500
    num_leaves = 1023
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
        'bagging_freq': 2,
        'verbosity': -1,
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
                      verbose_eval=500)

    sub = pd.read_csv('../../data/unzipped/sample_submission.csv')
    valid_score = evals_result['valid']['rmse'][model.best_iteration - 1]
    sub['deal_probability'] = np.clip(model.predict(
        test_feature, num_iteration=model.best_iteration), 0, 1)

    sub.to_csv(f'val_rmse_{valid_score}_it_{model.best_iteration}_lr_{learning_rate}_num_leaves_{num_leaves}.csv.gz',
               index=False, compression='gzip')


if __name__ == '__main__':
    main()
