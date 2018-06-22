import numpy as np
import pandas as pd
import joblib
import string
import scipy
import gc
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


class Preprocessor():

    def __init__(self, feature_path=None):
        if feature_path:
            self.feature = joblib.load(feature_path)
        else:
            feature_path = '../../data/features/default_feature.pkl'
            train_df = pd.read_csv('../../data/unzipped/train.csv')
            test_df = pd.read_csv('../../data/unzipped/test.csv')
            agg_feature = pd.read_csv(
                '../../data/features/aggregated_features.csv')
            self.train_df = train_df.merge(
                agg_feature, on='user_id', how='left')
            self.test_df = test_df.merge(agg_feature, on='user_id', how='left')
            self.agg_cols = list(agg_feature.columns)[1:]

            del train_df, test_df, agg_feature
            gc.collect()

            self.count_vectorizer_title = CountVectorizer(
                stop_words=stopwords.words('russian'),
                min_df=25, ngram_range=(1, 2))

            self.count_vectorizer_desc = TfidfVectorizer(
                stop_words=stopwords.words(
                    'russian'), ngram_range=(1, 2), max_features=20000)

            self.target_name = 'deal_probability'
            self.feature_names = [
                'num_desc_punct', 'words_vs_unique_description',
                'num_unique_words_description', 'num_unique_words_title',
                'num_words_description', 'num_words_title',
                'avg_times_up_user', 'avg_days_up_user', 'n_user_items',
                'price', 'item_seq_number'
            ]
            self.categorical_feature_names = [
                'image_top_1', 'param_1', 'param_2', 'param_3',
                'city', 'region', 'category_name', 'parent_category_name',
                'user_type'
            ]

            self.feature = self._create_feature()
            self.save_feature(feature_path)

            del (self.train_df, self.test_df, self. agg_cols,
                 self.feature_names, self.categorical_feature_names)
            gc.collect()

    def get_feature_vec(self):
        return {'x_train': self.feature['x_train'],
                'x_test': self.feature['x_test'],
                'y_train': self.feature['y_train'],
                'feature_names': self.feature['feature_names'],
                'categorical': self.feature['categorical_feature_name']
                }

    def add_feture(self, added_train_df, added_test_df, feature_name_prefix=''):

        # 一旦idだけ分離して、それ以外のカラム名を変更
        train_id = pd.DataFrame(added_train_df.item_id)
        test_id = pd.DataFrame(added_test_df.item_id)
        added_train_df.drop('item_id', axis=1, inplace=True)
        added_test_df.drop('item_id', axis=1, inplace=True)

        self._change_column_name(added_train_df, 'image')
        self._change_column_name(added_test_df, 'image')
        added_column_name = added_train_df.columns

        # idをくっつけ直す
        added_train_df = pd.concat(
            [train_id, added_train_df], axis=1)
        added_test_df = pd.concat([test_id, added_test_df], axis=1)

        # 追加するデータがすでにあるデータとおなじ順序になるようにする
        added_train_df = pd.merge(
            pd.DataFrame(self.feature['train_item_id']), added_train_df,
            on='item_id', how='left')
        added_test_df = pd.merge(
            pd.DataFrame(self.feature['test_item_id']), added_test_df,
            on='item_id', how='left')

        # キーとして使ってたitem_idをドロップ
        added_train_df.drop('item_id', axis=1, inplace=True)
        added_test_df.drop('item_id', axis=1, inplace=True)

        # 今までのやつとくっつける
        self.feature['x_train'] = scipy.sparse.hstack([
            self.feature['x_train'],
            added_train_df
        ])

        self.feature['x_test'] = scipy.sparse.hstack([
            self.feature['x_test'],
            added_test_df
        ])

        self.feature['feature_names'] = np.hstack([
            self.feature['feature_names'],
            added_column_name
        ])
        return

    def _create_feature(self, path=None):
        self._description_preprocess()
        # count_feature = self._get_counts_feature()
        self._label_encode()
        feature_names = self.feature_names + self.categorical_feature_names

        # x_train = scipy.sparse.hstack([
        #     count_feature['train_desc'],
        #     count_feature['train_title'],
        #     self.train_df.loc[:, feature_names]
        # ], format='csr')

        # x_test = scipy.sparse.hstack([
        #     count_feature['test_desc'],
        #     count_feature['test_title'],
        #     self.test_df.loc[:, feature_names]
        # ], format='csr')

        x_train = scipy.sparse.csr_matrix(self.train_df.loc[:, feature_names])
        x_test = scipy.sparse.csr_matrix(self.test_df.loc[:, feature_names])

        y_train = self.train_df.loc[:, self.target_name]

        return {
            'x_train': x_train,
            'x_test': x_test,
            'y_train': y_train,
            'categorical_feature_name': self.categorical_feature_names,
            # 'feature_names': self._get_feature_names(),
            'feature_names': self.feature_names + self.categorical_feature_names,
            'train_item_id': self.train_df.item_id,
            'test_item_id': self.test_df.item_id
        }

    def save_feature(self, path):
        joblib.dump(self.feature, path, compress=3)

    def _get_feature_names(self):
        return np.hstack([
            self.count_vectorizer_desc.get_feature_names(),
            self.count_vectorizer_title.get_feature_names(),
            self.feature_names + self.categorical_feature_names
        ])

    def _label_encode(self):
        for feature in self.categorical_feature_names:
            print(f'Transforming {feature}...')
            encoder = LabelEncoder()
            encoder.fit(self.train_df[feature].append(
                self.test_df[feature]).astype(str))

            self.train_df[feature] = encoder.transform(
                self.train_df[feature].astype(str))
            self.test_df[feature] = encoder.transform(
                self.test_df[feature].astype(str))

    def _count(self, l1, l2):
        return sum([1 for x in l1 if x in l2])

    def _description_preprocess(self):

        for df in [self.train_df, self.test_df]:
            df['description'].fillna('unknowndescription', inplace=True)
            df['title'].fillna('unknowntitle', inplace=True)
            df["price"] = np.log(df["price"] + 0.001)
            df["price"].fillna(-999, inplace=True)
            for col in ['description', 'title']:
                df['num_words_' +
                    col] = df[col].apply(lambda comment: len(comment.split()))
                df['num_unique_words_' +
                    col] = df[col].apply(lambda comment: len(set(w for w in comment.split())))
            df['words_vs_unique_title'] = df['num_unique_words_title'] / \
                df['num_words_title'] * 100
            df['words_vs_unique_description'] = df['num_unique_words_description'] / \
                df['num_words_description'] * 100
            df['city'] = df['region'] + '_' + df['city']
            df['num_desc_punct'] = df['description'].apply(
                lambda x: self._count(x, set(string.punctuation)))

            for col in self.agg_cols:
                df[col].fillna(-1, inplace=True)

    def _get_counts_feature(self):
        title_counts = self.count_vectorizer_title.fit_transform(
            self.train_df['title'].append(self.test_df['title']))
        train_title_counts = title_counts[:len(self.train_df)]
        test_title_counts = title_counts[len(self.train_df):]

        desc_counts = self.count_vectorizer_desc.fit_transform(
            self.train_df['description'].append(self.test_df['description']))
        train_desc_counts = desc_counts[:len(self.train_df)]
        test_desc_counts = desc_counts[len(self.train_df):]

        return {
            'train_title': train_title_counts,
            'test_title': test_title_counts,
            'train_desc': train_desc_counts,
            'test_desc': test_desc_counts
        }

    def _change_column_name(self, df, prefix):
        df.rename(columns=lambda x: prefix + '_' + str(x), inplace=True)
