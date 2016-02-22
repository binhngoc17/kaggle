import pandas
import xgboost
import numpy

dir_prefix = 'homesite/'

def get_features(train, test):
    train_features = [k for k in train.keys() if k not in ['QuoteConversion_Flag', 'QuoteNumber']]
    test_features = [k for k in test.keys() if k != 'QuoteNumber']
    return list(set(train_features).intersection(set(test_features)))

def get_train_data():
    train = pandas.read_csv(dir_prefix + 'processed_features.csv')
    test = pandas.read_csv(dir_prefix + 'processed_featurestest.csv')
    train = train.fillna(-1)
    test = test.fillna(-1)
    train_y = train['QuoteConversion_Flag']
    train_y.fillna(0)
    features = get_features(train, test)
    train_X = train[features]
    train_X.apply(lambda x: x.fillna(0))
    # Fill NaN values
    # train_X = train_X.apply(lambda x: x.fillna(x.mean()),axis=0)
    features = train_X.keys()
    return train_X, train_y, features

def create_feature_map():
    train = pandas.read_csv(dir_prefix + 'processed_features.csv')
    test = pandas.read_csv(dir_prefix + 'processed_featurestest.csv')
    features = get_features(train, test)
    outfile = open(dir_prefix + 'xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()

# def xgboost_train(dtrain, dvalidation):
#     bst = xgboost.train({
#         'bst:max_depth': 100,
#         'bst:eta': 1,
#         'objective': 'binary:logistic',
#     }, dtrain, 50,[(dvalidation, 'eval'), (dtrain, 'train')])
#
#     bst.save_model('0001.model')

# def cross_validation():
# dtrain = xgboost.DMatrix(train_X, label=train_y)
#     params = {
#         'max_depth': 100, 'eta':1, 'objective':'binary:logistic'
#     }
#     return xgboost.cv(params, dtrain, num_boost_round=2, nfold=5,
#         metrics={'error'}, seed=0)

def train():
    train_X, train_y, features = get_train_data()
    msk = numpy.random.rand(len(train_X)) < 0.8
    validation_X = train_X[~msk]
    train_X = train_X[msk]
    validation_y = train_y[~msk]
    train_y = train_y[msk]
    dtrain = xgboost.DMatrix(train_X, label=train_y)
    dvalidation = xgboost.DMatrix(validation_X, label=validation_y)
    bst = xgboost.train(
        {
            'max_depth': 5,
            'min_child_weight': 1,
            'num_class':2,
            'eval_metric':'auc',
            'colsample_bytree': 0.95,
            'subsample': 0.95,
            'eta': 0.06,
            'objective': 'multi:softprob',
        }, dtrain, 5000,[(dvalidation, 'eval'), (dtrain, 'train')]
    )

    bst.save_model('homesite/0001.model')

def predict():
    train_X, train_y, features = get_train_data()
    test_X = pandas.read_csv(dir_prefix + 'processed_featurestest.csv')
    test_X.apply(lambda x: x.fillna(0))

    # test_X = test_X.apply(lambda x: x.fillna(x.mean()),axis=0)
    dtest = xgboost.DMatrix(test_X[features])
    bst = xgboost.Booster()
    bst.load_model('homesite/0001.model')
    predictions = bst.predict(dtest)
    count = 0
    print '"QuoteNumber","QuoteConversion_Flag"'
    for k, v in enumerate(test_X['QuoteNumber']):
        if predictions[k] > 0.2:
            count += 1
            print '{},1'.format(v)
        else:
            print '{},0'.format(v)

def grid_search():
    pass

if __name__ == '__main__':
    # create_feature_map()
    # print cross_validation()
    train()
    # predict()