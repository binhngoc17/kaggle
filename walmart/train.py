import pandas
import xgboost
from pandas import DataFrame

train_X = pandas.read_csv('train_X.csv')
train_y = pandas.read_csv('train_y.csv')
test_X = pandas.read_csv('test_X.csv')

models = {}

submission = DataFrame()
submission['VisitNumber'] = test_X['VisitNumber']

def cross_validation():
    for k in sorted(train_y.keys()):
        if k.startswith('TripType_'):
            dtrain = xgboost.DMatrix(train_X, label=train_y)
            params = {
                'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic'
            }
            print xgboost.cv(params, dtrain, num_round=2, nfold=5,
                metrics={'error'}, seed=0)
            break

def prediction():
    for k in sorted(train_y.keys()):
        if k.startswith('TripType_'):
            dtrain = xgboost.DMatrix(train_X, label=train_y)
            params = {
                'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic'
            }
            bst = xgboost.train(params, dtrain, 10, [(dvalidation, 'eval'), (dtrain, 'train')])
            models[k].predict(test_X)
            prediction = models[k].predict(test_X)
            submission[k] = prediction
            break

    submission.to_csv('submission.csv')
