import pandas
from pandas import Series, DataFrame
import pandas as pd

from util import encode_onehot

train = pandas.read_csv('train.csv', index_col=False)

def process_train_data(train):
    df = encode_onehot(train, cols=['Weekday', 'DepartmentDescription'])
    processed_df = DataFrame()

    # One Hot Encoding for features
    for k in df.keys():
        if k.startswith('DepartmentDescription='):
            processed_df[k] = df.groupby(['VisitNumber'])[k].sum()
        if k.startswith('Weekday='):
            processed_df[k] = df.groupby(['VisitNumber'])[k].max()

    # Process all the class label to categorical values
    train['TripTypeStr'] = train['TripType'].apply(str)
    processed_train = encode_onehot(train, cols=['TripTypeStr'])
    cols_rename = {}

    for k in processed_train.keys():
        if k.startswith('TripTypeStr='):
            cols_rename[k] = k.replace('Str=', '_')
    processed_train.rename(columns=cols_rename,inplace=True)

    for k in processed_train.keys():
        if k.startswith('TripType_'):
            processed_df[k] = processed_train.groupby('VisitNumber')[k].max()
    return processed_df

processed_df = process_train_data(train)

test = pandas.read_csv('test.csv', index_col=False)

def process_test_data(test_data):

    df = encode_onehot(test_data, cols=['Weekday', 'DepartmentDescription'])
    processed_df = DataFrame()
    # One Hot Encoding for features
    for k in df.keys():
        if k.startswith('DepartmentDescription='):
            processed_df[k] = df.groupby(['VisitNumber'])[k].sum()
        if k.startswith('Weekday='):
            processed_df[k] = df.groupby(['VisitNumber'])[k].max()

    return processed_df

process_test_df = process_test_data(test)
targets = [k for k in processed_df.keys() if k.startswith('TripType_')]
features = [k for k in processed_df.keys() if not k.startswith('TripType_')]

processed_df[features].apply(int).to_csv('train_X.csv')
processed_df[targets].apply(int).to_csv('train_y.csv')
test_len = len(process_test_df['Weekday=Monday'])
for f in features:
    if f not in process_test_df.columns:
        process_test_df[f] = Series([0 for i in range(0, test_len)])

def convert_to_int(df):
    for col in df.columns:
        df[col].apply(int)
    return df

convert_to_int(process_test_df)
process_test_df[features].to_csv('test_X.csv')
