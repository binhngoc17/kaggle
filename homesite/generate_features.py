import pandas
import datetime
from util import encode_onehot, is_number, locale


def generate_features(name):
    data = pandas.read_csv('homesite/{}.csv'.format(name))
    processed_data = pandas.DataFrame()
    onehot_encode_fields = []
    dropped_fields = [
        'PersonalField16',
        'PersonalField17',
        'PersonalField18',
        'PersonalField19',
    ]

    for k in data.keys():
        if is_number(data[k][1]):
            processed_data[k] = data[k]
        elif k == 'Original_Quote_Date':
            months = []
            years = []
            for d in data[k]:
                dt = datetime.datetime.strptime(d, '%Y-%m-%d')
                months.append(dt.month)
                years.append(dt.year)
            processed_data['Month'] = pandas.Series(months)
            processed_data['Year'] = pandas.Series(years)
        else:
            if k in dropped_fields:
                continue
            onehot_encode_fields.append(k)

    df = encode_onehot(data, cols=onehot_encode_fields)

    for k in df.keys():
        if is_number(df[k][1]):
            print k
            processed_data[k] = df[k].apply(lambda x: locale.atof(x) if isinstance(x, basestring) else x)
        elif k == 'Original_Quote_Date':
            months = []
            years = []
            for d in df[k]:
                dt = datetime.datetime.strptime(d, '%Y-%m-%d')
                months.append(dt.month)
                years.append(dt.year)
            processed_data['Month'] = pandas.Series(months)
            processed_data['Year'] = pandas.Series(years)


    processed_data.to_csv('homesite/processed_features{}.csv'.format(name), quotechar='"')

if __name__ == '__main__':
    generate_features('test')