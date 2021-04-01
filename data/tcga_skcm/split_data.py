from os.path import join, dirname
import  logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config_path import MELANOMA_DATA_PATH

# current_dir = dirname(__file__)
# input_dir = 'raw_data'
# output_dir = 'splits'

processed_path = join(MELANOMA_DATA_PATH , 'processed')

# input_dir = join(current_dir, input_dir)
output_dir = join(MELANOMA_DATA_PATH, 'splits')

def get_response():
    response_filename= 'M1000_clinical.csv'
    logging.info('loading response from %s' % response_filename)
    labels = pd.read_csv(join(processed_path, response_filename),  index_col=0)
    # labels = labels.set_index('id')
    df = labels[['subtype']].copy()
    df.columns= ['response']
    pos_ind = df.response == 'BRAF'
    df.loc[:, 'response'] = 0
    df.loc[pos_ind, 'response'] = 1
    return df


response = get_response()
print response.head()
print response.info()
all_ids = response
ids = response.index
y = response['response'].values

ids_train, ids_test, y_train, y_test = train_test_split(ids, y, test_size=0.1, stratify=y, random_state=422342)
ids_train, ids_validate, y_train, y_validate = train_test_split(ids_train, y_train, test_size=len(y_test),
                                                                stratify=y_train, random_state=422342)

test_set = pd.DataFrame({'id': ids_test, 'response': y_test})
train_set = pd.DataFrame({'id': ids_train, 'response': y_train})
validate_set = pd.DataFrame({'id': ids_validate, 'response': y_validate})

print test_set.response.value_counts() / float(test_set.response.value_counts().sum())
print train_set.response.value_counts() / float(train_set.response.value_counts().sum())
print validate_set.response.value_counts() / float(validate_set.response.value_counts().sum())

test_set.to_csv(join(output_dir, 'test_set.csv'))
validate_set.to_csv(join(output_dir, 'validation_set.csv'))
train_set.to_csv(join(output_dir, 'training_set.csv'))

total_number_samples = train_set.shape[0]
number_patients = np.geomspace(100, total_number_samples, 20)
number_patients = [int(s) for s in number_patients][::-1]
print number_patients

for i, n, in enumerate(number_patients):
    if i == 0:
        filename = join(output_dir, 'training_set_0.csv')
        train_set.to_csv(filename)
        continue
    number_samples = n
    print i, number_samples
    print ids_train.shape, y_train.shape
    ids_train, ids_validate, y_train, y_validate = train_test_split(ids_train, y_train, train_size=n,
                                                                    stratify=y_train, random_state=422342)
    print ids_train.shape, y_train.shape
    train_set = pd.DataFrame({'id': ids_train, 'response': y_train})
    filename = join(output_dir, 'training_set_{}.csv'.format(i))
    train_set.to_csv(filename)
