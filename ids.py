import os
import pandas
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

root_folder = '/media/auro/RAID 5/networking'
train_dataset = 'KDDTrain+.txt'
test_dataset = 'KDDTest+.txt'

# categories + 'normal'
normal = ['normal']
DOS = ['apache2', 'back', 'land', 'mailbomb', 'neptune', 'pod', 'processtable', 'smurf', 'teardrop', 'udpstorm']
probe = ['ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan']
R2L = ['spy', 'warezclient', 'ftp_write', 'guess_passwd', 'httptunnel', 'imap', 'multihop',
       'named', 'phf', 'sendmail', 'snmpgetattack', 'warezmaster', 'xlock', 'xsnoop']
U2R = ['buffer_overflow', 'loadmodule', 'perl', 'ps', 'rootkit', 'snmpguess', 'sqlattack', 'worm xterm']


def attack_to_class(df):
    """ converts the attacks to N=4 classes. Model output os one of these classes plus normal"""
    for i in df.index:
        if df[i] in DOS:
            df.at[i] = 'DOS'
        if df[i] in probe:
            df.at[i] = 'probe'
        if df[i] in R2L:
            df.at[i] = 'R2L'
        if df[i] in U2R:
            df.at[i] = 'U2R'
    return df


all_data = pandas.read_csv(os.path.join(root_folder, train_dataset), header=None)
print('read records', len(all_data))
all_test = pandas.read_csv(os.path.join(root_folder, test_dataset), header=None)
print('read records', len(all_test))
print('test cols', all_test.columns)
all_data = all_data.append(all_test, ignore_index=True)
print('total train + test records', len(all_data))

col_grp_1 = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
             'wrong_fragment', 'urgent']

col_grp_2 = ['hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
             'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
             'is_host_login', 'is_guest_login']

col_grp_3 = ['count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
             'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate']

col_grp_4 = ['dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
             'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
             'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']

columns = col_grp_1 + col_grp_2 + col_grp_3 + col_grp_4
all_data.columns = columns + ['class', 'successful_pred']

print(len(columns))
print(all_data.head())
categorical_cols = ['protocol_type', 'service', 'flag']
train_categorical_vars = all_data[categorical_cols]
# train_categorical_vars = train_categorical_vars[categorical_cols].astype('category')

print('total categorical vars', train_categorical_vars.columns)
train_continuous_vars = all_data.drop(categorical_cols + ['class', 'successful_pred'], axis=1)
print('train continuous vars', train_continuous_vars.columns)
train_labels = all_data['class']

print(train_categorical_vars.head())
print(train_continuous_vars.head())
print(train_labels.head())

labels = attack_to_class(train_labels)

# encode the labels
# https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
# values = np.array(labels).astype('float32')
# print(values)
# integer encode

label_encoder = LabelEncoder()
encoded_labels = np.floor(label_encoder.fit_transform(labels))
print(encoded_labels)
print('unique_labels', np.unique(encoded_labels))


# creating instance of label_encoder
# label_encoder = LabelEncoder()  # Assigning numerical values and storing in another column
train_categorical_vars['protocol_type'] = label_encoder.fit_transform(train_categorical_vars['protocol_type'])
train_categorical_vars['service'] = label_encoder.fit_transform(train_categorical_vars['protocol_type'])
train_categorical_vars['flag'] = label_encoder.fit_transform(train_categorical_vars['flag'])

# now encode the categorical input
one_hot_encoder = OneHotEncoder()
train_categorical_vars = one_hot_encoder.fit_transform(train_categorical_vars).todense()

print('categorical data:', train_categorical_vars.shape, type(train_categorical_vars))

train_continuous_vars = np.array(train_continuous_vars)

data_matrix = np.concatenate([train_continuous_vars, train_categorical_vars], axis=1)
print(data_matrix.shape)
print(encoded_labels.shape)

train_matrix = data_matrix[:25000]
print('train matrix', train_matrix.shape, type(train_matrix))
trains_labels = encoded_labels[:25000]
print('train labels', train_labels.shape, type(train_labels))
test_matrix = data_matrix[25000:]
print('test matrix', test_matrix.shape, type(test_matrix))
test_labels = encoded_labels[25000:]


model = Sequential([
    Dense(64, activation='relu', input_shape=[55]),
    Dense(5, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_matrix, trains_labels, epochs=10, validation_data=(test_matrix, test_labels), verbose=2)