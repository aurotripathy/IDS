import os
import pandas
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import plot_graphs

root_folder = '/media/auro/RAID 5/networking'
train_dataset = 'KDDTrain+.txt'
test_dataset = 'KDDTest+.txt'

# TODO shuffle data

def attack_to_class(df):
    """ converts the attacks to N=5 classes. Model output is one of these classes plus normal"""
    # categories + 'normal'
    normal = ['normal']
    dos = ['apache2', 'back', 'land', 'mailbomb', 'neptune', 'pod', 'processtable', 'smurf', 'teardrop', 'udpstorm']
    probe = ['ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan']
    r2l = ['spy', 'warezclient', 'ftp_write', 'guess_passwd', 'httptunnel', 'imap', 'multihop',
           'named', 'phf', 'sendmail', 'snmpgetattack', 'warezmaster', 'xlock', 'xsnoop']
    u2r = ['buffer_overflow', 'loadmodule', 'perl', 'ps', 'rootkit', 'snmpguess', 'sqlattack', 'worm', 'xterm']
    dos_count = probe_count = r2l_count = u2r_count = normal_count = 0
    for i in df.index:
        if df[i] in dos:
            df.at[i] = 'DOS'
            dos_count += 1
            continue
        if df[i] in probe:
            df.at[i] = 'probe'
            probe_count += 1
            continue
        if df[i] in r2l:
            df.at[i] = 'R2L'
            r2l_count += 1
            continue
        if df[i] in u2r:
            df.at[i] = 'U2R'
            u2r_count += 1
            continue
        # else 'normal'
        if df[i] in normal:
            normal_count += 1
            continue
    print('normal count:', normal_count)
    print('dos count:', dos_count)
    print('probe count:', probe_count)
    print('r2l count:', r2l_count)
    print('u2r count:', u2r_count)
    print('total count:', normal_count + dos_count + probe_count + r2l_count + u2r_count)
    return df


all_data = pandas.read_csv(os.path.join(root_folder, train_dataset), header=None)
print('read records', len(all_data))
all_test = pandas.read_csv(os.path.join(root_folder, test_dataset), header=None)
print('read records', len(all_test))
print('test cols', all_test.columns)
all_data = all_data.append(all_test, ignore_index=True)
print('total train + test records', len(all_data))

column_grp_1 = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
                'wrong_fragment', 'urgent']

column_grp_2 = ['hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
                'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
                'is_host_login', 'is_guest_login']

column_grp_3 = ['count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
                'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate']

column_grp_4 = ['dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']

columns = column_grp_1 + column_grp_2 + column_grp_3 + column_grp_4
all_data.columns = columns + ['class', 'success_score']  # success_score unused

print(len(columns))
print(all_data.head())
categorical_cols = ['protocol_type', 'service', 'flag']
all_categorical_vars = all_data[categorical_cols]
all_categorical_vars = all_categorical_vars[categorical_cols].astype('category')

print('total categorical vars', all_categorical_vars.columns)
all_continuous_vars = all_data.drop(categorical_cols + ['class', 'success_score'], axis=1)
print('train continuous vars', all_continuous_vars.columns)
all_labels = all_data['class']

print(all_categorical_vars.head())
print(all_continuous_vars.head())
print(all_labels.head())

all_labels = attack_to_class(all_labels)

# encode the labels
label_encoder = LabelEncoder()
all_encoded_labels = np.floor(label_encoder.fit_transform(all_labels))
print(all_encoded_labels)

total_classes = len(np.unique(all_encoded_labels))
print('Unique labels:', np.unique(all_encoded_labels))

all_encoded_labels = to_categorical(all_encoded_labels)
print('labels categorical')
print(all_encoded_labels)

# Encode categories
all_categorical_vars['protocol_type'] = label_encoder.fit_transform(all_categorical_vars['protocol_type'])
all_categorical_vars['service'] = label_encoder.fit_transform(all_categorical_vars['protocol_type'])
all_categorical_vars['flag'] = label_encoder.fit_transform(all_categorical_vars['flag'])
print('unique protocol types', np.unique(all_categorical_vars['protocol_type']))
print('unique service types', np.unique(all_categorical_vars['service']))
print('unique flag types', np.unique(all_categorical_vars['flag']))

# One-hot encode the categorical inputs
one_hot_encoder = OneHotEncoder()
all_categorical_vars = one_hot_encoder.fit_transform(all_categorical_vars).toarray()

print('categorical data shape:', all_categorical_vars.shape, type(all_categorical_vars))

all_continuous_vars = np.array(all_continuous_vars)

data_matrix = np.concatenate([all_categorical_vars, all_continuous_vars], axis=1)
print('data matrix shape:', data_matrix.shape)
print('labels shape', all_encoded_labels.shape)

train_test_split = 0.9
split_index = int(148517 * train_test_split)
print('split at', split_index)

train_matrix = data_matrix[:split_index]
print('train matrix', train_matrix.shape, type(train_matrix))
train_labels = all_encoded_labels[:split_index]
print('train labels', train_labels.shape, type(train_labels))

scaler = StandardScaler()
train_matrix = scaler.fit_transform(train_matrix)

test_matrix = data_matrix[split_index:]
print('test matrix', test_matrix.shape, type(test_matrix))
test_labels = all_encoded_labels[split_index:]
print('test labels', test_labels.shape, type(test_labels))

test_matrix = scaler.transform(test_matrix)

model = Sequential([
    Dense(50, activation='relu', input_shape=[55]),
    Dropout(0.5),
    Dense(25, activation='relu'),
    Dropout(0.5),
    Dense(total_classes, activation='softmax')
])
print(model.summary())

# this commented code is to scan for a good learning-rate
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epochs: 1e-4 * 10 ** (epochs / 50), verbose=1)
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_matrix, train_labels, epochs=100, validation_data=(test_matrix, test_labels), callbacks=[lr_scheduler], verbose=2)

# visually scan for the appropriate learning rate
plt.semilogx(history.history["lr"], history.history["loss"])
plt.axis([1e-4, 1e-2, 0, 1])
plt.xlabel('Learning rate')
plt.ylabel('Loss')
plt.title('Learning Rate vs Loss')
plt.grid()
plt.show()

# learning rate of 0.0006 found to have minimal loss
nb_epochs = 30
model.compile(optimizer=Adam(lr=0.0006), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_matrix, train_labels, epochs=nb_epochs, validation_data=(test_matrix, test_labels), verbose=2)

epoch_range = range(nb_epochs)
acc = history.history['accuracy']
loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
plot_graphs(history, "loss")
plot_graphs(history, "accuracy")
