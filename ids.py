import os
import pandas
# import numpy as np

root_folder = '/media/auro/RAID 5/networking'
train_dataset = 'KDDTrain+.txt'
# categories + 'normal'
normal = ['normal']
DOS = ['apache2', 'back', 'land', 'mailbomb', 'neptune', 'pod', 'processtable', 'smurf', 'teardrop', 'udpstorm']
probe = ['ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan']
R2L = ['spy', 'warezclient', 'ftp_write', 'guess_passwd', 'httptunnel', 'imap', 'multihop',
       'named', 'phf', 'sendmail', 'snmpgetattack', 'warezmaster', 'xlock', 'xsnoop']
U2R = ['buffer_overflow', 'loadmodule', 'perl', 'ps', 'rootkit', 'snmpguess', 'sqlattack', 'worm xterm']
category_dict = {'DOS': 1, 'probe': 2, 'R2L': 3, 'U2R': 4, 'normal': 5}

protocol = {'tcp': 1, 'udp': 2, 'http': 3}


def text_to_numeric(file_path):
    matrix = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            matrix.append(line.split(',')[:-1])  # drop complexity
    print('total lines', len(lines))

    for x in matrix:
        if x[-1] in DOS:
            x[-1] = category_dict['DOS']
        else:
            if x[-1] in probe:
                x[-1] = category_dict['probe']
            else:
                if x[-1] in R2L:
                    x[-1] = category_dict['R2L']
                else:
                    if x[-1] in U2R:
                        x[-1] = category_dict['U2R']
                    else:
                        if x[-1] in normal:
                            x[-1] = category_dict['normal']

    print(set([x[-1]for x in matrix]))
    # print(matrix[-1])
    return matrix


text_to_numeric(os.path.join(root_folder, train_dataset))

all_data = pandas.read_csv(os.path.join(root_folder, train_dataset))
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
categorical_cols = ['protocol_type', 'service', 'flag', 'land', 'logged_in',
                    'is_host_login', 'is_guest_login']
train_categorical_vars = all_data[categorical_cols]
train_continuous_vars = all_data.drop(categorical_cols + ['class', 'successful_pred'], axis=1)
train_labels = all_data['class']

print(train_categorical_vars.head())
print(train_continuous_vars.head())
print(train_labels.head())


