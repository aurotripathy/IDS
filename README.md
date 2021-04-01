### Summary

This project builds a neural-network based classifier 
on an intrusion detection dataset. Out-of-the-box accuracy is at **89%** with a two-layer neural network.
The intrusion detector is a learnt predictive model (a five-class classifier) 
capable of distinguishing between connections categorized as 
intrusions/attacks and normal connections. 

**No rules, no logic, simply learns from the data.**

### Dataset Review

The dataset is the NSL-KDD dataset that you can download from the 
[University of New Brunswick](https://www.unb.ca/cic/datasets/nsl.html)
repository.

The training and the test set are two highlighted below.

`KDDTrain+.TXT`: The full NSL-KDD train-set including attack-type 
labels in CSV format

`KDDTest+.TXT`: The full NSL-KDD test-set including attack-type 
labels in CSV 

The attack categories are listed below:
```
normal = ['normal']

dos = ['apache2', 'back', 'land', 'mailbomb', 'neptune', 'pod', 
       'processtable', 'smurf', 'teardrop', 'udpstorm']
       
probe = ['ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan']

r2l = ['spy', 'warezclient', 'ftp_write', 'guess_passwd', 
       'httptunnel', 'imap', 'multihop', 'named', 'phf', 'sendmail', 
       'snmpgetattack', 'warezmaster', 'xlock', 'xsnoop']
       
u2r = ['buffer_overflow', 'loadmodule', 'perl', 'ps', 'rootkit', 
       'snmpguess', 'sqlattack', 'worm', 'xterm']
```
For the purposes of classification, we reduce them to just five classes as shown above.
Also, note that the class distribution is in heavy imbalance 
(will be addressed n a future model).

The input consists of columns shown in the appendix. 

### Model
The model is very simple and consists of a two hidden layers of densely (fully) 
connected neurons with dropout layers to reduce over-fitting through regularization. 
```
model = Sequential([
    Dense(50, activation='relu', input_shape=[55]),
    Dropout(0.5),
    Dense(25, activation='relu'),
    Dropout(0.5),
    Dense(total_classes, activation='softmax')
])
```

### Results
The model accuracy is ~89% (scroll to the third graph below).

In the first graph, we search for the best learning rate 
which turns out to be `0.0006` (the lowest point in 
the learning-rate versus loss graph).

![lr-search](/assets/LR-search.png)

![loss](/assets/losses.png)

![lr-search](/assets/accuracies.png)

### Appendix
You can find a list of the input features at the 
[original KDD site](http://kdd.ics.uci.edu/databases/kddcup99/kddcup.names) 
Note that a few of the features are 'symbolic', i.e., they represent categories 
whereas the rest are continuous. So effort has been spent to 
harmonize the mix of categorical and continuous features to train them together.   
```
duration: continuous.
protocol_type: symbolic.
service: symbolic.
flag: symbolic.
src_bytes: continuous.
dst_bytes: continuous.
land: symbolic.
wrong_fragment: continuous.
urgent: continuous.
hot: continuous.
num_failed_logins: continuous.
logged_in: symbolic.
num_compromised: continuous.
root_shell: continuous.
su_attempted: continuous.
num_root: continuous.
num_file_creations: continuous.
num_shells: continuous.
num_access_files: continuous.
num_outbound_cmds: continuous.
is_host_login: symbolic.
is_guest_login: symbolic.
count: continuous.
srv_count: continuous.
serror_rate: continuous.
srv_serror_rate: continuous.
rerror_rate: continuous.
srv_rerror_rate: continuous.
same_srv_rate: continuous.
diff_srv_rate: continuous.
srv_diff_host_rate: continuous.
dst_host_count: continuous.
dst_host_srv_count: continuous.
dst_host_same_srv_rate: continuous.
dst_host_diff_srv_rate: continuous.
dst_host_same_src_port_rate: continuous.
dst_host_srv_diff_host_rate: continuous.
dst_host_serror_rate: continuous.
dst_host_srv_serror_rate: continuous.
dst_host_rerror_rate: continuous.
dst_host_srv_rerror_rate: continuous.
```
