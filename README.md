### Introduction

This project builds a neural-network based classifier 
on an intrusion detection dataset.
The intrusion detector is a learnt predictive model (a five-class classifier) 
capable of distinguishing between connections categorized as 
intrusions/attacks and normal connections. 

### Dataset Review

The dataset is the NSL-KDD dataset that you can download from the 
[University of New Brunswick](https://www.unb.ca/cic/datasets/nsl.html)
repository.

The training and the test set are two highlighted below.

`KDDTrain+.TXT`: The full NSL-KDD train-set including attack-type 
labels in CSV format

`KDDTest+.TXT`: The full NSL-KDD test-set including attack-type 
labels in CSV 



### Appendix
You can find a list of the input features at the 
[original KDD site](http://kdd.ics.uci.edu/databases/kddcup99/kddcup.names) 
Note that a few of the features and 'symbolic', i.e., they are categorical 
whereas the rest are continuous. So effort has been spent to 
harmonize these mix of categorical and continuous features.   
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
