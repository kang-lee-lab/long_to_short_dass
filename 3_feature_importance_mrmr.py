# Feature importance based on Minimum Redundancy Maximum Relevance (MRMR) criteria

import os
import pymrmr
import pandas as pd

data_folder = "./data"
train_feats  = pd.read_csv(os.path.join(data_folder, "features.csv"))
train_labels = pd.read_csv(os.path.join(data_folder, "labels.csv"))

print(pymrmr.mRMR(train_feats, 'MIQ', 20))
# print(pymrmr.mRMR(train_feats, 'MID', 20))