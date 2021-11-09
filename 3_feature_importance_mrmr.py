# Feature importance based on Minimum Redundancy Maximum Relevance (MRMR) criteria

import os
import pymrmr
import pandas as pd
from sklearn.preprocessing import LabelEncoder

data_folder = "./data"
show_top = 30
target = "depression"   # "anxiety", "depression" or "stress"

# Processed dataset
# train_feats = pd.read_csv(os.path.join(data_folder, "features.csv"))
# train_feats = train_feats.drop(["age_norm", "gender_m", "gender_f", "region_other", "region_east", "region_west"], axis=1)  # Comment this line to include demographics

# print("Processed dataset:")
# print(pymrmr.mRMR(train_feats, 'MIQ', show_top))
# print(pymrmr.mRMR(train_feats, 'MID', show_top))


# Unprocessed dataset
train_feats = pd.read_csv(os.path.join(data_folder, "data_filtered.csv"))

label_encoder = LabelEncoder()
region = label_encoder.fit_transform(train_feats["region"])
region = pd.DataFrame(region)
region.columns = ["region1"]
train_feats = pd.concat([train_feats, region], axis=1)

train_feats = train_feats.drop(["{}_score".format(target), "{}_status".format(target), "country", "agegroup", "continent", "region"], axis=1)
train_feats = train_feats.drop(["gender", "age", "region1"], axis=1)  # Comment this line to include demographics

print("\nUnprocessed dataset:")
print(pymrmr.mRMR(train_feats, 'MIQ', 43))
# print(pymrmr.mRMR(train_feats, 'MID', show_top))