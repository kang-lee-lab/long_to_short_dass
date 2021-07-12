# Data pre-processing

import pandas as pd
import numpy as np
import os
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

seed = 42
data_folder = "./data"
target = "depression"   # "anxiety", "depression" or "stress"


def preprocess(data_df):
    """
    Pre-processing: rebalance, one-hot encode, normalize
    """
    # Separate majority and minority classes
    df_majority = data_df[data_df["{}_status".format(target)] == 1]
    df_minority = data_df[data_df["{}_status".format(target)] == 0]
    
    # Upsample minority class
    data_minority = resample(df_minority, 
                            replace=True,                       # sample with replacement
                            n_samples=len(df_majority.index),   # to match majority class
                            random_state=123)                   # reproducible results

    # Downsample majority class
    # data_majority = resample(df_majority, 
    #                         replace=False,                      # sample without replacement
    #                         n_samples=len(df_minority.index),   # to match minority class
    #                         random_state=123)                   # reproducible results

    data_df = pd.concat([df_majority, data_minority])
    data_df = data_df.reset_index(drop=True)

    # Extract the label columns; separate features and labels
    labels_df = data_df[["{}_status".format(target)]].copy()
    feats_df = data_df.drop(["{}_score".format(target), "{}_status".format(target)], axis=1)

    # z-score normalization
    def z_score_norm(row, col, mean, stdev):
        z_score = (float(row[col]) - mean) / stdev
        return float(z_score)

    # One hot encode gender and region
    label_encoder = LabelEncoder()
    oneh_encoder = OneHotEncoder()

    # Gender
    gender = label_encoder.fit_transform(feats_df["gender"])
    gender = pd.DataFrame(gender)
    gender = pd.DataFrame(oneh_encoder.fit_transform(gender).toarray())
    gender.columns = ["gender_m", "gender_f"]

    # Region
    region = label_encoder.fit_transform(feats_df["region"])
    region = pd.DataFrame(region)
    region = pd.DataFrame(oneh_encoder.fit_transform(region).toarray())
    region.columns = ["region_other", "region_east", "region_west"]

    # Combine and remove original columns
    feats_df = feats_df.drop(["gender", "country", "region", "agegroup", "continent"], axis=1)
    feats_df = pd.concat([feats_df, gender, region], axis=1)

    # One-hot encode question answers
    for col in feats_df.columns:
        if col[0] == "Q" and col[-1] == "A":
            temp = label_encoder.fit_transform(feats_df[col])
            temp = pd.DataFrame(temp)
            temp = pd.DataFrame(oneh_encoder.fit_transform(temp).toarray())

            col_names = []
            for c in temp.columns:
                col_names.append("{0}_{1}".format(col, c))
            temp.columns = col_names

            feats_df = feats_df.drop([col], axis=1)
            feats_df = pd.concat([feats_df, temp], axis=1)

    # Normalize numerical columns (Use z-score)
    mean = feats_df["age"].mean()
    stdev = feats_df["age"].std()
    feats_df["age_norm"] = feats_df.apply(
                    lambda row: z_score_norm(row, "age", mean, stdev), axis=1)
    feats_df = feats_df.drop(["age"], axis=1)

    return feats_df, labels_df


def train_val_test_split(feats_df, labels_df, rand=0, save=False):
    """
    Train / validation / test (holdout) dataset split
    """
    feats_arr = np.array(feats_df)
    labels_arr = np.array(labels_df)
    traintest_feats, valid_feats, traintest_labels, valid_labels = \
        train_test_split(feats_arr, labels_arr, test_size=0.10, random_state=seed)
    train_feats, holdout_feats, train_labels, holdout_labels = \
        train_test_split(traintest_feats, traintest_labels, test_size=0.1111, random_state=rand)

    train_feats = train_feats.astype(float)
    train_labels = train_labels.astype(float)
    valid_feats = valid_feats.astype(float)
    valid_labels = valid_labels.astype(float)
    holdout_feats = holdout_feats.astype(float)
    holdout_labels = holdout_labels.astype(float)

    if save:
        train_feats.to_csv(os.path.join(data_folder, "train_feats.csv"), index=None)
        train_labels.to_csv(os.path.join(data_folder, "train_labels.csv"), index=None)
        valid_feats.to_csv(os.path.join(data_folder, "valid_feats.csv"), index=None)
        valid_labels.to_csv(os.path.join(data_folder, "valid_labels.csv"), index=None)
        holdout_feats.to_csv(os.path.join(data_folder, "holdout_feats.csv"), index=None)
        holdout_labels.to_csv(os.path.join(data_folder, "holdout_labels.csv"), index=None)

    return train_feats, train_labels, valid_feats, valid_labels, holdout_feats, holdout_labels


if __name__ == "__main__":
    data = pd.read_csv(os.path.join(data_folder, "data_filtered.csv"))
    feats_df, labels_df = preprocess(data)

    feats_df.to_csv(os.path.join(data_folder, "features.csv"), index=None)
    labels_df.to_csv(os.path.join(data_folder, "labels.csv"), index=None)