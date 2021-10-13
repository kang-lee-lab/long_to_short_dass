import os
import json
import pandas as pd
import numpy as np
import pycountry_convert as pc
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

data_folder = "./data"
target = "anxiety"   # "anxiety", "depression" or "stress"
level = "moderate"   # moderate or severe
seed = 42


def encode_country(row):
    # Encode country into three major regions (east, west, other)
    country_code = row["country"]
    try:
        if country_code and country_code != "NONE":
            continent_name = pc.country_alpha2_to_continent_code(country_code)
            if continent_name == "AS":
                region_name = "east"
            elif continent_name in ["NA", "EU", "OC"]:
                region_name = "west"
            else:
                region_name = "other"
        else:
            region_name = ""
    except:
        region_name = ""
    return region_name


def encode_continent(row):
    # Encode country into three major regions (east, west, other)
    country_code = row["country"]
    try:
        if country_code and country_code != "NONE":
            continent_name = pc.country_alpha2_to_continent_code(country_code)
        else:
            continent_name = ""
    except:
        continent_name = ""
    return continent_name


def encode_age(row):
    # Encode age into groups
    age = int(row["age"])
    if age < 18:
        agegroup = 0
    elif age < 28:
        agegroup = 1
    elif age < 38:
        agegroup = 2
    elif age < 48:
        agegroup = 3
    elif age < 58:
        agegroup = 4
    elif age < 68:
        agegroup = 5
    else:
        agegroup = 6
    return agegroup


def calc_score(row):
    # Calculate DASS-42 score for the target
    with open(os.path.join(data_folder, "dass42_qcategories.json"), "r") as f:
        categories = json.load(f)
    target_questions = [key for key in categories if categories[key] == target]

    score = 0
    for qnum in target_questions:
        score += int(row["Q{}A".format(qnum)])
    return score - len(target_questions)


def categorize(row):
    # Classify as positive or negative (high or low) status based on threshold
    with open(os.path.join(data_folder, "dass42_scoring.json"), "r") as f:
        scoring = json.load(f)
    threshold = scoring["{}_score".format(target)][level]["min"]
    return (1 if row["{}_score".format(target)] >= threshold else 0)


dataset = pd.read_csv("./data/data.csv", sep='\t')
dataset["agegroup"] = dataset.apply(lambda row: encode_age(row), axis=1)
dataset["continent"] = dataset.apply(lambda row: encode_continent(row), axis=1)
dataset["region"] = dataset.apply(lambda row: encode_country(row), axis=1)
dataset["{}_score".format(target)] = dataset.apply(lambda row: calc_score(row), axis=1)
dataset["{}_status".format(target)] = dataset.apply(lambda row: categorize(row), axis=1)

# Filter data
dataset = dataset.drop(dataset[(dataset['gender'] == 0) | (dataset['gender'] == 3)].index)  # Male and females only
dataset = dataset[dataset['age'] >= 18]  # Adults only
dataset = dataset[dataset['region'] != ""]  # Must have region

# Drop unnecessary columns
to_drop = ["source", "screensize", "uniquenetworklocation", 
            "education", "urban", "engnat", "hand", "religion", 
            "orientation", "race", "voted", "married", "major",
            "introelapse", "testelapse", "surveyelapse", "familysize"]
dataset = dataset.drop(to_drop, axis=1)

for col in dataset.columns:
    if "TIPI" in col or "VCL" in col:
        dataset = dataset.drop([col], axis=1)
    elif col[0] == "Q" and (col[-1] == "E" or col[-1] == "I"):
        dataset = dataset.drop([col], axis=1)

# Saved filtered dataset
# dataset.to_csv(os.path.join(data_folder, "data_filtered.csv"), index=None)

# Separate majority and minority classes
df_majority = dataset[dataset["{}_status".format(target)] == 1]
df_minority = dataset[dataset["{}_status".format(target)] == 0]

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

np.random.seed(seed)
shufId = np.random.permutation(int(len(labels_df)))
index = int(0.1 * len(labels_df.index))

df_prist = feats_df.iloc[shufId[0:index]]
gt_prist = labels_df.iloc[shufId[0:index]]

df_prist.to_csv(os.path.join(data_folder, "prist_features.csv"), index=False)
gt_prist.to_csv(os.path.join(data_folder, "prist_labels.csv"), index=False)