# Data inspection and filtering

import os
import json
import pandas as pd
import numpy as np
import pycountry_convert as pc

data_folder = "./data"
target = "depression"   # "anxiety", "depression" or "stress"
level = "moderate"   # moderate or severe


def gen_id(row):
    return "DASS42_P{}".format(format(row["row_num"], '05d'))


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
dataset["row_num"] = np.arange(len(dataset))
dataset["ID"] = dataset.apply(lambda row: gen_id(row), axis=1)
dataset["agegroup"] = dataset.apply(lambda row: encode_age(row), axis=1)
dataset["continent"] = dataset.apply(lambda row: encode_continent(row), axis=1)
dataset["region"] = dataset.apply(lambda row: encode_country(row), axis=1)
dataset["{}_score".format(target)] = dataset.apply(lambda row: calc_score(row), axis=1)
dataset["{}_status".format(target)] = dataset.apply(lambda row: categorize(row), axis=1)

print("Before filtering:")
print(dataset['gender'].value_counts())
print(dataset['age'].value_counts())
print(dataset['age'].mean(), dataset['age'].std())
print(dataset['continent'].value_counts())
print(dataset['region'].value_counts())
print(dataset['agegroup'].value_counts())
print(dataset["{}_status".format(target)].value_counts())

print("\nBreakdown by continent:")
df1 = dataset[dataset['continent'] == 'AS']
print(df1['age'].mean(), df1['age'].std())
print(df1['gender'].value_counts())
print(df1["{}_status".format(target)].value_counts())

df2 = dataset[dataset['continent'] == 'NA']
print(df2['age'].mean(), df2['age'].std())
print(df2['gender'].value_counts())
print(df2["{}_status".format(target)].value_counts())

df3 = dataset[dataset['continent'] == 'EU']
print(df3['age'].mean(), df3['age'].std())
print(df3['gender'].value_counts())
print(df3["{}_status".format(target)].value_counts())

df4 = dataset[dataset['continent'] == 'SA']
print(df4['age'].mean(), df4['age'].std())
print(df4['gender'].value_counts())
print(df4["{}_status".format(target)].value_counts())

df5 = dataset[dataset['continent'] == 'AF']
print(df5['age'].mean(), df5['age'].std())
print(df5['gender'].value_counts())
print(df5["{}_status".format(target)].value_counts())

df6 = dataset[dataset['continent'] == 'OC']
print(df6['age'].mean(), df6['age'].std())
print(df6['gender'].value_counts())
print(df6["{}_status".format(target)].value_counts())


# Filter data
dataset = dataset.drop(dataset[(dataset['gender'] == 0) | (dataset['gender'] == 3)].index)  # Male and females only
dataset = dataset[dataset['age'] >= 18]  # Adults only
dataset = dataset[dataset['region'] != ""]  # Must have region

print("\nAfter filtering:")
print(dataset['gender'].value_counts())
print(dataset['agegroup'].value_counts())
print(dataset['age'].mean(), dataset['age'].std())
print(dataset['continent'].value_counts())
print(dataset['region'].value_counts())
print(dataset["{}_status".format(target)].value_counts())

print("\nBreakdown by continent:")
df1 = dataset[dataset['continent'] == 'AS']
print(df1['age'].mean(), df1['age'].std())
print(df1['gender'].value_counts())
print(df1["{}_status".format(target)].value_counts())

df2 = dataset[dataset['continent'] == 'NA']
print(df2['age'].mean(), df2['age'].std())
print(df2['gender'].value_counts())
print(df2["{}_status".format(target)].value_counts())

df3 = dataset[dataset['continent'] == 'EU']
print(df3['age'].mean(), df3['age'].std())
print(df3['gender'].value_counts())
print(df3["{}_status".format(target)].value_counts())

df4 = dataset[dataset['continent'] == 'SA']
print(df4['age'].mean(), df4['age'].std())
print(df4['gender'].value_counts())
print(df4["{}_status".format(target)].value_counts())

df5 = dataset[dataset['continent'] == 'AF']
print(df5['age'].mean(), df5['age'].std())
print(df5['gender'].value_counts())
print(df5["{}_status".format(target)].value_counts())

df6 = dataset[dataset['continent'] == 'OC']
print(df6['age'].mean(), df6['age'].std())
print(df6['gender'].value_counts())
print(df6["{}_status".format(target)].value_counts())

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
dataset.to_csv(os.path.join(data_folder, "data_filtered.csv"), index=None)