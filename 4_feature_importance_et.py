# Feature importance based on the Gini importance of features in an Extra Tree (ET) classifier

import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

data_folder = "./data"
show_top = 30
target = "depression"   # "anxiety", "depression" or "stress"


# Processed dataset
# features = pd.read_csv(os.path.join(data_folder, "features.csv"))
# labels = pd.read_csv(os.path.join(data_folder, "labels.csv"))

# # First fit a model for questions plus demographics
# #features = features.drop(["gender_m", "gender_f", "region_other", "region_east", "region_west", "age_norm"], axis=1)  # Comment this line to include demographics

# model = ExtraTreesClassifier()
# model.fit(features, labels)
# print(model.feature_importances_) # Use inbuilt class feature_importances of tree based classifiers (Gini importance)

# # Plot graph of feature importances for better visualization
# feat_importances = pd.Series(model.feature_importances_, index=features.columns)
# feat_importances.nlargest(show_top).plot(kind='barh')
# plt.show()


# Unprocessed dataset
df = pd.read_csv(os.path.join(data_folder, "data_filtered.csv"))
features = df
labels = df["{}_status".format(target)]

label_encoder = LabelEncoder()
region = label_encoder.fit_transform(features["region"])
region = pd.DataFrame(region)
region.columns = ["region1"]
features = pd.concat([features, region], axis=1)

# First fit a model for questions plus demographics
features = features.drop(["{}_score".format(target), "{}_status".format(target), "country", "agegroup", "continent", "region"], axis=1)
#features = features.drop(["gender", "age", "region1"], axis=1)  # Comment this line to include demographics

model = ExtraTreesClassifier()
model.fit(features, labels)
# print(model.feature_importances_) # Use inbuilt class feature_importances of tree based classifiers (Gini importance)

# Plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=features.columns)
feat_importances.nlargest(show_top).plot(kind='barh')
print(feat_importances)
plt.show()