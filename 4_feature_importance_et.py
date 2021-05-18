# Feature importance based on the Gini importance of features in an Extra Tree (ET) classifier

import pandas as pd
import os

target = "anxiety_status"

data_folder = "./data"
feats_df = pd.read_csv(os.path.join(data_folder, "features.csv"))
labels_df = pd.read_csv(os.path.join(data_folder, "labels.csv"))

cols = ["gender_m", "gender_f", "region_other", 
            "region_east", "region_west", "age_norm"]
for q in range(1, 43):
    for j in range(4):
        cols.append("Q{0}A_{1}".format(q, j))
features = feats_df[cols]
labels = labels_df[[target]].copy()

X = features 
y = labels 

from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X, y)
print(model.feature_importances_) # Use inbuilt class feature_importances of tree based classifiers (Gini importance)

# Plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(15).plot(kind='barh')
plt.show()