import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, balanced_accuracy_score, confusion_matrix, 
                roc_auc_score, accuracy_score)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB


def test_model(model_path, test_data_csv, onehot=False):
    data = pd.read_csv(test_data_csv)
    target = "depression"
    target_col = "{}_status".format(target)
    pred_col = "{}_prob_pred".format(target)
    labels = data[[target_col]]

    with open(model_path, "rb") as f:
        obj = pickle.load(f)
        l = len(obj)

    obj["stats"] = {
        "Dem_Age": {"mean": 23.61, "stdev": 21.58}
    } 
    xgbpprist_prob = pd.DataFrame([0] * len(data))
    for i in range(l):
        # Call predict function
        models = obj[i]

        if not onehot:
            cols = ["Dem_Gender", "Dem_Region", "Dem_Age"]
            for q in models["questions"]:
                cols.append("BS_DASS42_Q{0}".format(q))
        else:
            cols = ["gender_m", "gender_f", "region_other", "region_east", "region_west", "age_norm"]
            for q in models["questions"]:
                for j in range(4):
                    cols.append("Q{0}A_{1}".format(q, j))

        feats = data[cols]
        if not onehot:
            feats["Dem_Age"] = (feats["Dem_Age"] - 23.61) / 21.58

        tot = pd.DataFrame([0] * len(feats))
        for model in models["models"]:
            pred = model.predict_proba(feats)
            tot += pred[:, 1].reshape(-1, 1)
        pred = tot / len(models["models"])
        xgbpprist_prob += pred

    xgbpprist_prob = xgbpprist_prob / l
    xgbpprist = xgbpprist_prob.round(0)

    target_names = ["negative", "positive"]
    cr = classification_report(labels, xgbpprist, target_names=target_names, output_dict=True)
    PRE = cr["weighted avg"]["precision"]
    F1 = cr["weighted avg"]["f1-score"]
    TN, FP, FN, TP = confusion_matrix(labels, xgbpprist).ravel()
    ACC = accuracy_score(labels, xgbpprist)
    AUC = roc_auc_score(labels, xgbpprist)
    
    if pred_col not in data.columns:
        data[pred_col] = xgbpprist_prob
    
    # Calculate overall test accuracy metrics (across all models)
    print("Class 1 count: {0}; Class 0 count: {1}".format(len(labels[labels[target_col] == 1]), len(labels[labels[target_col] == 0])))
    print("TN: {0}; FP: {1}; FN: {2}; TP: {3}".format(round(TN), round(FP), round(FN), round(TP)))  

    TPR = round(100 * TP / (FN + TP), 2)
    FPR = round(100 * FP / (TN + FP), 2)
    FNR = round(100 * FN / (FN + TP), 2)
    TNR = round(100 * TN / (TN + FP), 2)
    ACC = round(100 * (TP + TN) / (TN + TP + FN + FP), 2)
    PRE = round(100 * TP / (FP + TP), 2)
    F1 = round(200 * TP / (2 * TP + FP + FN), 2)
    BACC = round((TPR + TNR) / 2, 2)
    AUC = round(100 * AUC, 2)

    print("TPR: {0}; FPR: {1}; FNR: {2}; TNR: {3}".format(TPR, FPR, FNR, TNR))
    print("ACC: {0}; AUC: {1}; PRE: {2}; REC: {3}; F1: {4}; BACC: {5}".format(ACC, AUC, PRE, TPR, F1, BACC))
        

if __name__ == "__main__":
    test_model("./models/models_xgb downsample-depression1.bin", "./data/test_data_depression_moderate.csv", True)