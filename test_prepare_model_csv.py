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


def test_accuracy(models, df_prist, gt_prist):
    TN = 0
    FP = 0
    FN = 0
    TP = 0
    ACC = 0
    AUC = 0
    PRE = 0
    REC = 0
    F1 = 0

    for model in models:
        # Call predict function
        xgbpprist = model.predict(df_prist)
        xgbpprist = pd.DataFrame(xgbpprist)

        # Evaluate model metrics
        target_names = ['negative', 'positive']
        cr = classification_report(gt_prist, xgbpprist, target_names=target_names, output_dict=True)
        precision = cr["weighted avg"]["precision"]
        recall = cr["weighted avg"]["recall"]
        f1score = cr["weighted avg"]["f1-score"]
        tn, fp, fn, tp = confusion_matrix(gt_prist, xgbpprist).ravel()
        acc_score = accuracy_score(gt_prist, xgbpprist)
        auc_score = roc_auc_score(gt_prist, xgbpprist)

        TN += tn
        FP += fp
        FN += tp
        TP += fn
        ACC += acc_score
        AUC += auc_score
        PRE += precision
        REC += recall
        F1 += f1score

    l = len(models)
    return TN/l, FP/l, FN/l, TP/l, ACC/l, AUC/l, PRE/l, REC/l, F1/l, xgbpprist


def test_model(model_path, test_feats, test_labels):
    features = pd.read_csv(test_feats)
    labels = pd.read_csv(test_labels)
    if "anxiety" in model_path:
        target = "anxiety"
    elif "depression" in model_path:
        target = "depression"
    elif "stress" in model_path:
        target = "stress"
    target_col = "{}_status".format(target)
    
    TN = 0
    FP = 0
    FN = 0
    TP = 0
    AUC = 0

    with open(model_path, "rb") as f:
        obj = pickle.load(f)
        l = len(obj)

    for i, model in obj.items():
        cols = ["Dem_Gender", "Dem_Region", "Dem_Age"]

        for q in model["questions"]:
            cols.append("BS_DASS42_Q{0}".format(q))
            # for j in range(4):
            #     cols.append("Q{0}A_{1}".format(q, j))
        obj[i]["feature_names"] = cols
        feats = features[cols]

        tn, fp, fn, tp, acc, auc, precision, recall, f1score, pred = test_accuracy(model["models"], feats, 
                                            labels[target_col])
        TN += tn
        FP += fp
        FN += tp
        TP += fn
        AUC += auc

        # Record model test accuracy metrics
        obj[i]["test_accuracy_metrics"] = {
            "Accuracy": round(acc * 100, 2),
            "AUC ROC": round(auc * 100, 2),
            "Precision": round(precision * 100, 2),
            "Recall": round(recall * 100, 2),
            "F1 Score": round(f1score * 100, 2)
        }

        if "model_{}_pred".format(i) not in labels.columns:
            labels["model_{}_pred".format(i)] = pred
    
    # Calculate overall test accuracy metrics (across all models)
    print("Class 1 count: {0}; Class 0 count: {1}".format(len(labels[labels[target_col] == 1]), len(labels[labels[target_col] == 0])))
    print("TN: {0}; FP: {1}; FN: {2}; TP: {3}".format(round(TN/l), round(FP/l), round(FN/l), round(TP/l)))  

    TPR = round(100 * TP / (FN + TP), 2)
    FPR = round(100 * FP / (TN + FP), 2)
    FNR = round(100 * FN / (FN + TP), 2)
    TNR = round(100 * TN / (TN + FP), 2)
    ACC = round(100 * (TP + TN) / (TN + TP + FN + FP), 2)
    PRE = round(100 * TP / (FP + TP), 2)
    F1 = round(200 * TP / (2 * TP + FP + FN), 2)
    BACC = round((TPR + TNR) / 2, 2)
    AUC = round(100 * AUC / l, 2)

    print("TPR: {0}; FPR: {1}; FNR: {2}; TNR: {3}".format(TPR, FPR, FNR, TNR))
    print("ACC: {0}; AUC: {1}; PRE: {2}; REC: {3}; F1: {4}".format(ACC, AUC, PRE, TPR, F1))
    labels.to_csv(test_labels, index=False)

    # Save updated model object
    with open(model_path, "wb") as f:
        pickle.dump(obj, f)
        

if __name__ == "__main__":
    test_model("./models/depression/BS_depression_moderate_xgb_01.bin", 
                "./data/prist_features.csv", 
                "./data/prist_labels.csv")