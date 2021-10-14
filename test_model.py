import pandas as pd
import numpy as np
import scipy
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, balanced_accuracy_score, confusion_matrix, 
                roc_auc_score, accuracy_score, roc_curve, plot_roc_curve, plot_confusion_matrix)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import (GaussianNB, MultinomialNB, ComplementNB, BernoulliNB, CategoricalNB)


def test_accuracy(models, df_prist, gt_prist):
    TN = 0
    FP = 0
    FN = 0
    TP = 0
    AUC = 0

    for model in models:
        xgbpprist = model.predict(df_prist)
        xgbpprist = pd.DataFrame(xgbpprist)

        # Evaluation
        target_names = ['negative', 'positive']
        cr = classification_report(gt_prist, xgbpprist, target_names=target_names, output_dict=True)
        precision = cr["weighted avg"]["precision"]
        recall = cr["weighted avg"]["recall"]
        f1score = cr["weighted avg"]["f1-score"]

        balanced_accuracy_score(gt_prist, xgbpprist) # Average of Recall on both classes
        tn, fp, fn, tp = confusion_matrix(gt_prist, xgbpprist).ravel()

        TN += tn
        FP += fp
        FN += tp
        TP += fn

        # acc_score = accuracy_score(gt_prist, xgbpprist)
        auc_score = roc_auc_score(gt_prist, xgbpprist)
        AUC += auc_score
        # fpr, tpr, thresh = roc_curve(gt_prist, xgbpprist)
        # plt.plot(fpr, tpr)

        # plot_confusion_matrix(model, df_prist, gt_prist, normalize="all")
        # plt.show()
        # plot_roc_curve(model, df_prist, gt_prist)
        # plt.show()  

    l = len(models)
    return TN/l, FP/l, FN/l, TP/l, AUC/l


def test_model(model_path, test_feats, test_labels):
    features = pd.read_csv(test_feats)
    labels = pd.read_csv(test_labels)
    TN = 0
    FP = 0
    FN = 0
    TP = 0
    AUC = 0

    with open(model_path, "rb") as f:
        obj = pickle.load(f)
        l = len(obj)
        for _, model in obj.items():
            cols = ["gender_m", "gender_f", "region_other", 
                        "region_east", "region_west", "age_norm"]

            for q in model["questions"]:
                for j in range(4):
                    cols.append("Q{0}A_{1}".format(q, j))
            feats = features[cols]

            tn, fp, fn, tp, auc = test_accuracy(model["models"], feats, labels)
            TN += tn
            FP += fp
            FN += tp
            TP += fn
            AUC += auc
    
    print(TN/l, FP/l, FN/l, TP/l)   

    TPR = round(100 * TP / (FN + TP), 2)
    FPR = round(100 * FP / (TN + FP), 2)
    FNR = round(100 * FN / (FN + TP), 2)
    TNR = round(100 * TN / (TN + FP), 2)
    ACC = round(100 * (TP + TN) / (TN + TP + FN + FP), 2)
    PRE = round(100 * TP / (FP + TP), 2)
    F1 = round(200 * TP / (2 * TP + FP + FN), 2)
    BA = round((TPR + TNR) / 2, 2)
    AUC = round(100 * AUC / l, 2)

    print(TPR, FPR, FNR, TNR)
    print(ACC, AUC, PRE, BA, F1) 


if __name__ == "__main__":
    test_model("./models/models_xgb downsample-anxiety.bin", "./data/prist_features.csv", "./data/prist_labels.csv")