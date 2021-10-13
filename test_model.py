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
        print(tn, fp, fn, tp)

        acc_score = accuracy_score(gt_prist, xgbpprist)
        auc_score = roc_auc_score(gt_prist, xgbpprist)
        fpr, tpr, thresh = roc_curve(gt_prist, xgbpprist)
        plt.plot(fpr, tpr)

        plot_confusion_matrix(model, df_prist, gt_prist, normalize="all")
        plt.show()
        plot_roc_curve(model, df_prist, gt_prist)
        plt.show()  


def test_model(model_path, test_feats, test_labels):
    df_prist = pd.read_csv(test_feats)
    gt_prist = pd.read_csv(test_labels)

    with open(model_path, "rb") as f:
        obj = pickle.load(f)
        for _, model in obj.items():
            test_accuracy(model["models"], df_prist, gt_prist)


if __name__ == "__main__":
    test_model("./models/models_ensemble.bin", "./data/prist_features.csv", "./data/prist_labels.csv")