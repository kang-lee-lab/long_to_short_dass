# Model training for Scikit-learn based models 
# (Random Forest, SVM, Logistic Regression, XGBoost, MLP)

import pandas as pd
import numpy as np
import scipy
import os
import random
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, balanced_accuracy_score, confusion_matrix, 
                roc_auc_score, accuracy_score, roc_curve, plot_roc_curve, plot_confusion_matrix)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier


def confidence_interval(data, confidence=0.95):
    # Calculate confidence interval
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n-1)
    return m-h, m+h


question_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]         # Numbers of questions from DASS to run through
target = "anxiety_status"
models_to_train = 1        # Number of models for each number of questions from DASS
models_per_question = 50    # Number of ensembles per model
test_split = 0.1
model_type = "lr"          # Specify model type (xgb, rf, lr, svm, mlp)
seed = 42
random.seed(seed)


ACCS = []
AUCS = []
PRES = []
RECS = []
F1S = []
AUC_STDEV = []
F1_STDEV = []
AUC_95CI_U = []
AUC_95CI_D = []
F1_95CI_U = []
F1_95CI_D = []


data_folder = "./data"
models_folder = "./models"

feats_df = pd.read_csv(os.path.join(data_folder, "features.csv"))
labels_df = pd.read_csv(os.path.join(data_folder, "labels.csv"))

questions = [20, 9, 30, 11, 19, 2, 36, 28, 4, 23, 7, 27] # dropped q1, q18, q40
# questions = [20, 9, 40, 30, 11, 19, 2, 36, 28, 4, 1, 23, 7, 27, 18] 
#questions = [15, 21, 41, 1, 32, 13, 36, 31, 4, 18]          # Change the questions
# [21, 7, 18, 11, 20, 4, 6, 1, 36, 40, 23] 

# For different numbers of questions from DASS-42
for num_questions in question_numbers:
    models = {}

    accs = []
    aucs = []
    pres = []
    recs = []
    f1s = []
    auc_stdev = []
    f1_stdev = []
    auc_95ci_u = []
    auc_95ci_d = []
    f1_95ci_u = []
    f1_95ci_d = []

    model_num = 0
    for a in range(models_to_train):
        model = {}

        print("Training model", a)
        cols = ["gender_m", "gender_f", "region_other", 
                    "region_east", "region_west", "age_norm"]

        if num_questions == 1:
            question_nums = [questions[a]]
        else:
            question_nums = random.sample(questions, num_questions)
        for j in range(len(question_nums)):
            question_nums[j] = int(question_nums[j])
        question_nums.sort()

        for q in question_nums:
            for j in range(4):
                cols.append("Q{0}A_{1}".format(q, j))
        features = feats_df[cols]

        labels = labels_df[[target]].copy()

        np.random.seed(seed)
        shufId = np.random.permutation(int(len(labels)))
        index = int(test_split * len(labels.index))

        df_prist = features.iloc[shufId[0:index]]
        df_trainvalid = features.iloc[shufId[index:-1]]

        gt_prist = labels.iloc[shufId[0:index]]
        gt_trainvalid = labels.iloc[shufId[index:-1]]

        df_prist.to_csv(os.path.join(data_folder, "prist_features.csv"), index=False)
        gt_prist.to_csv(os.path.join(data_folder, "prist_labels.csv"), index=False)

        accs1 = []
        aucs1 = []
        pres1 = []
        recs1 = []
        f1s1 = []
        ensemble_models = []

        for b in range(models_per_question):
            if b % 10 == 0:
                print("Training iteration", b)

            np.random.seed(b)
            shufId = np.random.permutation(int(len(gt_trainvalid)))
            index = int((1/9) * len(gt_trainvalid.index))

            df_valid = df_trainvalid.iloc[shufId[0:index]]
            df_train = df_trainvalid.iloc[shufId[index:-1]]

            gt_valid = gt_trainvalid.iloc[shufId[0:index]]
            gt_train = gt_trainvalid.iloc[shufId[index:-1]]

            df_valid = df_valid.reset_index(drop=True)
            df_train = df_train.reset_index(drop=True)

            gt_valid = gt_valid.reset_index(drop=True)
            gt_train = gt_train.reset_index(drop=True)

            if model_type == "lr":
                clf = LogisticRegression(random_state=0)
            elif model_type == "svm":
                clf = SVC()
            elif model_type == "rf":
                clf = RandomForestClassifier(max_depth=4, random_state=0)
            elif model_type == "xgb":
                nest = 100
                md = 10
                nj = -1
                clf = XGBClassifier(n_estimators=nest, n_jobs=nj, max_depth=md, objective='reg:logistic')
                # clf = GradientBoostingClassifier
            elif model_type == "mlp":
                clf = MLPClassifier()
            else:
                print("INVALID MODEL TYPE")
            clf.fit(df_train, gt_train.values.ravel())

            xgbpprist = clf.predict(df_prist)
            xgbpprist = pd.DataFrame(xgbpprist)

            # Evaluation
            target_names = ['negative', 'positive']
            cr = classification_report(gt_prist, xgbpprist, target_names=target_names, output_dict=True)
            precision = cr["weighted avg"]["precision"]
            recall = cr["weighted avg"]["recall"]
            f1score = cr["weighted avg"]["f1-score"]

            # balanced_accuracy_score(gt_prist, xgbpprist) # Average of Recall on both classes
            # from sklearn.metrics import confusion_matrix
            # tn, fp, fn, tp = confusion_matrix(gt_prist, xgbpprist).ravel()
            # print(tn, fp, fn, tp)

            acc_score = accuracy_score(gt_prist, xgbpprist)
            auc_score = roc_auc_score(gt_prist, xgbpprist)
            fpr, tpr, thresh = roc_curve(gt_prist, xgbpprist)
            plt.plot(fpr, tpr)

            # plot_confusion_matrix(clf, df_prist, gt_prist, normalize="all")
            # plt.show()
            # plot_roc_curve(clf, df_prist, gt_prist)
            # plt.show()            

            accs1.append(acc_score)
            aucs1.append(auc_score)
            pres1.append(precision)
            recs1.append(recall)
            f1s1.append(f1score)
            ensemble_models.append(clf)

        mean_acc1 = np.mean(accs1)
        mean_auc1 = np.mean(aucs1)
        stdev_auc1 = np.std(aucs1)
        ci_auc1_u, ci_auc1_d = confidence_interval(aucs1)
        mean_pre1 = np.mean(pres1)
        mean_rec1 = np.mean(recs1)
        mean_f11 = np.mean(f1s1)
        stdev_f11 = np.std(f1s1)
        ci_f11_u, ci_f11_d = confidence_interval(f1s1)

        accs.append(mean_acc1)
        aucs.append(mean_auc1)
        auc_stdev.append(stdev_auc1)
        auc_95ci_u.append(ci_auc1_u)
        auc_95ci_d.append(ci_auc1_d)
        pres.append(mean_pre1)
        recs.append(mean_rec1)
        f1s.append(mean_f11)
        f1_stdev.append(stdev_f11)
        f1_95ci_u.append(ci_f11_u)
        f1_95ci_d.append(ci_f11_d)

        model["questions"] = question_nums
        model["models"] = ensemble_models
        model["auc_score"] = mean_auc1
        model["f1_score"] = mean_f11

        models[model_num] = model
        model_num += 1

        # if mean_auc1 > 0.90 and mean_f11 > 0.90:
        #     models[model_num] = model
        #     model_num += 1
        #     plt.xlim([0.0, 1.0])
        #     plt.ylim([0.0, 1.0])
        #     plt.xlabel("False Positive Rate")
        #     plt.ylabel("True Positive Rate")
        #     plt.plot([0.0, 1.0], [0.0, 1.0], linestyle='--')
        #     plt.show()
        # plt.cla()

    # plt.show()

    mean_acc = np.mean(accs)
    mean_auc = np.mean(aucs)
    stdev_auc = np.mean(auc_stdev)
    ci_auc_u = np.mean(auc_95ci_u)
    ci_auc_d = np.mean(auc_95ci_d)
    mean_pre = np.mean(pres)
    mean_rec = np.mean(recs)
    mean_f1  = np.mean(f1s)
    stdev_f1 = np.mean(f1_stdev)
    ci_f1_u = np.mean(f1_95ci_u)
    ci_f1_d = np.mean(f1_95ci_d)

    percentile_list = pd.DataFrame(
    {
        'accuracy': accs,
        'auc_roc': aucs,
        'auc_stdev': auc_stdev,
        'auc_95ci_u': auc_95ci_u,
        'auc_95ci_d': auc_95ci_d,
        'precision': pres,
        'recall': recs,
        'f1_score': f1s,
        'f1_stdev': f1_stdev,
        'f1_95ci_u': f1_95ci_u,
        'f1_95ci_d': f1_95ci_d,
    })
    percentile_list.to_csv('./data/results_{}.csv'.format(model_type), mode='a', header=True)

    print("\nNumber of questions:", num_questions)
    # print("Mean Accuracy :", mean_acc)
    print("Mean AUC      :", mean_auc)
    print("Stdev AUC     :", stdev_auc)
    print("95th CI AUC   :", ci_auc_u, ci_auc_d)
    # print("Mean Precision:", mean_pre)
    # print("Mean Recall   :", mean_rec)
    print("Mean F1-Score :", mean_f1)
    print("Stdev F1      :", stdev_f1)
    print("95th CI F1    :", ci_f1_u, ci_f1_d)

    ACCS.append(mean_acc)
    AUCS.append(mean_auc)
    AUC_STDEV.append(stdev_auc)
    AUC_95CI_U.append(ci_auc_u)
    AUC_95CI_D.append(ci_auc_d)
    PRES.append(mean_pre)
    RECS.append(mean_rec)
    F1S.append(mean_f1)
    F1_STDEV.append(stdev_f1)
    F1_95CI_U.append(ci_f1_u)
    F1_95CI_D.append(ci_f1_d)

    with open("./data/models_{}.bin".format(model_type), "wb") as f:
        pickle.dump(models, f)

print("\nAll accuracies:", ACCS)
print("All AUCs:", AUCS)
print("Stdev of AUCs:", AUC_STDEV)
print("95th CI of AUCs:", AUC_95CI_U)
print("95th CI of AUCs:", AUC_95CI_D)
print("All precisions:", PRES)
print("All recalls:", RECS)
print("All F1s:", F1S)
print("Stdev of F1s:", F1_STDEV)
print("95th CI of F1s:", F1_95CI_U)
print("95th CI of F1s:", F1_95CI_D)

# Plot accuracy results
plt.figure(figsize=(10,10)) # Make new figure
plt.plot(question_numbers, ACCS)
plt.plot(question_numbers, AUCS)
plt.plot(question_numbers, F1S)
plt.plot(question_numbers, PRES)
plt.plot(question_numbers, RECS)
plt.xlabel("Number of DASS questions")
plt.ylabel("Accuracy")
plt.legend(["Accuracy score", "AUC ROC score", "F1 score", "Precision", "Recall"])
plt.show()