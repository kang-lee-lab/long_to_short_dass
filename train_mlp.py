import pandas as pd
import numpy as np
import scipy
import torch
import time
import math
import json
import os
import random
import matplotlib.pyplot as plt

from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from torch.utils.data import DataLoader
from scipy.stats.stats import pearsonr
from data_preprocessing import train_val_test_split

import torch.utils.data as data

class DASSDataset(data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        feats = self.features[index]
        label = self.labels[index]
        return feats, label


import torch.nn as nn
import torch.nn.functional as F
import torch
import pickle
import json


class MultiLayerPerceptron(nn.Module):
    """
    Customizable PyTorch MLP model
    """
    def __init__(self, input_size, hidden_sizes, output_size, activation_funcs):
        """
        Constructor method; initialize model layers

        Inputs: Input array size (int),
                List of hidden layer sizes (list:int),
                Output array size (int),
                List of activation functions for each layer (list:str; "tanh", "sigmoid", "relu")
        """
        super(MultiLayerPerceptron, self).__init__()
        self.activation_funcs = activation_funcs

        try:
            self.layers = []
            if len(hidden_sizes) == 0:
                self.fc1 = nn.Linear(int(input_size), int(output_size))
                self.layers.append(self.fc1)
            elif len(hidden_sizes) == 1:
                self.fc1 = nn.Linear(int(input_size), int(hidden_sizes[0]))
                self.layers.append(self.fc1)
                self.fc2 = nn.Linear(int(hidden_sizes[0]), int(output_size))
                self.layers.append(self.fc2)
            elif len(hidden_sizes) == 2:
                self.fc1 = nn.Linear(int(input_size), int(hidden_sizes[0]))
                self.layers.append(self.fc1)
                self.fc2 = nn.Linear(int(hidden_sizes[0]), int(hidden_sizes[1]))
                self.layers.append(self.fc2)
                self.fc3 = nn.Linear(int(hidden_sizes[1]), int(output_size))
                self.layers.append(self.fc3)
            elif len(hidden_sizes) == 3:
                self.fc1 = nn.Linear(int(input_size), int(hidden_sizes[0]))
                self.layers.append(self.fc1)
                self.fc2 = nn.Linear(int(hidden_sizes[0]), int(hidden_sizes[1]))
                self.layers.append(self.fc2)
                self.fc3 = nn.Linear(int(hidden_sizes[1]), int(hidden_sizes[2]))
                self.layers.append(self.fc3)
                self.fc4 = nn.Linear(int(hidden_sizes[2]), int(output_size))
                self.layers.append(self.fc4)
        except:
            raise ValueError("Hidden sizes must be a list of integers")
        
    def forward(self, features):
        """
        Forward method; computes model output

        Input: Array of features (torch.Tensor)
        Output: Array of output results (torch.Tensor)
        """
        x = features
        for i in range(len(self.layers)):
            layer = self.layers[i]
            activation_func = self.activation_funcs[i]
            x = layer(x)
            if activation_func == "sigmoid":
                x = torch.sigmoid(x)
            elif activation_func == "tanh":
                x = torch.tanh(x)
            elif activation_func == "relu":
                x = torch.relu(x)
            elif activation_func == "leakyrelu":
                x = nn.functional.leaky_relu(x)
            elif activation_func == "softmax":
                x = F.log_softmax(x)

        return x


def confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m-h, m+h


seed = 42
data_folder = "./data"
models_folder = "./models"

anxiety_questions = [15, 21, 41, 1, 32, 13, 36, 31, 4, 18]
models_per_question = 10


def load_data(batchsize, 
              training_features,
              training_labels,
              validation_features,
              validation_labels,
              test_features,
              test_labels):
    """
    Load train validation test data
    """
    training_data = DASSDataset(training_features, training_labels)
    validation_data = DASSDataset(validation_features, validation_labels)
    test_data = DASSDataset(test_features, test_labels)

    train_loader = DataLoader(training_data, batch_size=batchsize, shuffle=True)
    val_loader = DataLoader(validation_data, batch_size=len(validation_features))
    test_loader = DataLoader(test_data, batch_size=len(test_features))

    return train_loader, val_loader, test_loader


def load_model(learn_rate, 
               input_size, 
               hidden_sizes, 
               output_size, 
               activation_funcs, 
               categorical=False):
    """
    Load model function using PyTorch convention
    """
    model = MultiLayerPerceptron(input_size, hidden_sizes, output_size, activation_funcs)
    loss_fnc = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate)  # Stochastic gradient descent
    return model, loss_fnc, optimizer


def evaluate(model, data_loader, plot=False, categorical=False, advanced_metrics=True):
    """
    Evaluate model on a set of data
    Returns Pearson correlation coefficient and average error
    """
    if categorical:
        total_corr = 0
    else:
        output_arr = np.array([])
        label_arr = np.array([])

    for _, batch in enumerate(data_loader):
        feats, labels = batch
        feats = feats.float()
        outputs = model(feats)

        if categorical:
            corr = np.array((outputs > 0.5).squeeze().long() == labels)
            total_corr += int(corr.sum()/2)
        else:
            outputs = outputs[:, 0]
            labels = labels[:]
            outputs = outputs.detach().numpy()
            labels = labels.detach().numpy()
            output_arr = np.concatenate((output_arr, outputs))
            label_arr = np.concatenate((label_arr, labels))

    # Percent correct
    if categorical:
        corr = float(total_corr) / len(data_loader.dataset)
    else:
        corr = pearsonr(output_arr, label_arr)[0]

    auc = -1
    precision = -1
    recall = -1
    f1score = -1

    if advanced_metrics:
        try:
            outputs_binary = np.array((outputs > 0.5).squeeze().long())[:, 1].reshape(-1, 1)
        except:
            print(outputs)
            outputs_binary = outputs
        labels_binary = labels[:, 1]
        auc = roc_auc_score(labels_binary, outputs_binary)

        from sklearn.metrics import classification_report
        target_names = ['negative', 'positive']
        cr = classification_report(labels_binary, outputs_binary, target_names=target_names, output_dict=True)
        precision = cr["weighted avg"]["precision"]
        recall = cr["weighted avg"]["recall"]
        f1score = cr["weighted avg"]["f1-score"]

    # Plot predicted vs actual
    if plot and not categorical:
        x = [0, 42]
        y = [0, 42]
        plt.plot(x, y, color='black', zorder=1)
        plt.scatter(output_arr, label_arr, s=2, zorder=2)
        plt.show()

    return corr, auc, precision, recall, f1score


def train_one_model(train_feats,
                    train_labels,
                    valid_feats,
                    valid_labels,
                    test_feats,
                    test_labels,
                    input_size=20,
                    output_size=1,
                    batch_size=50, 
                    learn_rate=0.1, 
                    num_epochs=20, 
                    eval_every=10, 
                    hidden_sizes=[15],
                    activation_functions=["sigmoid"],
                    categorical=False):
    """
    Training loop for one model
    """

    t = 0
    i = 0
    model, loss_fnc, optimizer = load_model(learn_rate, 
                                            input_size, 
                                            hidden_sizes, 
                                            output_size, 
                                            activation_functions,
                                            categorical)

    train_loader, val_loader, test_loader = load_data(batch_size,
                                                      train_feats,
                                                      train_labels,
                                                      valid_feats,
                                                      valid_labels,
                                                      test_feats,
                                                      test_labels)

    batches = []
    valid_accuracy = []
    train_accuracy = []

    for epoch in range(num_epochs):
        tot_loss = 0

        for _, batch in enumerate(train_loader):
            feats, label = batch
            optimizer.zero_grad()
            predictions = model(feats.float())

            predictions = predictions.squeeze().float()
            labels = label.squeeze().float()

            try:
                batch_loss = loss_fnc(input=predictions, target=labels)
                tot_loss += batch_loss
                batch_loss.backward()
                optimizer.step()
                batches.append(i)
            except:
                pass

            # Evaluate on validation set
            if (t + 1) % eval_every == 0:
                train_acc, train_err, _, _, _ = evaluate(model, train_loader, categorical=categorical, advanced_metrics=False)
                train_accuracy.append(float(train_acc))

                valid_acc, valid_err, _, _, _ = evaluate(model, val_loader, categorical=categorical)
                valid_accuracy.append(float(valid_acc))

                if math.isnan(train_acc):
                    return None, -1, -1

                # print("Epoch: {}, Step {} | Loss: {} | Train: {}, AUC: {} | Validation: {}, AUC: {}".format(
                #     epoch + 1, t + 1, round(float(tot_loss), 5), round(float(train_acc), 5), 
                #     round(train_err, 5), round(float(valid_acc), 5), round(valid_err, 5)))
                tot_loss = 0

            t += 1
            i += 1

    # Finally evaluate on test set
    test_acc, test_auc, precision, recall, f1score = \
        evaluate(model, test_loader, plot=False, categorical=categorical)
    print("Test Accuracy: {0}, Average Test AUC: {1}".format(test_acc, test_auc))

    return model, test_acc, test_auc, precision, recall, f1score


def train(feats_df, labels_df, target, num_questions=5, num_combinations=5):
    """
    Train multiple models

    Target: depression_score, anxiety_score, stress_score
    """
    questions = anxiety_questions
    labels = labels_df[["anxiety_status"]].copy()

    models_info = {}
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

    for i in range(num_combinations):
        if num_questions == 1:
            question_nums = [questions[i]]
        else:
            question_nums = random.sample(questions, num_questions)
        for j in range(len(question_nums)):
            question_nums[j] = int(question_nums[j])
        question_nums.sort()
        model_info = {
            "questions": question_nums
        }
        cols = ["gender_m", "gender_f", "region_other", 
                    "region_east", "region_west", "age_norm"]
        for q in question_nums:
            for j in range(4):
                cols.append("Q{0}A_{1}".format(q, j))

        accs1 = []
        aucs1 = []
        pres1 = []
        recs1 = []
        f1s1 = []

        for b in range(models_per_question):
            features = feats_df[cols]
            train_feats, train_labels, valid_feats, valid_labels, test_feats, test_labels = \
                train_val_test_split(features, labels, rand=b)

            print("Training model {0}; Iteration {1}; Questions {2}".format(i, b, question_nums))

            input_size = len(cols)
            output_size = (1 if "score" in target else 2)
            is_categorical = (False if "score" in target else True)
            model, test_acc, test_auc, precision, recall, f1score = \
                train_one_model(train_feats,
                                train_labels,
                                valid_feats,
                                valid_labels,
                                test_feats,
                                test_labels,
                                input_size=input_size,
                                output_size=output_size,
                                batch_size=50, 
                                learn_rate=0.01, 
                                num_epochs=10, 
                                eval_every=100, 
                                hidden_sizes=[input_size+10],
                                activation_functions=["sigmoid", "sigmoid"],
                                categorical=is_categorical)
            
            accs1.append(test_acc)
            aucs1.append(test_auc)
            pres1.append(precision)
            recs1.append(recall)
            f1s1.append(f1score)
        
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

    #         if model and test_acc > 0.90:
    #             model_info["test_accuracy"] = test_acc
    #             model_info["test_area_under_curve"] = test_auc
    #             models_info[str(i+1)] = model_info
    #             torch.save(model, os.path.join(models_folder, target, 
    #                                 "{0}_model_{1}.pt".format(target, i+1)))
    #             torch.save(model.state_dict(), os.path.join(models_folder, 
    #                                 target, "{0}_model_{1}.pth".format(target, i+1)))

    # with open(os.path.join(models_folder, target, "{}_models.json".format(target)), "w") as f:
    #     json.dump(models_info, f)

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
    percentile_list.to_csv('./data/results_mlp.csv', mode='a', header=True)

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

    return mean_acc, mean_auc, stdev_auc, ci_auc_u, ci_auc_d, mean_pre, mean_rec, mean_f1, stdev_f1, ci_f1_u, ci_f1_d


def main():
    feats_df = pd.read_csv(os.path.join(data_folder, "features.csv"))
    labels_df = pd.read_csv(os.path.join(data_folder,"labels.csv"))

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

    for n in [1, 2, 3, 4, 5, 6, 7, 8]:
        mean_acc, mean_auc, stdev_auc, ci_auc_u, ci_auc_d, mean_pre, mean_rec, mean_f1, stdev_f1, ci_f1_u, ci_f1_d = \
            train(feats_df, labels_df, "anxiety_status", num_questions=n, num_combinations=10)

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


if __name__ == "__main__":
    main()