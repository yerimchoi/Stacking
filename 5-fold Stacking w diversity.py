# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 15:22:05 2020

@author: lvaid
"""

import os
import numpy as np
import pandas as pd
from random import randint
from itertools import combinations
import warnings

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score


def randomlist(list):
    index = len(list) - 1
    value = list[randint(0, index)]
    return value

def define_model(model_name):
    global model
    # Define the model
    i = randint(1, 42)
    activation_list = ['relu', 'tanh', 'logistic']
    hidden_list = [(10,), (50,), (100,), (10, 10), (50, 50), (100, 100)]
    solver_list = ['lbfgs', 'sgd', 'adam']
    c_list = [0.01, 0.1, 0.5, 1, 10, 30, 50, 100]
    gamma_list = [0.01, 0.1, 0.5, 1, 5, 10]
    knn_neigh = [3, 5, 7, 9]
    knn_weight = ['uniform', 'distance']
    max_feature_list = ['auto', 'sqrt', 'log2']
    criterion_list = ['gini', 'entropy']
    min_sample_list = [2, 4, 6, 10, 20, 50]

    if model_name == 'neural_network':
        model = MLPClassifier(activation=randomlist(activation_list), hidden_layer_sizes= randomlist(hidden_list),
                              solver=randomlist(solver_list), random_state=i)
    elif model_name == 'logistic_regression':
        model = LogisticRegression(solver='lbfgs', random_state=i, C=randomlist(c_list))
    elif model_name == 'svc':
        model = SVC(random_state=i, C=randomlist(c_list), gamma=randomlist(gamma_list))
    elif model_name == 'decision_tree':
        model = DecisionTreeClassifier(random_state=i, max_features=randomlist(max_feature_list), criterion=randomlist(criterion_list),
                                       min_samples_split=randomlist(min_sample_list))
    elif model_name == 'knn':
        model = KNeighborsClassifier(n_neighbors=randomlist(knn_neigh), weights=randomlist(knn_weight))

    return model

def kfold_data(model_name, train_x, train_y, test_x, n_fold = 5):
    global test_pred_mean, train_score_mean, valid_score_mean

    train_x, train_y, test_x = pd.DataFrame(train_x), pd.DataFrame(train_y), pd.DataFrame(test_x)
    stk = StratifiedKFold(n_splits = n_fold, shuffle=True)
    valid_fold_pred = np.ones((train_x.shape[0], 1))
    test_pred = np.ones((test_x.shape[0], n_fold))

    model = define_model(model_name)
    train_score_list = []
    valid_score_list = []

    for cnt, (train_idx, valid_idx) in enumerate(stk.split(train_x, train_y)):
        train_fold_x, train_fold_y = train_x.iloc[train_idx], train_y.iloc[train_idx]
        valid_fold_x, valid_fold_y = train_x.iloc[valid_idx], train_y.iloc[valid_idx]

        model.fit(train_fold_x, train_fold_y)
        train_score = model.score(train_fold_x, train_fold_y)
        valid_score = model.score(valid_fold_x, valid_fold_y)

        train_score_list.append(train_score)
        valid_score_list.append(valid_score)

        valid_fold_pred[valid_idx, :] = model.predict(valid_fold_x).reshape(-1, 1)
        test_pred[:, cnt] = model.predict(test_x)

    test_pred_mean = np.mean(test_pred, axis=1).reshape(-1, 1)
    train_score_mean = np.mean(train_score_list)
    valid_score_mean = np.mean(valid_score_list)

    return valid_fold_pred, test_pred_mean, train_score_mean, valid_score_mean

def learn_meta_learner(stacked_train, stacked_test, meta_name, train_y, test_y):
    global model
    i = randint(1, 42)
    
    if meta_name == 'neural_network':
        model = MLPClassifier(activation= 'relu', solver='lbfgs', random_state=i)
        model.fit(stacked_train, train_y)
    elif meta_name == 'logistic_regression':
        model = LogisticRegression(solver='lbfgs', random_state=i)
        model.fit(stacked_train, train_y)
    elif meta_name == 'svc':
        model = SVC(random_state=i)
        model.fit(stacked_train, train_y)
    elif meta_name == 'decision_tree':
        model = DecisionTreeClassifier(random_state=i)
        model.fit(stacked_train, train_y)
    elif meta_name == 'knn':
        model = KNeighborsClassifier()
        model.fit(stacked_train, train_y)
    else:
        "Model Name ERROR"
        return False

    stacked_test_pred = model.predict(stacked_test).reshape(-1, 1)
    stacked_test_acc = accuracy_score(stacked_test_pred, test_y)
    
    try:
        stacked_test_auc = roc_auc_score(stacked_test_pred, test_y)
    except ValueError:
        stacked_test_auc = 0

    return stacked_test_acc, stacked_test_auc


class DiversityMeasure:
    def __init__(self, con_list):
        self.con_list = con_list

    def pairmatrix(self, matrix_A, matrix_B):
        # matrix_A == classifer 1, matrix_B == classifier 2
        correct = np.where(matrix_A == matrix_B, 1, 10)
        false = np.where(matrix_A != matrix_B, 1, -10)

        num1 = np.where(matrix_A == 1, 1, 0)
        num4 = np.where(matrix_A == 0, 1, 0)

        n11 = np.sum(np.where(correct == num1, 1, 0))
        n00 = np.sum(np.where(correct == num4, 1, 0))
        n10 = np.sum(np.where(false == num1, 1, 0))
        n01 = np.sum(np.where(false == num4, 1, 0))
        
        q = ((n11 * n00 - n01 * n10) / (n11 * n00 + n01 * n10))
        lo = ((n11 * n00 - n01 * n10) / np.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)))
        dis = ((n01 + n10) / (n11 + n10 + n01 + n00))
        df = (n00 / (n11 + n10 + n01 + n00))
        dc = (n11 / (n11+ n10 + n01 + n00))
        
        div = [q, lo, dis, df, dc]
        return div
    
    def return_measures(self):
        div_list = []
        length = len(self.con_list) - 1
        
        for i in range(length):
            index = length - i
            
            for j in range(index):
                div = self.pairmatrix(self.con_list[i], self.con_list[i + j + 1])
                div_list.append(div)
                
        mean_div = np.mean(div_list, axis = 0)
        return mean_div


if __name__ == "__main__":
    global stacked_train, stacked_test, end_index
    
    model_nlsdk = ['neural_network', 'logistic_regression', 'svc', 'decision_tree', 'knn']
    model_list = []
    for j in range(3, 6):
        models = list(combinations(model_nlsdk, j))
        for length in range(len(models)):
            model_list.append(models[length])

    warnings.filterwarnings('ignore')
    file_path = 'C:/Users/lvaid/skku/Technometrics/에바/'
    path_dir = 'C:/Users/lvaid/skku/Technometrics/Data/Dataset/this laptop'
    file_list = os.listdir(path_dir)
    file_list.sort()

    for model_comb in model_list:
        for file in file_list:
            csv_file = pd.read_csv(path_dir + '/' + file)
            filename = file.rstrip('.csv')
            
            div_result = []
            score_result = []
            
            for i in range(30):
                print("Model ", model_comb, " for ", file, " & ", i, "th")
                
                meta_name = 'logistic_regression'
                x = csv_file.drop(['class'], axis=1)
                y = csv_file['class']
                train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.3)

                ss = StandardScaler()
                ss.fit(train_x)
                train_x = ss.transform(train_x)
                test_x = ss.transform(test_x)

                con_list = []
                score_list = []
                for m in range(len(model_comb)):
                    valid_yhat, test_yhat, train_score_mean, valid_score_mean = \
                        kfold_data(model_comb[m], train_x, train_y, test_x, 5)

                    # save yhat
                    yhat = valid_yhat.reshape(valid_yhat.shape[0], )
                    correct_or_not = np.where(yhat == train_y, 1, 0)
                    con_list.append(correct_or_not)

                    # save base learner validation score
                    score_list.append(valid_score_mean)
 
                    # stacking yhat
                    if m == 0:
                        stacked_train = valid_yhat
                        stacked_test = test_yhat
                    elif m > 0:
                        stacked_train = np.concatenate((stacked_train, valid_yhat), axis=1)
                        stacked_test = np.concatenate((stacked_test, test_yhat), axis=1)
                
                # Calculating diversity
                # temp = [filename, q, lo, dis, df, dc, acc, auc]
                temp = [filename]
                div = DiversityMeasure(con_list)
                div_list = div.return_measures()
                temp.extend(div_list)

                # learning meta learner
                test_acc, test_auc = learn_meta_learner(stacked_train, stacked_test, meta_name, train_y, test_y)
                temp.extend([test_acc, test_auc])
                
                # aggregate
                div_result.append(temp)
                score_result.append(score_list)
                
                print(">>> final ", test_acc, test_auc)
                print("\n")
            
            df_div = pd.DataFrame(div_result)
            df_score = pd.DataFrame(score_result)
            df_div.to_csv(file_path + 'diversity/' + str(model_comb) + '_' + str(file), index = None, header = None)
            df_score.to_csv(file_path + 'base performance/' + str(model_comb) + '_' + str(file), index = None, header = None)