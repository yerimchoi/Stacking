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
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score


def randomlist(list):
    index = len(list) - 1
    value = list[randint(0, index)]
    return value
#######################################################################################################################
# kfold stacking Code
#######################################################################################################################

def define_model(model_name):
    global model
    # Define the model
    i = randint(1, 42)
    activation_list = ['relu', 'tanh', 'logistic']
    hidden_list = [(50,), (100,), (50, 50), (100, 100)]
    solver_list = ['lbfgs', 'sgd', 'adam']
    c_list = [0.1, 0.5, 1, 10, 50, 100]
    gamma_list = [0.1, 0.5, 1, 5, 10]
    knn_neigh = [3, 5, 7, 9]
    knn_weight = ['uniform', 'distance']
    max_feature_list = ['auto', 'sqrt', 'log2']
    criterion_list = ['gini', 'entropy']
    min_sample_list = [2, 4, 6, 10, 20]

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
    # Splitting train sets
    train_x, train_y, test_x = pd.DataFrame(train_x), pd.DataFrame(train_y), pd.DataFrame(test_x)
    stk = StratifiedKFold(n_splits = n_fold, shuffle=True)
    valid_fold_pred = np.ones((train_x.shape[0], 1))
    test_pred = np.ones((test_x.shape[0], n_fold))

    index = True
    end = 0
    end_index = 0
    while index == True:
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

        end += 1
        # print("\n")
        if valid_score_mean >= 0.5:
            index = False
        elif end >= 30:
            end_index = 1
            break

    return valid_fold_pred, test_pred_mean, train_score_mean, valid_score_mean, end_index

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
    # print("Stacked test accuracy : ", stacked_test_acc)

    return stacked_test_acc, stacked_test_auc

if __name__ == "__main__":
    global stacked_train, stacked_test, end_index

    model_nlsdk = ['neural_network', 'logistic_regression', 'svc', 'decision_tree', 'knn']
    model_list = []
    for j in range(3, 6):
        models = list(combinations(model_nlsdk, j))
        for length in range(len(models)):
            model_list.append(models[length])

    warnings.filterwarnings('ignore')
    file_path = 'C:/Users/lvaid/skku/Technometrics/Paper/'
    path_dir = 'C:/Users/lvaid/skku/Technometrics/Data/Dataset/222'
    file_list = os.listdir(path_dir)
    file_list.sort()

    for file in file_list:
        csv_file = pd.read_csv(path_dir + '/' + file)
        for model_comb in model_list:
            yhat_csv = []
            score_csv = []
            result = []
            for i in range(1000):
                print(file, '& iteration : ', i)
                while True:
                    meta_name = 'neural_network'
                    x = csv_file.drop(['class'], axis=1)
                    y = csv_file['class']
                    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3)

                    ss = StandardScaler()
                    ss.fit(train_x)
                    train_x = ss.transform(train_x)
                    test_x = ss.transform(test_x)

                    rbf = RandomForestClassifier()
                    rbf.fit(train_x, train_y)
                    rbf_pred = rbf.predict(test_x).reshape(-1, 1)
                    rbf_acc = accuracy_score(rbf_pred, test_y)
                    
                    try:
                        rbf_auc = roc_auc_score(rbf_pred, test_y)
                    except ValueError:
                        rbf_auc = 0

                    ada = AdaBoostClassifier()
                    ada.fit(train_x, train_y)
                    ada_pred = ada.predict(test_x).reshape(-1, 1)
                    ada_acc = accuracy_score(ada_pred, test_y)
                    
                    try:
                        ada_auc = roc_auc_score(ada_pred, test_y)
                    except ValueError:
                        ada_auc = 0
                    print(rbf_acc, ada_auc)

                    for m in range(len(model_comb)):
                        valid_yhat, test_yhat, train_score_mean, valid_score_mean, end_index = \
                            kfold_data(model_comb[m], train_x, train_y, test_x, 5)

                        if end_index == 1:
                            break

                        # save yhat
                        yhat = valid_yhat.reshape(valid_yhat.shape[0], )
                        correct_or_not = np.where(yhat == train_y, 1, 0)
                        yhat_list = [model_comb[m]]
                        yhat_list.extend(correct_or_not)
                        yhat_csv.append(yhat_list)
                        yhat_csv_df = pd.DataFrame(yhat_csv)
                        yhat_csv_df.to_csv(file_path + 'base/yhat/' + str(model_comb) + '_' + str(file),
                                           header=None, index=None)

                        # save score
                        score_list = [model_comb[m]]
                        score_list.extend([train_score_mean, valid_score_mean])
                        score_csv.append(score_list)
                        score_csv_df = pd.DataFrame(score_csv)
                        score_csv_df.to_csv(file_path + 'base/performance/' + str(model_comb) + '_' + str(file),
                                            header=None, index=None)

                        # stacking yhat
                        if m == 0:
                            stacked_train = valid_yhat
                            stacked_test = test_yhat
                        elif m > 0:
                            stacked_train = np.concatenate((stacked_train, valid_yhat), axis=1)
                            stacked_test = np.concatenate((stacked_test, test_yhat), axis=1)

                    if end_index == 1:
                        continue

                    test_acc, test_auc = learn_meta_learner(stacked_train, stacked_test, meta_name, train_y, test_y)
                    empt_list = []
                    empt_list.extend([file, rbf_acc, rbf_auc, ada_acc, ada_auc, test_acc, test_auc])
                    result.append(empt_list)
                    temp = pd.DataFrame(result)
                    temp.columns = ['file', 'rbf_acc', 'rbf_auc', 'ada_acc', 'ada_auc', 'test_acc', 'test_auc']
                    temp.to_csv(file_path + 'meta/' + str(model_comb) + '_' + str(file), index=None)
                    print(">>> final ", test_acc, test_auc)
                    print("\n")
                    break
        print("\n")