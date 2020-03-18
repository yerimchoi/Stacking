import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
from itertools import combinations

def make_performace_list(model_list, file_list):
    total_acc_list, total_auc_list = [], []

    for model in model_list:
        acc_per_model, auc_per_model = [], []

        for file in file_list:
            meta_csv = np.array(pd.read_csv(file_path + 'meta/' + str(model) + '_' + str(file), header=0))
            mean_acc_per_file = np.mean(meta_csv[:, 5], axis=0)
            mean_auc_per_file = np.mean(meta_csv[:, 6], axis=0)
            acc_per_model.append(mean_acc_per_file)
            auc_per_model.append(mean_auc_per_file)

        mean_acc = np.mean(acc_per_model)
        mean_auc = np.mean(auc_per_model)
        total_acc_list.append(mean_acc)
        total_auc_list.append(mean_auc)

    return total_acc_list, total_auc_list

def performace_bar_graph(total_acc_list, total_auc_list):
    xticks = []
    for i in range(len(total_acc_list)):
        xticks.append(i)

    color_list = []
    for i in range(10):
        color_list.append('salmon')
    for i in range(5):
        color_list.append('gold')
    color_list.append('lightseagreen')

    plt.figure(figsize=(30, 10))
    total_fontsize = 20
    label_fontsize = 40
    plt.rc('font', size = total_fontsize)
    plt.rc('axes', labelsize = label_fontsize)

    # ACC
    rc('font', family='serif')
    rc('font', serif='Times New Roman')
    ax = plt.subplot(121)
    plt.bar(range(len(total_acc_list)), total_acc_list, color=color_list)
    plt.title('Accuracy Comparison', fontsize=20)
    plt.ylabel('Average Accuracy', fontsize=18)
    plt.xlabel('Model', fontsize=18)
    print("ACC min & max : ", np.min(total_acc_list), np.max(total_acc_list))
    plt.ylim(np.min(total_acc_list) - 0.0025, np.max(total_acc_list) + 0.0010)
    ax.set_xticks(xticks)
    ax.set_xticklabels(['NLS', 'NLD', 'NLK', 'NSD', 'NSK', 'NDK', 'LSD', 'LSK', 'LDK', 'SDK',
                        'NLSD', 'NLSK', 'NLDK', 'NSDK', 'LSDK',
                        'NLSDK'], rotation=40)


    # AUC
    rc('font', family='serif')
    rc('font', serif='Times New Roman')
    ax = plt.subplot(122)
    plt.bar(range(len(total_auc_list)), total_auc_list, color=color_list)
    plt.title('AUC Comparison', fontsize=20)
    plt.ylabel('Average AUC', fontsize=18)
    plt.xlabel('Model', fontsize=18)
    print("AUC min & max : ", np.min(total_auc_list), np.max(total_auc_list))
    plt.ylim(np.min(total_auc_list) - 0.0025, np.max(total_auc_list) + 0.0025)
    ax.set_xticks(xticks)
    ax.set_xticklabels(['NLS', 'NLD', 'NLK', 'NSD', 'NSK', 'NDK', 'LSD', 'LSK', 'LDK', 'SDK',
                        'NLSD', 'NLSK', 'NLDK', 'NSDK', 'LSDK',
                        'NLSDK'], rotation=40)
    plt.show()

def save_correlation(model_list, file_list, file_path):
    for model in model_list:
        cor_acc_per_model, cor_auc_per_model = [], []

        for file in file_list:
            result_csv = np.array(pd.read_csv(file_path + 'result/' + str(model) + '_' + str(file), header=0))
            q_acc = np.nan_to_num(np.corrcoef(result_csv[:, 0], result_csv[:, 5])[0][1])
            lo_acc = np.nan_to_num(np.corrcoef(result_csv[:, 1], result_csv[:, 5])[0][1])
            dis_acc = np.corrcoef(result_csv[:, 2], result_csv[:, 5])[0][1]
            df_acc = np.corrcoef(result_csv[:, 3], result_csv[:, 5])[0][1]
            dc_acc = np.corrcoef(result_csv[:, 4], result_csv[:, 5])[0][1]
            acc_list = np.array([file, q_acc, lo_acc, dis_acc, df_acc, dc_acc])
            cor_acc_per_model.append(acc_list)

            q_auc = np.nan_to_num(np.corrcoef(result_csv[:, 0], result_csv[:, 6])[0][1])
            lo_auc = np.nan_to_num(np.corrcoef(result_csv[:, 1], result_csv[:, 6])[0][1])
            dis_auc = np.corrcoef(result_csv[:, 2], result_csv[:, 6])[0][1]
            df_auc = np.corrcoef(result_csv[:, 3], result_csv[:, 6])[0][1]
            dc_auc = np.corrcoef(result_csv[:, 4], result_csv[:, 6])[0][1]
            auc_list = np.array([file, q_auc, lo_auc, dis_auc, df_auc, dc_auc])
            cor_auc_per_model.append(auc_list)

        dt_cor_acc = pd.DataFrame(np.array(cor_acc_per_model))
        dt_cor_auc = pd.DataFrame(np.array(cor_auc_per_model))
        dt_cor_acc.to_csv(file_path + 'correlation/acc/' + str(model) + '.csv', header = None, index = None)
        dt_cor_auc.to_csv(file_path + 'correlation/auc/' + str(model) + '.csv', header = None, index = None)

def save_mmm(model_list, file_path):
    # save min, max, mean value of correlation
    m = 0
    mean_acc, mean_auc, median_acc, median_auc = [], [], [], []
    dt_min_acc = pd.DataFrame(columns=("Q", "lo", "dis", "df", "dc"))
    dt_max_acc = pd.DataFrame(columns=("Q", "lo", "dis", "df", "dc"))
    dt_min_auc = pd.DataFrame(columns=("Q", "lo", "dis", "df", "dc"))
    dt_max_auc = pd.DataFrame(columns=("Q", "lo", "dis", "df", "dc"))

    for model in model_list:
        cor_acc_csv = np.array(pd.read_csv(file_path + 'correlation/acc/' + str(model) + '.csv', header = None))[:, 1:]
        cor_auc_csv = np.array(pd.read_csv(file_path + 'correlation/auc/' + str(model) + '.csv', header = None))[:, 1:]

        # mean & median value
        mean_cor_acc = np.mean(cor_acc_csv, axis = 0)
        mean_cor_auc = np.mean(cor_auc_csv, axis = 0)
        mean_acc.append(mean_cor_acc)
        mean_auc.append(mean_cor_auc)

        median_cor_acc = np.median(cor_acc_csv, axis = 0)
        median_cor_auc = np.median(cor_auc_csv, axis = 0)
        median_acc.append(median_cor_acc)
        median_auc.append(median_cor_auc)

        # min & max value
        min_acc, max_acc, min_auc, max_auc = [], [], [], []

        for i in range(5):
            column_acc = np.ravel(cor_acc_csv[:, i])
            column_auc = np.ravel(cor_auc_csv[:, i])
            acc_index = np.where(column_acc == 0)
            auc_index = np.where(column_auc == 0)
            column_acc = np.delete(column_acc, acc_index)
            column_auc = np.delete(column_auc, auc_index)

            min_acc.append(np.min(column_acc))
            max_acc.append(np.max(column_acc))
            min_auc.append(np.min(column_auc))
            max_auc.append(np.max(column_auc))

        dt_min_acc.loc[m], dt_max_acc.loc[m], dt_min_auc.loc[m], dt_max_auc.loc[m] \
            = min_acc, max_acc, min_auc, max_auc

        m += 1

    dt_mean_acc = pd.DataFrame(np.array(mean_acc))
    dt_mean_auc = pd.DataFrame(np.array(mean_auc))
    dt_median_acc = pd.DataFrame(np.array(median_acc))
    dt_median_auc = pd.DataFrame(np.array(median_auc))

    # save the results
    # dt_mean_acc.to_csv(file_path + 'correlation/mean/mean accuracy.csv', header = None, index = None)
    # dt_mean_auc.to_csv(file_path + 'correlation/mean/mean AUC.csv', header = None, index = None)
    dt_median_acc.to_csv(file_path + 'correlation/mean/median accuracy.csv', header = None, index = None)
    dt_median_auc.to_csv(file_path + 'correlation/mean/median AUC.csv', header = None, index = None)

    # dt_min_acc.to_csv(file_path + 'correlation/mean/min acc.csv', header = None, index = None)
    # dt_max_acc.to_csv(file_path + 'correlation/mean/max acc.csv', header = None, index = None)
    # dt_min_auc.to_csv(file_path + 'correlation/mean/min auc.csv', header = None, index = None)
    # dt_max_auc.to_csv(file_path + 'correlation/mean/max auc.csv', header = None, index = None)

def save_correlation_boxplot(model_list, file_path):
    global dt_acc, dt_auc
    m = 0
    for model in model_list:
        model_name = [model for n in range(23)]
        cor_acc_csv = np.array(pd.read_csv(file_path + 'correlation/acc/' + str(model) + '.csv', header=None))
        cor_auc_csv = np.array(pd.read_csv(file_path + 'correlation/auc/' + str(model) + '.csv', header=None))

        if m == 0:
            dt_acc = pd.DataFrame({"model" : model_name, "Q": cor_acc_csv[:, 1], "lo": cor_acc_csv[:, 2],
                                   "dis": cor_acc_csv[:, 3], "df": cor_acc_csv[:, 4], "dc": cor_acc_csv[:, 5]})
            dt_auc = pd.DataFrame({"model": model_name, "Q": cor_auc_csv[:, 1], "lo": cor_auc_csv[:, 2],
                                   "dis": cor_auc_csv[:, 3], "df": cor_auc_csv[:, 4], "dc": cor_auc_csv[:, 5]})
        else:
            temp_acc = pd.DataFrame({"model" : model_name, "Q": cor_acc_csv[:, 1], "lo": cor_acc_csv[:, 2],
                                   "dis": cor_acc_csv[:, 3], "df": cor_acc_csv[:, 4], "dc": cor_acc_csv[:, 5]})
            temp_auc = pd.DataFrame({"model": model_name, "Q": cor_auc_csv[:, 1], "lo": cor_auc_csv[:, 2],
                                   "dis": cor_auc_csv[:, 3], "df": cor_auc_csv[:, 4], "dc": cor_auc_csv[:, 5]})
            dt_acc = pd.concat([dt_acc, temp_acc])
            dt_auc = pd.concat([dt_auc, temp_auc])

        m += 1

    dt_acc.to_csv(file_path + 'correlation/mean/acc for boxplot.csv', index = None)
    dt_auc.to_csv(file_path + 'correlation/mean/auc for boxplot.csv', index = None)

def plot_boxplot(file_path):
    acc_csv = pd.read_csv(file_path + 'correlation/mean/acc for boxplot.csv')
    auc_csv = pd.read_csv(file_path + 'correlation/mean/auc for boxplot.csv')

    plt.figure(figsize=(20, 20))
    total_fontsize = 20
    label_fontsize = 30
    plt.rc('font', size = total_fontsize)
    plt.rc('axes', labelsize = label_fontsize)

    plt.subplot(321)
    ax = sns.boxplot(x = "model", y = "Q", data = auc_csv)
    ax.set_xticklabels(['NLS', 'NLD', 'NLK', 'NSD', 'NSK', 'NDK', 'LSD', 'LSK', 'LDK', 'SDK',
                        'NLSD', 'NLSK', 'NLDK', 'NSDK', 'LSDK',
                        'NLSDK'], rotation=45)
    ax.set_ylim(-0.7, 0.7)
    ax.set_xlabel('')
    ax.grid(color = 'gray', dashes= (3, 3))

    plt.subplot(322)
    ax = sns.boxplot(x = "model", y = "lo", data = auc_csv)
    ax.set_xticklabels(['NLS', 'NLD', 'NLK', 'NSD', 'NSK', 'NDK', 'LSD', 'LSK', 'LDK', 'SDK',
                        'NLSD', 'NLSK', 'NLDK', 'NSDK', 'LSDK',
                        'NLSDK'], rotation=45)
    ax.set_ylim(-0.7, 0.7)
    ax.set_xlabel('')
    ax.grid(color = 'gray', dashes= (3, 3))

    plt.subplot(323)
    ax = sns.boxplot(x = "model", y = "dis", data = auc_csv)
    ax.set_xticklabels(['NLS', 'NLD', 'NLK', 'NSD', 'NSK', 'NDK', 'LSD', 'LSK', 'LDK', 'SDK',
                        'NLSD', 'NLSK', 'NLDK', 'NSDK', 'LSDK',
                        'NLSDK'], rotation=45)
    ax.set_ylim(-0.7, 0.7)
    ax.set_xlabel('')
    ax.grid(color = 'gray', dashes= (3, 3))

    plt.subplot(324)
    ax = sns.boxplot(x = "model", y = "df", data = auc_csv)
    ax.set_xticklabels(['NLS', 'NLD', 'NLK', 'NSD', 'NSK', 'NDK', 'LSD', 'LSK', 'LDK', 'SDK',
                        'NLSD', 'NLSK', 'NLDK', 'NSDK', 'LSDK',
                        'NLSDK'], rotation=45)
    ax.set_ylim(-0.7, 0.7)
    ax.set_xlabel('')
    ax.grid(color = 'gray', dashes= (3, 3))

    plt.subplot(325)
    ax = sns.boxplot(x = "model", y = "dc", data = auc_csv)
    ax.set_xticklabels(['NLS', 'NLD', 'NLK', 'NSD', 'NSK', 'NDK', 'LSD', 'LSK', 'LDK', 'SDK',
                        'NLSD', 'NLSK', 'NLDK', 'NSDK', 'LSDK',
                        'NLSDK'], rotation=45)
    ax.set_ylim(-0.7, 0.7)
    ax.set_xlabel('')
    ax.grid(color = 'gray', dashes= (3, 3))

    plt.show()


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    file_path = 'C:/Users/lvaid/skku/Technometrics/Paper/'
    path_dir = 'C:/Users/lvaid/skku/Technometrics/Data/Dataset/well'
    file_list = os.listdir(path_dir)
    file_list.sort()

    model_nlsdk = ['neural_network', 'logistic_regression', 'svc', 'decision_tree', 'knn']
    model_list = []
    for j in range(3, 6):
        models = list(combinations(model_nlsdk, j))
        for length in range(len(models)):
            model_list.append(models[length])

#    total_acc_list, total_auc_list = make_performace_list(model_list, file_list)
#    performace_bar_graph(total_acc_list, total_auc_list)
#    # save_correlation(model_list, file_list, file_path)
#    # save_mmm(model_list, file_path)
#
#    # save_correlation_boxplot(model_list, file_path)
    plot_boxplot(file_path)





