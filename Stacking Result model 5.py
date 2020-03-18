import numpy as np
import pandas as pd
import os
import warnings
from itertools import combinations

class DiversityMeasure:
    def __init__(self, yhat1, yhat2, yhat3, yhat4, yhat5, acc, auc):
        self.yhat1 = yhat1
        self.yhat2 = yhat2
        self.yhat3 = yhat3
        self.yhat4 = yhat4
        self.yhat5 = yhat5
        self.acc = acc
        self.auc = auc

    def transform_yhat(self):
        list = [self.yhat1, self.yhat2, self.yhat3, self.yhat4, self.yhat5]
        return list

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

        try:
            q = (n11*n00 - n01*n10) / (n11*n00 + n01*n10)
        except ZeroDivisionError:
            q = 0
        try:
            lo = (n11*n00 - n01*n10) / np.sqrt((n11+n10)*(n01+n00)*(n11+n01)*(n10+n00))
        except ZeroDivisionError:
            lo = 0

        dc = ((n11) / (n11 + n10 + n01 + n00))
        dis = ((n01 + n10) / (n11 + n10 + n01 + n00))
        df = (n00 / (n11 + n10 + n01 + n00))

        return q, lo, dis, df, dc

    def return_measures(self):
        self.yhat_list = self.transform_yhat()
        length = len(self.yhat_list) - 1
        q_list = []
        lo_list = []
        dis_list = []
        df_list = []
        dc_list = []
        for i in range(length):
            index = length - i
            for j in range(index):
                q, lo, dis, df, dc = self.pairmatrix(self.yhat_list[i], self.yhat_list[i + j + 1])
                q_list.append(q)
                lo_list.append(lo)
                dis_list.append(dis)
                df_list.append(df)
                dc_list.append(dc)
        mean_q = (2 * np.sum(q_list)) / (len(q_list) * (len(q_list) - 1))
        mean_lo = (2 * np.sum(lo_list)) / (len(q_list) * (len(q_list) - 1))
        mean_dis = (2 * np.sum(dis_list)) / (len(q_list) * (len(q_list) - 1))
        mean_df = (2 * np.sum(df_list)) / (len(q_list) * (len(q_list) - 1))
        mean_dc = (2 * np.sum(dc_list)) / (len(q_list) * (len(q_list) - 1))

        measure_list = [mean_q, mean_lo, mean_dis, mean_df, mean_dc]
        return measure_list

    def gather_measure_acc(self):
        measure_list = self.return_measures()
        measure_list.extend([self.acc, self.auc])  # [mean_q, mean_lo, mean_dis, mean_df, mean_dc, acc, auc]
        return measure_list

def print_cor(array):
    array = np.nan_to_num(array)
    # acc correlation
    q_acc = np.corrcoef(array[:, 0], array[:, 5])[0][1]
    lo_acc = np.corrcoef(array[:, 1], array[:, 5])[0][1]
    dis_acc = np.corrcoef(array[:, 2], array[:, 5])[0][1]
    df_acc = np.corrcoef(array[:, 3], array[:, 5])[0][1]
    dc_acc = np.corrcoef(array[:, 4], array[:, 5])[0][1]
    result_acc = np.array([q_acc, lo_acc, dis_acc, df_acc, dc_acc])
    result_acc = np.nan_to_num(result_acc)

    # auc correlation
    q_auc = np.corrcoef(array[:, 0], array[:, 6])[0][1]
    lo_auc = np.corrcoef(array[:, 1], array[:, 6])[0][1]
    dis_auc = np.corrcoef(array[:, 2], array[:, 6])[0][1]
    df_auc = np.corrcoef(array[:, 3], array[:, 6])[0][1]
    dc_auc = np.corrcoef(array[:, 4], array[:, 6])[0][1]
    result_auc = np.array([q_auc, lo_auc, dis_auc, df_auc, dc_auc])
    result_auc = np.nan_to_num(result_auc)

    return result_acc, result_auc

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    file_path = 'C:/Users/lvaid/skku/Technometrics/Paper/'
    path_dir = 'C:/Users/lvaid/skku/Technometrics/Data/Dataset/save/'
    file_list = os.listdir(path_dir)
    file_list.sort()

    i = 0
    model_nlsdk = ['neural_network', 'logistic_regression', 'svc', 'decision_tree', 'knn']
    model_list = []
    models = list(combinations(model_nlsdk, 5))
    for length in range(len(models)):
        model_list.append(models[length])

    for model in model_list:
        print(model)
        acc_mean_cor = []
        auc_mean_cor = []

        for file in file_list:
            yhat_csv = np.array(pd.read_csv(file_path + 'base/yhat/' + str(model) + '_' + str(file), header = None))[:, 1:]
            meta_csv = np.array(pd.read_csv(file_path + 'meta/' + str(model) + '_' + str(file) , header=0))
            result_per_file = []

            for j in range(len(meta_csv)):   # meta performance 한 행에 대하여,
                yhat1 = yhat_csv[5*j, :]
                yhat2 = yhat_csv[5*j+1, :]
                yhat3 = yhat_csv[5*j+2, :]
                yhat4 = yhat_csv[5*j+3, :]
                yhat5 = yhat_csv[5*j+4, :]

                row = meta_csv[j]
                test_acc = row[5]
                test_auc = row[6]

                diversity = DiversityMeasure(yhat1, yhat2, yhat3, yhat4, yhat5, test_acc, test_auc)
                diversity_acc_list = diversity.gather_measure_acc()  # [mean_q, mean_lo, mean_dis, mean_df, mean_dc, mean_dc_m, acc, auc]
                result_per_file.append(diversity_acc_list)

            result_per_file = np.array(result_per_file)
            result_csv = pd.DataFrame(result_per_file)
            result_csv.to_csv(file_path + 'result/' + str(model) + '_' + str(file), header = None, index = None)

            acc_cor, auc_cor = print_cor(result_per_file)
            acc_mean_cor.append(acc_cor)
            auc_mean_cor.append(auc_cor)
            i += 1

        acc_mean_cor = np.array(acc_mean_cor)
        auc_mean_cor = np.array(auc_mean_cor)
        print("Accuracy Mean correlation : ", np.mean(acc_mean_cor, axis = 0))
        print("AUC Mean correlation : ", np.mean(auc_mean_cor, axis = 0))
        print("\n")