#!/usr/bin python
# coding: utf-8
########################### 代码功能：利用GBR特征重要性和末尾淘汰机制进行特征选择####################################
################################################## 导入模块 #########################################################

import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series
import numpy as np
from sklearn import ensemble,preprocessing
from sklearn.preprocessing import minmax_scale,StandardScaler, Normalizer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split,LeaveOneOut
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from sklearn.feature_selection import SelectFromModel
import seaborn as sb
import os
import sys
import time
Run_files_num = 10
############################################### 输入要导入数据文件的名称 ###############################################
path_name       = os.getcwd()
file_name       = 'RhCo--data-preprocessing.py'
read_excel_name = 'Features_table_data_cleaning.xlsx'                   # 导入的excel名称
columns_used_in_excel = [i for i in range(0,14)]                # excel表中有数据的列
print(columns_used_in_excel)
y_column_in_excel = 'lgk1'                                            # excel表中目标值那一列的title
cluster_name_column = ['Cluster','lgk1']                                        # excel表中团簇名称那一列的title
#######################################################################################################################

########################################### 定义新建文件夹函数 ########################################################
def mkdir(path):
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' 目录已存在')
        return False
# 定义要创建的目录
mkpath = "{}/data_preprocessing/{}_data_preprocessing/".format(path_name, time.strftime("%Y-%m-%d-%H-%M-%S"))
time_str = time.strftime("%Y-%m-%d-%H-%M-%S")
mkdir(mkpath)
#######################################################################################################################


fp=open( '{}/data_preprocessing/{}_data_preprocessing/data_preprocessing_information.txt'.format(path_name, time_str), 'a+')


############################################ Define copy function ######################################################
# srcfile 需要复制、移动的文件   
# dstpath 目的地址
import glob
import shutil

def mycopyfile(srcfile,dstpath):                       # 复制函数
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(srcfile)             # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)                       # 创建路径
        shutil.copy(srcfile, dstpath + fname)          # 复制文件
        print ("copy %s -> %s"%(srcfile, dstpath + fname))

src_dir_info = '{}/'.format(path_name)

dst_dir_info = '{}/data_preprocessing/{}_data_preprocessing/'.format(path_name, time_str)

src_file_list =  glob.glob(src_dir_info + '{}'.format(read_excel_name))

src_file_list2 =  glob.glob(src_dir_info + '{}'.format(file_name))

for srcfile in src_file_list:
    mycopyfile(srcfile, dst_dir_info)
    
for srcfile in src_file_list2:
    mycopyfile(srcfile, dst_dir_info)
#######################################################################################################################
  

    
################################################## 导入数据 #########################################################

df                  = pd.read_excel('{}'.format(read_excel_name),usecols=columns_used_in_excel)
Table_header_name   = df.columns.to_list()
print(Table_header_name)
print('Read excel name : {}\n'.format(read_excel_name),file=fp)
y                   = df['{}'.format(y_column_in_excel)]
X                   = df.drop(cluster_name_column, axis=1)
X                   = pd.DataFrame(X)
save_path           = '{}/data_preprocessing/{}_data_preprocessing'.format(path_name, time_str)
Non_features_column_1 = df[['Cluster']]
Non_features_column_1_DataFrame = pd.DataFrame(Non_features_column_1)

Non_features_column_2 = df[['lgk1']]
Non_features_column_2_DataFrame = pd.DataFrame(Non_features_column_2)
######################################################################################################################
  
############################################## 原始数据标准化 #####################################################

#******************************************************************************************************************

print('#################################################### StandardScaler ##################################################',file=fp)

ss = StandardScaler()
ss.fit(X)
X_scaler = ss.transform(X)
X_scaler_output=pd.DataFrame(X_scaler)
scaler_data_1 = pd.concat([Non_features_column_1_DataFrame,X_scaler_output],axis=1)
scaler_data = pd.concat([scaler_data_1,Non_features_column_2_DataFrame],axis=1)
scaler_data.to_excel("{}/Features_table_scaler.xlsx".format(save_path), header=Table_header_name, index=False)
#**********************************************************************************************************************

   
################################################### 原始数据归一化 ##########################################################
print('#################################################### minmax_scale ##################################################',file=fp)
#***********************************************************************************************************************************

X_minmax = minmax_scale(X)
X_minmax_output=pd.DataFrame(X_minmax)
minmax_data_1 = pd.concat([Non_features_column_1_DataFrame,X_minmax_output],axis=1)
minmax_data = pd.concat([minmax_data_1,Non_features_column_2_DataFrame],axis=1)
minmax_data.to_excel("{}/Features_table_minmax.xlsx".format(save_path), header=Table_header_name, index=False)

#**********************************************************************************************************************

#######################################################################################################################
    
print('*** Data preprocessing finished ***')
fp.close()

src_file_list =  glob.glob(mkpath + 'Features_table_scaler.xlsx')
for srcfile in src_file_list:
    mycopyfile(srcfile, '{}/'.format(path_name))

for i in range(0, Run_files_num):
    Run_files_path = '{}/Run_{}/'.format(path_name, i+1)
    mkdir(Run_files_path)
    src_file_list_0 = glob.glob('{}/'.format(path_name) + '*.py')
    src_file_list_1 = glob.glob('{}/'.format(path_name) + '*.xlsx')
    for srcfile in src_file_list_0:
        mycopyfile(srcfile, Run_files_path)
    for srcfile in src_file_list_1:
        mycopyfile(srcfile, Run_files_path)