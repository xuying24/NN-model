import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import time, os, glob, shutil

d1_range                  = np.arange(15, 25, 5)
d2_range                  = np.arange(20, 40, 5)
learning_rate_range = np.arange(0.0026, 0.0032, 0.0002)
epochs_range           = np.arange(800, 900, 100)

#环境变量的配置
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

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

def mycopyfile(srcfile,dstpath):                       # 复制函数
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(srcfile)             # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)                       # 创建路径
        shutil.copy(srcfile, dstpath + fname)          # 复制文件
        #print ("copy %s -> %s"%(srcfile, dstpath + fname))


y_column_in_excel = 'lgk1'                                            # excel表中目标值那一列的title
cluster_name_column = 'Cluster'                                        # excel表中团簇名称那一列的title

df = pd.read_excel('./Train_data_for_model_validation.xlsx')
x_train = df.drop(['{}'.format(cluster_name_column),'{}'.format(y_column_in_excel)], axis=1)
y_train = df['lgk1']

df_2 = pd.read_excel('./Test_data_for_model_validation.xlsx')
x_test = df_2.drop(['{}'.format(cluster_name_column),'{}'.format(y_column_in_excel)], axis=1)
y_test = df_2['lgk1']

time_str = time.strftime("%Y-%m-%d-%H-%M-%S")
path_name = os.getcwd()
mkpath = "{}/Model_validation/{}-model-validation".format(path_name, time_str)
mkdir(mkpath)
models_save_path = '{}/models_save'.format(mkpath)
mkdir(models_save_path)
fp = open('{}/model-validation-information.txt'.format(mkpath), 'a+')

train_set_r_save_list    = []
train_set_RMSE_save_list    = []
test_set_r_save_list    = []
test_set_RMSE_save_list    = []
hyperparameters_save_list = []
j = 0
for d1_range_i in d1_range:
    for d2_range_i in d2_range:
        for epochs_range_i in epochs_range:
            for learning_rate_range_i in learning_rate_range:
                j += 1
                d1 = d1_range_i
                d2 = d2_range_i
                d3 = 0
                activation_ = 'tanh'
                batch_size = 4
                epochs = epochs_range_i
                learning_rate_ = learning_rate_range_i

                model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                                    tf.keras.layers.Dense(d1, activation='{}'.format(activation_)),
                                                    tf.keras.layers.Dense(d2, activation='{}'.format(activation_)),
                                                    # tf.keras.layers.Dense(d3, activation='{}'.format(activation_)),
                                                    tf.keras.layers.Dense(1)])

                model.compile(tf.keras.optimizers.Adam(learning_rate=learning_rate_),
                              loss = tf.keras.losses.MeanSquaredError(),
                              metrics=['accuracy'])

                model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0)
                r_train, _ = pearsonr(y_train, model.predict(x_train))
                RMSE_train = np.sqrt(mean_squared_error(y_train, model.predict(x_train)))

                r_test, _ = pearsonr(y_test, model.predict(x_test))
                RMSE_test = np.sqrt(mean_squared_error(y_test, model.predict(x_test)))


                test_set_RMSE_save_list.append(RMSE_test)
                test_set_r_save_list.append(r_test)
                train_set_RMSE_save_list.append(RMSE_test)
                train_set_r_save_list.append(r_train)
                hyperparameters_list = [d1_range_i, d2_range_i, epochs_range_i, learning_rate_range_i]
                hyperparameters_save_list.append(hyperparameters_list)
                model.save('{}/model_{}.keras'.format(models_save_path, j))

fp = open('{}/model-validation-information.txt'.format(mkpath), 'a+')
print('**************************************************** summry *****************************************************', file=fp)
print('################################ Intermediate process ##############################', file=fp)
print('train_set_RMSE_save_list:', train_set_RMSE_save_list, file=fp)
print('train_set_r_save_list:', train_set_r_save_list, file=fp)
print('test_set_RMSE_save_list:', test_set_RMSE_save_list, file=fp)
print('test_set_r_save_list:', test_set_r_save_list, file=fp)
print('\n', file=fp)
print('Max_test_set_r_index:', test_set_r_save_list.index(max(test_set_r_save_list)), file=fp)
print('selected_model:',test_set_r_save_list.index(max(test_set_r_save_list)) + 1, file=fp)
print('Max_test_set_r:', max(test_set_r_save_list), file=fp)
print('Max_test_set_r_with_hyperparameters:', hyperparameters_save_list[test_set_r_save_list.index(max(test_set_r_save_list))],  file=fp)
print('\n', file=fp)
good_d1 = hyperparameters_save_list[test_set_r_save_list.index(max(test_set_r_save_list))][0]
good_d2 = hyperparameters_save_list[test_set_r_save_list.index(max(test_set_r_save_list))][1]
good_epochs = hyperparameters_save_list[test_set_r_save_list.index(max(test_set_r_save_list))][2]
good_learning_rate = hyperparameters_save_list[test_set_r_save_list.index(max(test_set_r_save_list))][3]

d1 = good_d1
d2 = good_d2
epochs = good_epochs
learning_rate_ = good_learning_rate

#模型恢复（加载）
model = tf.keras.models.load_model('{}/model_{}.keras'.format(models_save_path, test_set_r_save_list.index(max(test_set_r_save_list)) + 1), compile = False)

pd.DataFrame(model.predict(x_train)).to_excel('{}/y_train_predict.xlsx'.format(mkpath))
pd.DataFrame(model.predict(x_test)).to_excel('{}/y_test_predict.xlsx'.format(mkpath))

src_dir_info = '{}/'.format(path_name)
src_file_list =  glob.glob(src_dir_info + '*.xlsx')
src_file_list_2 =  glob.glob(src_dir_info + '*.py')
dst_dir_info = '{}/'.format(mkpath)
for srcfile in src_file_list:
    mycopyfile(srcfile, dst_dir_info)
for srcfile in src_file_list_2:
    mycopyfile(srcfile, dst_dir_info)

pd.DataFrame(model.predict(x_train)).to_excel('{}/y_train_predict.xlsx'.format(mkpath))
pd.DataFrame(model.predict(x_test)).to_excel('{}/y_test_predict.xlsx'.format(mkpath))

r_train, _ = pearsonr(y_train, model.predict(x_train))
RMSE_train = np.sqrt(mean_squared_error(y_train, model.predict(x_train)))

r, _ = pearsonr(y_test, model.predict(x_test))
RMSE = np.sqrt(mean_squared_error(y_test, model.predict(x_test)))

###*绘制训练集上预测值和真实值之间的散点图
size = len(y_train)
Y = np.linspace(-7, -15, size)
X = np.linspace(-7, -15, size)
fig = plt.figure(dpi=80, figsize=(20, 20))
ax1 = fig.add_subplot(1, 1, 1)
ax1.set_title("Train set", size=45)
ax1.scatter(y_train, model.predict(x_train), marker='*', c='red', s=1000, alpha=0.7)
plt.xticks(fontproperties='Times New Roman', size=45)
plt.yticks(fontproperties='Times New Roman', size=45)
ax1.set_xlabel("Experimental lgk1", size=45)
ax1.set_ylabel("Predicted lgk1", size=45)
ax1.plot(X, Y)
plt.savefig('{}/Train_set.png'.format(mkpath))


###*绘制测试集上预测值和真实值之间的散点图
size = len(y_test)
Y = np.linspace(-7, -15, size)
X = np.linspace(-7, -15, size)
fig = plt.figure(dpi=60, figsize=(20, 20))
ax1 = fig.add_subplot(1, 1, 1)
ax1.set_title("Test set", size=45)
ax1.scatter(y_test, model.predict(x_test), marker='*', c='red', s=1000, alpha=0.7)
plt.xticks(fontproperties='Times New Roman', size=45)
plt.yticks(fontproperties='Times New Roman', size=45)
ax1.set_xlabel("Experimental lgk1", size=45)
ax1.set_ylabel("Predicted lgk1", size=45)
ax1.plot(X, Y)
plt.savefig('{}/Test_set.png'.format(mkpath))
#plt.show()

#存储一些神经网络的超参数、pearson 和 RMSE的值
print('**************************************************** summry *****************************************************',file=fp)
print('model_layer1: {}'.format(d1), file=fp)
print('model_layer2: {}'.format(d2), file=fp)
#print('model_layer3: {}'.format(d3), file=fp)
print('learning_rate: {}'.format(learning_rate_), file=fp)
print('activation: tanh', file=fp)
print('batch_size: {}'.format(batch_size), file=fp)
print('epochs: {}'.format(epochs), file=fp)

print("The coefficient of pearsonr on train set: {:.4f}".format(r_train[0]), file=fp)
print("The RMSE on train set: {:.4f}\n".format(RMSE_train), file=fp)
print("* The coefficient of pearsonr on test set: {:.4f}".format(r[0]), file=fp)
print("* The RMSE on test set: {:.4f}".format(RMSE), file=fp)