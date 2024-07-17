import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

# Import relevant scikit-learn modules
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# 训练模型
def fit_models(X_train,
               y_train,
               X_test,
               y_test,
               models=[]):
    predictions = []
    r2_values = []
    rmse_values = []
    for model in models:
        print(model)
        # fit the model and generate predictions
        model.fit(X_train, y_train.ravel())
        preds = model.predict(X_test)

        # calculate an R-squared and RMSE values
        r_squared = r2_score(y_test, preds)
        rmse = mean_squared_error(y_test, preds) ** 0.5

        # append all to lists
        predictions.append(preds)
        r2_values.append(r_squared)
        rmse_values.append(rmse)
        print('r2 scores:', r_squared,)
        print('rmse score:',rmse)
    print('Done fitting models')
    return predictions, r2_values, rmse_values

def fit_models_train_val(X_train, y_train, X_valid, y_valid, X_test, y_test, models=[]):
    predictions_valid = []
    r2_values_valid = []
    rmse_values_valid = []
    
    predictions_test = []
    r2_values_test = []
    rmse_values_test = []
    
    for model in models:
        print(f'Model: {model}')
        
        # Fit the model and generate predictions for validation set
        model.fit(X_train, y_train.ravel())
        preds_valid = model.predict(X_valid)

        # Calculate R-squared and RMSE for validation set
        r_squared_valid = r2_score(y_valid, preds_valid)
        rmse_valid = mean_squared_error(y_valid, preds_valid) ** 0.5

        # Append results to validation lists
        predictions_valid.append(preds_valid)
        r2_values_valid.append(r_squared_valid)
        rmse_values_valid.append(rmse_valid)
        
        print('Validation set results:')
        print('R2:', r_squared_valid)
        print('RMSE:', rmse_valid)
        
        # Generate predictions for test set
        preds_test = model.predict(X_test)

        # Calculate R-squared and RMSE for test set
        r_squared_test = r2_score(y_test, preds_test)
        rmse_test = mean_squared_error(y_test, preds_test) ** 0.5

        # Append results to test lists
        predictions_test.append(preds_test)
        r2_values_test.append(r_squared_test)
        rmse_values_test.append(rmse_test)
        
        print('Test set results:')
        print('R2:', r_squared_test)
        print('RMSE:', rmse_test)

    print('Done fitting models')
    return (predictions_valid, r2_values_valid, rmse_values_valid,
            predictions_test, r2_values_test, rmse_values_test)

def data2matrix(filepath):

    df = pd.read_csv(filepath) #将cvs数据转化为dataframe格式

#     len(df) #行数
#     len(df.columns) #列数

    # datamat = np.zeros((3960,120))
    #创建一个对应的数据库
    datamat = list = df.values.tolist() #将dataframe格式转化为list
    labels = np.zeros((len(df),1)) #创建一个对应行数空的标签
    # len(datamat[0])
#     print(len(labels))
    num_lines = len(datamat)
    for i in range(num_lines):
        labels[i] = datamat[i][3]
        i+=1
#     len(datamat[0])
    return datamat, labels

def plot_models(predictions,
                r2_values,
                rmse_values,
                y_test,
                titles =['Linear Regression',
                          'k-NearestNeighbor',
                          'Support Vector Machine',
                          'Multi-Layer Perceptron',
                          'Multi-Layer Perceptron',
                          'Random Forest',
                          ],    
                # titles =['Linear Regression',
                #           'Partial least squares',
                #           'k-NearestNeighbor',
                #           'Multi-Layer Perceptron',
                #           'Support Vector Machine',
                #           'Random Forest',
                #           ],               
                positions=[231,232,233,234,235,236],
                save=False,
                fig_title=None):

    fig = plt.figure(figsize=(15,10))
    for pos, pred, r2, rmse, title in zip(positions,
                                          predictions,
                                          r2_values,
                                          rmse_values,
                                          titles):
        # create subplot
        plt.subplot(pos)
        plt.grid(alpha=0.2)
        plt.title(title, fontsize=15)
        
        # add score patches
        r2_patch = mpatches.Patch(label="R2 = {:04.2f}".format(r2))#for legend
        rmse_patch = mpatches.Patch(label="RMSE = {:4.1f}".format(rmse))
        plt.xlim(-10,100)
        plt.ylim(-10,100)
        plt.scatter(pred, y_test, alpha=0.2)
        plt.legend(handles=[r2_patch, rmse_patch], fontsize=12)
        plt.plot(np.arange(110), np.arange(110), ls="--", c=".3")#ls:linestyle c:line color
        fig.text(0.5, 0.08, 'predicted ee', ha='center', va='center', fontsize=15)
        fig.text(0.09, 0.5, 'observed ee', ha='center', va='center', rotation='vertical', fontsize=15)
    if save:
        #plt.savefig(save, dpi = 300)
        if fig_title is not None:            
            plt.savefig(fig_title)
        else:
            plt.savefig('show_precision.jpg')
    else:
        plt.show()
    

# metrics是R2或RMSE的列表
def plot_bar_of_metric(metrics):        
    #from matplotlib.font_manager import FontProperties
    #font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)  
    stick_num = 20
    stick_skip = len(metrics)/stick_num#柱子间隔
    
#     plt.bar(stick_skip*range(1, stick_num), metrics, label='graph 1')
    plt.bar(range(len(metrics)), metrics, label='metric value')
    
    #plt.bar([2, 4, 6, 8, 10], [4, 6, 8, 13, 15], label='graph 2')
    
    # params
    
    # x: 条形图x轴
    # y：条形图的高度
    # width：条形图的宽度 默认是0.8
    # bottom：条形底部的y坐标值 默认是0
    # align：center / edge 条形图是否以x轴坐标为中心点或者是以x轴坐标为边缘
    
    plt.legend()
    
    plt.xlabel('number')
    plt.ylabel('value')
    
    plt.title(u'test all data')#, FontProperties=font)
    
    plt.show()
    
def run_fold_pipline(origin_df, fea_df, label_df, split_idx, models = [], task = ''):

    r2 = []
    rmse = []

    unique_values = origin_df.iloc[:, split_idx].unique()
    sub_ori_index = origin_df.index.tolist()

    for value in unique_values:
        
        test_idx = origin_df.index[origin_df.iloc[:, split_idx] == value].tolist()
        train_idx = [x for x in sub_ori_index if x not in test_idx]

        test_set = fea_df.loc[test_idx]
        train_set = fea_df.loc[train_idx]

        test_yield = label_df.loc[test_idx].values
        train_yield = label_df.loc[train_idx].values

        x = pd.DataFrame(train_set)
        y = np.array(train_yield)

        preds, r2_values, rmse_values = fit_models(x,y,test_set,test_yield,models)
        r2.append(r2_values)
        rmse.append(rmse_values)
        print(f'******************The {value} run******************')

    df_r2 = pd.DataFrame(r2,columns= models)
    df_rmse = pd.DataFrame(rmse,columns=models)
    df = pd.concat([df_r2,df_rmse], axis=1, names=['R2','RMSE'])
    df.to_csv(f'results/HTE_Cyclopep/Split_by_compound/{task}_{origin_df.columns[split_idx]}.csv')
    
    
# data = np.array([[1,2,3,-4,5], [1,2,3,6,7], [2,3,4,5,7], [3,4,5,6,7], [4,5,6,7,8]])
# # idex=np.lexsort([-1*data[:,2], data[:,1], data[:,0]])
# np.random.shuffle(data)
# # print(data)
# idex=np.lexsort([-1*data[:,0]])
# # print(data[idex])
# # print(np.abs(data)/np.abs(data))
# # print((data[:, 1])[1])

# ind = np.array(range(5))
# inde = ind in np.array([1,3])
# print(data[inde])



if False:
    DATA_DIR = 'data/remove_csf_Xan_lessthan10/'
    my_data = np.loadtxt(open(DATA_DIR + 'clean_data_index.csv',"rb"),delimiter=",",skiprows=0)
    #my_data = my_data.astype(int)
    print(my_data)
    b = data[:, 1]#array([2, 2, 3, 4, 5])
    ind = b>2#array([False, False,  True,  True,  True])
    c = b[ind]#array([3, 4, 5])
    print(c)