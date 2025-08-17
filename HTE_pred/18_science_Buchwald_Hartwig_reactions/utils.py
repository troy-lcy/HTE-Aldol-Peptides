import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import matplotlib.colors as mcolors

# Import relevant scikit-learn modules
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim


def fit_models(X_train, 
               X_test,
               y_train, 
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
    print('Done fitting models')
    return predictions, r2_values, rmse_values

# 有的模型是提前训练好的直接加载来测试，有的是没训练过现在要训练
def load_or_fit_models(X_train, 
               X_test,
               y_train, 
               y_test,
               models_need_fit=[], models_already_fitted=[]):
    predictions = []
    r2_values = []
    rmse_values = []
#     models = []#所有模型，包括训练过和现在要训练的
#     for model in models_need_fit:
#         print(model)
#         # fit the model and generate predictions
#         model.fit(X_train, y_train.ravel())
#         models.append(model)
#     
#     models.extend(models_already_fitted)
        
    # models_need_fit中的都是sklearn创建的模型
    #if False:
    for model in models_need_fit:
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

    # models_already_fitted中的暂时都是用pytorch创建的模型
    for model in models_already_fitted:
        print(model)
        # fit the model and generate predictions
        #model.fit(X_train, y_train.ravel())
        features_test = Variable(torch.from_numpy(X_test).to(torch.float))                    
        # Forward propagation
        outputs_test = model(features_test)
        preds = outputs_test.data
        #preds = model.predict(X_test)

        # calculate an R-squared and RMSE values
        r_squared = r2_score(y_test, preds)
        rmse = mean_squared_error(y_test, preds) ** 0.5

        # append all to lists
        predictions.append(preds)
        r2_values.append(r_squared)
        rmse_values.append(rmse)
    print('Done fitting models')
    return predictions, r2_values, rmse_values


def plot_models(predictions,
                r2_values,
                rmse_values,
                y_test,
                titles =[ 'AdaBoost',
                          'Linear Regression',
                          'Support Vector Machine',
                          'k-Nearest Neighbors',
                          'MLPRegressor',
                          'Random Forest',
                        #   'Neural Network'
                        ],
                positions=[231,232,233,234,235,236],
                # colors = [1,2,3,4,5,6],
                save=False):

    fig = plt.figure(figsize=(15,10))
    for pos, pred, r2, rmse, title  in zip(positions, # ,color
                                          predictions,
                                          r2_values,
                                          rmse_values,
                                          titles,
                                          #colors
                                          ):
        # create subplot
        plt.subplot(pos)
        plt.grid(alpha=0.2)
        plt.title(title, fontsize=15)
        # colors=list(mcolors.TABLEAU_COLORS.keys())
        # add score patches
        r2_patch = mpatches.Patch(label="R2 = {:04.2f}".format(r2)) # ,color=mcolors.TABLEAU_COLORS[colors[color]]
        rmse_patch = mpatches.Patch(label="RMSE = {:4.2f}".format(rmse)) # ,color=mcolors.TABLEAU_COLORS[colors[color]]
        # plt.xlim(-40,130)
        # plt.ylim(-10,130)
        plt.scatter(pred, y_test, alpha=0.2,) # color=mcolors.TABLEAU_COLORS[colors[color]]
        plt.legend(handles=[r2_patch, rmse_patch], fontsize=12,loc='upper left')
        plt.plot(np.arange(-1,5), np.arange(-1,5), ls="--", c=".3")
        fig.text(0.5, 0.07, 'predicted deltaG%', ha='center', va='center', fontsize=15)
        fig.text(0.09, 0.5, 'observed deltaG%', ha='center', va='center', rotation='vertical', fontsize=15)
        plt.savefig('compare.png', dpi = 300)
    plt.show()

def plot_models_Sun(predictions,
                r2_values,
                rmse_values,
                y_test,
#                 titles =['Linear Regression',
#                           'k-Nearest Neighbors',
#                           'Support Vector Machine',
#                           'Neural Network [5 neurons]',
#                           'Neural Network [100 neurons]',
#                           'Random Forest'],
                titles =['Multi sub net',
                    'Linear Regression',
                    'k-Nearest Neighbors',
                    'Support Vector Machine',
                    'Neural Network [5 neurons]',
                    'Random Forest'
                    #'Neural Network [100 neurons]',
                    ],                
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
        plt.xlim(-25,105)
        plt.ylim(0,105)
        plt.scatter(pred, y_test, alpha=0.2)
        plt.legend(handles=[r2_patch, rmse_patch], fontsize=12)
        # plt.plot(np.arange(100), np.arange(100), ls="--", c=".3")#ls:linestyle c:line color
        plt.plot(np.arange(4), np.arange(4), ls="--", c=".3")#ls:linestyle c:line color
        fig.text(0.5, 0.08, 'predicted yield', ha='center', va='center', fontsize=15)
        fig.text(0.09, 0.5, 'observed yield', ha='center', va='center', rotation='vertical', fontsize=15)
    if save:
        #plt.savefig(save, dpi = 300)
        if fig_title is not None:            
            plt.savefig(fig_title)
        else:
            plt.savefig('show_precision.jpg')
    else:
        plt.show()
        
class MLPRegressor(nn.Module):    
    def __init__(self, input_dim, hidden_dim=512, num_layers=3, dropout=0.1):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(hidden_dim, 1)  # 产率为标量

    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x

# 训练函数
def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb).squeeze()
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()

# 评估函数
def evaluate_model(model, test_loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            pred = model(xb).squeeze().cpu().numpy()
            preds.extend(pred)
            targets.extend(yb.numpy())
    return np.array(preds), np.array(targets)