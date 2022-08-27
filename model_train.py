import pickle

from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, r2_score
from xgboost import XGBClassifier, plot_importance
import pandas as pd
import matplotlib.pyplot as plt

def loaddata():
    train_data = pd.read_csv('lob.csv', low_memory=False)
    train_target = train_data['trade price']
    train_data = train_data.drop(['trade price'], axis=1)
    train_data = train_data.drop(['asks'], axis=1)
    train_data = train_data.drop(['bids'], axis=1)
    train_data['flag'] = train_data['flag'].astype(bool)
    #print(train_data.dtypes)

    X_train, X_test, y_train, y_test = train_test_split(train_data, train_target, test_size=0.2, random_state=123)
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=123)

    return X_train, X_test, y_train, y_test, X_valid, y_valid


def loaddata2(X_test,y_test):
    test_data = X_test.drop(['trade price'], axis=1)

    #print(train_data.dtypes)



def xgboost_parameters():
    """模型调参过程"""
    # 第一步：确定迭代次数 n_estimators
    # 参数的最佳取值：{'n_estimators': 50}
    # 最佳模型得分:0.9180952380952381
    params = {'n_estimators': [5, 10, 50, 75, 100, 200]}

    # 第二步：min_child_weight[default=1],range: [0,∞] 和 max_depth[default=6],range: [0,∞]
    # min_child_weight:如果树分区步骤导致叶节点的实例权重之和小于min_child_weight,那么构建过程将放弃进一步的分区,最小子权重越大,算法就越保守
    # max_depth:树的最大深度,增加该值将使模型更复杂,更可能过度拟合,0表示深度没有限制
    # 参数的最佳取值：{'max_depth': 2, 'min_child_weight': 1}
    # 最佳模型得分:0.9180952380952381，模型分数未提高
    # params = {'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'min_child_weight': [1, 2, 3, 4, 5, 6]}

    # 第三步:gamma[default=0, alias: min_split_loss],range: [0,∞]
    # gamma:在树的叶子节点上进行进一步分区所需的最小损失下降,gamma越大,算法就越保守
    # 参数的最佳取值：{'gamma': 0.1}
    # 最佳模型得分:0.9247619047619049
    # params = {'gamma': [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}

    # 第四步：subsample[default=1],range: (0,1] 和 colsample_bytree[default=1],range: (0,1]
    # subsample:训练实例的子样本比率。将其设置为0.5意味着XGBoost将在种植树木之前随机抽样一半的训练数据。这将防止过度安装。每一次提升迭代中都会进行一次子采样。
    # colsample_bytree:用于列的子采样的参数,用来控制每颗树随机采样的列数的占比。有利于满足多样性要求,避免过拟合
    # 参数的最佳取值：{'colsample_bytree': 1, 'subsample': 1}
    # 最佳模型得分:0.9247619047619049, 无提高即默认值
    # params = {'subsample': [0.6, 0.7, 0.8, 0.9, 1], 'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1]}

    # 第五步：alpha[default=0, alias: reg_alpha], 和 lambda[default=1, alias: reg_lambda]
    # alpha:L1关于权重的正则化项。增加该值将使模型更加保守
    # lambda:关于权重的L2正则化项。增加该值将使模型更加保守
    # 参数的最佳取值：{'reg_alpha': 0.01, 'reg_lambda': 3}
    # 最佳模型得分:0.9380952380952381
    # params = {'alpha': [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 1, 2, 3], 'lambda': [0.05, 0.1, 1, 2, 3, 4]}

    # 第六步：learning_rate[default=0.3, alias: eta],range: [0,1]
    # learning_rate:一般这时候要调小学习率来测试,学习率越小训练速度越慢,模型可靠性越高,但并非越小越好
    # 参数的最佳取值：{'learning_rate': 0.3}
    # 最佳模型得分:0.9380952380952381, 无提高即默认值
    #params = {'learning_rate': [0.01, 0.05, 0.07, 0.1, 0.2, 0.25, 0.3, 0.4]}

    # 其他参数设置，每次调参将确定的参数加入
    fine_params = {}
    return params, fine_params

def model_adjust_parameters(X_valid, y_valid):

    #model = xgb.XGBRegressor(**other_params)
    # The parameter adjustment tool provided by sklearn, the training set k-fold cross-validation
    param_grid = {
        'max_depth': [2, 3, 4, 5, 6, 7, 8],
        'n_estimators': [30, 50, 100, 300, 500, 1000, 2000],
        'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.01, 0.02, 0.03, 0.05, 0.5],
        "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
        "reg_alpha": [0.0001, 0.001, 0.01, 0.1, 1, 100],
        "reg_lambda": [0.0001, 0.001, 0.01, 0.1, 1, 100],
        "min_child_weight": [2, 3, 4, 5, 6, 7, 8],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
        "subsample": [0.6, 0.7, 0.8, 0.9]}
    model = xgb.XGBRegressor()
    gsearch1 = RandomizedSearchCV(model, param_grid)
    gsearch1.fit(X_valid, y_valid)
    #print('params_scores_', gsearch1.)
    print("best_score_:", gsearch1.best_params_, gsearch1.best_score_)
    #best_score_: {'subsample': 0.9, 'reg_lambda': 100, 'reg_alpha': 1, 'n_estimators': 2000, 'min_child_weight': 8, 'max_depth': 2, 'learning_rate': 0.05, 'gamma': 0.3, 'colsample_bytree': 0.6} 0.9953066363814754

def model_train(filename, X_train, y_train):
    model = xgb.XGBRegressor(
        subsample=0.9,
        reg_lambda=100,
        reg_alpha=1,
        n_estimators=2000,
        min_child_weight=8,
        max_depth=2,
        learning_rate=0.05,
        gamma=0.3,
        colsample_bytree=0.6)

    model.fit(X_train, y_train)

    # Save the model to disk

    pickle.dump(model, open(filename, 'wb'))







if __name__ == "__main__":
    # xgboost参数组合
    #adj_params, fixed_params = xgboost_parameters()
    # 模型调参
    X_train, X_test, y_train, y_test, X_valid, y_valid = loaddata()
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    print(X_valid.shape)
    print(y_valid.shape)
    # Tuning parameters
    #model_adjust_parameters(X_valid, y_valid)
    # Train the model
    filename = 'finalized_model.sav'
    #model_train(filename, X_train, y_train)

    # Load the model
    loaded_model = pickle.load(open(filename, 'rb'))
    # Predict with the best parameters
    y_test_pred = loaded_model.predict(X_test)
    y_train_pred = loaded_model.predict(X_train)
    y_val_pred = loaded_model.predict(X_valid)
    print('The r2 score of the Xgboost Regression on test dataset is ', r2_score(y_test, y_test_pred))
    print('The r2 score of the Xgboost Regression on train dataset is ', r2_score(y_train, y_train_pred))
    print('The r2 score of the Xgboost Regression on train dataset is ', r2_score(y_valid, y_val_pred))
    print("---")
    print(mean_squared_error(y_test, y_test_pred))
    print(mean_squared_error(y_train, y_train_pred))
    print(mean_squared_error(y_valid, y_val_pred))

    fname = 'pre.csv'
    # blotter_data_file = open(fname, 'w')
    #print(y_test_pred)
    lob_data_file = open(fname, 'w')
    fname2 = 'pre2.csv'
    # blotter_data_file = open(fname, 'w')
    # print(y_test_pred)
    lob_data_file2 = open(fname2, 'w')
    for i1, i2 in zip(y_test_pred, y_test):
        #print(i1)
        lob_data_file.write('%s\n' % i1)
        lob_data_file2.write('%s\n' % i2)



    plot_importance(loaded_model)


    plt.figure(figsize=(4, 3))
    plt.scatter(y_train, y_train_pred)
    plt.plot([60, 120], [60,120], '--k')
    plt.axis('tight')
    plt.xlabel('True price')
    plt.ylabel('Predicted price')
    plt.xlim(60, 120)
    plt.ylim(60, 120)
    plt.show()






















