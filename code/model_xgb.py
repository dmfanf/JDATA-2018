
import pandas as pd 
import numpy as np
import xgboost as xgb
from datetime import datetime, timedelta
from sklearn import metrics
from sklearn.decomposition import PCA


dataset1 = pd.read_csv('../data/month6.csv')
dataset3 = pd.read_csv('../data/month7.csv') # valid
dataset2 = pd.read_csv('../data/month8.csv') 
testset = pd.read_csv('../data/month9.csv')

dataset1_X = dataset1.drop(['user_id','label1','label2'],axis=1)

dataset12 = pd.concat([dataset1,dataset2],axis=0)
dataset12_X = dataset12.drop(['user_id','label1','label2'],axis=1)
dataset3_X = dataset3.drop(['user_id','label1','label2'],axis=1)
test_X = testset.drop(['user_id','label1','label2'],axis=1)

# pca = PCA(n_components=100,random_state=0)

# dataset12_X = pca.fit_transform(dataset12_X)
# dataset3_X = pca.transform(dataset3_X)
# test_X = pca.transform(test_X)

dtrain_s1 = xgb.DMatrix(dataset12_X,label = dataset12.label1)
dtrain_s2 = xgb.DMatrix(dataset12_X,label = dataset12.label2)

# dtrain_s1 = xgb.DMatrix(dataset1_X,label = dataset1.label1)
# dtrain_s2 = xgb.DMatrix(dataset1_X,label = dataset1.label2)

dvalid_s1 = xgb.DMatrix(dataset3_X,label = dataset3.label1)
dvalid_s2 = xgb.DMatrix(dataset3_X,label = dataset3.label2)

dtest = xgb.DMatrix(test_X)



################################################################
#### S1
params={'booster':'gbtree',
	    'objective': 'binary:logistic',
	    'eval_metric':'auc',
	    'gamma':0.1,
	    'min_child_weight':1,
	    'max_depth':5,
	    'lambda':10,
	    'subsample':0.7,
	    'colsample_bytree':0.7,
	    'colsample_bylevel':0.7,
	    'eta': 0.01,
	    'tree_method':'exact',
	    'seed':0,
	    'nthread':12
	    }
watchlist=[(dtrain_s1,'train'),(dvalid_s1,'valid')]
print('fit model ...')
model=xgb.train(params,dtrain_s1,num_boost_round=5000,evals=watchlist,early_stopping_rounds=200)
# 预测概率
s1_pred = model.predict(dtest)

model.save_model('s1.model') #保存模型

# 特征重要性
feature_score = model.get_fscore()
feature_score = sorted(feature_score.items(), key=lambda x:x[1],reverse=True)
fs = []
for (key,value) in feature_score:
    fs.append("{0},{1}\n".format(key,value))
    
with open('xgb_feature_score_s1.csv','w') as f:
    f.writelines("feature,score\n")
    f.writelines(fs)



###############################################################
#####回归模型  S2

params={'booster':'gbtree',
	    'objective': 'reg:linear',
	    'eval_metric':'rmse',
	    'gamma':0.1,
	    'min_child_weight':1,
	    'max_depth':5,
	    'lambda':10,
	    'subsample':0.7,
	    'colsample_bytree':0.7,
	    'colsample_bylevel':0.7,
	    'eta': 0.01,
	    'tree_method':'exact',
	    'seed':0,
	    'nthread':12
	    }

watchlist=[(dtrain_s2,'train'),(dvalid_s2,'valid')]
print('fit model ...')
model_s2=xgb.train(params,dtrain_s2,num_boost_round=5000,evals=watchlist,early_stopping_rounds=200)
# 预测
s2_pred = model_s2.predict(dtest)

model_s2.save_model('s2.model') #保存模型

# 特征重要性
feature_score = model_s2.get_fscore()
feature_score = sorted(feature_score.items(), key=lambda x:x[1],reverse=True)
fs = []
for (key,value) in feature_score:
    fs.append("{0},{1}\n".format(key,value))
    
with open('xgb_feature_score_s2.csv','w') as f:
    f.writelines("feature,score\n")
    f.writelines(fs)


####################################################################
## submit
out_submit = pd.concat([testset[['user_id']],pd.DataFrame({'s1_pred':s1_pred}),pd.DataFrame({'pred_date':s2_pred})],axis=1)
out_submit.to_csv('predict_xgb.csv',index=False)

out_submit = out_submit.sort_values(['s1_pred'],ascending=False)
out_submit['pred_date'] = out_submit['pred_date'].map(lambda day: datetime(2017, 9, 1)+timedelta(days=int(day+0.49-1)))



out_submit = out_submit.drop(['s1_pred'],axis=1)
out_submit.head(50000).to_csv('../submit/predict_xgb.csv',index=False,header=True)
