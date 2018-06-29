import pandas as pd 
import numpy as np
import xgboost as xgb
from datetime import datetime, timedelta
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier,MLPRegressor
from my_ensemble import StackingClassifier,StackingRegressor

dataset1 = pd.read_csv('../data/month6.csv')
dataset3 = pd.read_csv('../data/month7.csv') # valid
dataset2 = pd.read_csv('../data/month8.csv') 
testset = pd.read_csv('../data/month9.csv')


dataset12 = pd.concat([dataset1,dataset2],axis=0)
dataset12_X = dataset12.drop(['user_id','label1','label2'],axis=1)
dataset3_X = dataset3.drop(['user_id','label1','label2'],axis=1)
test_X = testset.drop(['user_id','label1','label2'],axis=1)


dataset12_s1 = dataset12.label1
dataset12_s2 = dataset12.label2
dataset3_s1 = dataset3.label1
dataset3_s2 = dataset3.label2

rf_clf = RandomForestClassifier(n_estimators=500,max_features='sqrt',\
                                max_depth=10,min_samples_split=10,\
                                min_samples_leaf=10,n_jobs=3,random_state=5)

ada_clf = AdaBoostClassifier(n_estimators=500,learning_rate=0.8,random_state=0)

gbdt_clf = GradientBoostingClassifier(n_estimators=500,max_depth=3,\
                                min_samples_split=10,min_samples_leaf=10,\
                                max_features='sqrt',random_state=0)

nn_clf = MLPClassifier(alpha=0.01)

base_models = [nn_clf,rf_clf,ada_clf,gbdt_clf]
meta_model = rf_clf

stack_clf = StackingClassifier(base_models=base_models,meta_model=meta_model,n_folds=5)
stack_clf.fit(dataset12_X.values,dataset12_s1.values)
s1_valid = stack_clf.predict_proba(dataset3_X.values)[:,1]
print('valid auc:',metrics.roc_auc_score(dataset3_s1.values,s1_valid))

s1_pred = stack_clf.predict_proba(test_X)[:,1]
s1_pred = pd.concat([testset[['user_id']],pd.DataFrame({'s1_pred':s1_pred})],axis=1)
s1_pred.to_csv('s1_pred_stack.csv',index=False)


#############  S2
rf_reg = RandomForestRegressor(n_estimators=500,max_features='sqrt',\
                                max_depth=10,min_samples_split=10,\
                                min_samples_leaf=10,n_jobs=3,random_state=5)


ada_reg = AdaBoostRegressor(n_estimators=500,learning_rate=0.8,random_state=0)

gbdt_reg = GradientBoostingRegressor(n_estimators=500,max_depth=3,\
                                min_samples_split=10,min_samples_leaf=10,\
                                max_features='sqrt',random_state=0)

nn_reg = MLPRegressor(alpha=0.01)


reg_list = [nn_reg,rf_reg,ada_reg,gbdt_reg]


stack_reg = StackingRegressor(base_models=reg_list,meta_model=rf_reg,n_folds=5)
stack_reg.fit(dataset12_X.values,dataset12_s2.values)
s2_valid = stack_reg.predict(dataset3_X.values)
print('valid rmse:',np.sqrt(metrics.mean_squared_error(dataset3_s2,s2_valid)))


s2_pred = stack_reg.predict(test_X.values)
s2_pred = pd.concat([testset[['user_id']],pd.DataFrame({'pred_date':s2_pred})],axis=1)

######################### submit
out_submit = pd.merge(s1_pred,s2_pred,on='user_id',how='left')

out_submit.to_csv('predict_stack.csv',index=False)

out_submit = out_submit.sort_values(['s1_pred'],ascending=False)
out_submit['pred_date'] = out_submit['pred_date'].map(lambda day: datetime(2017, 9, 1)+timedelta(days=int(day+0.49-1)))

# out_submit.to_csv('predict_stack.csv',index=False)

out_submit = out_submit.drop(['s1_pred'],axis=1)
out_submit.head(50000).to_csv('../submit/predict_stack.csv',index=False,header=True)



