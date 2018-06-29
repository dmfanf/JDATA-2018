# JDATA 2018 用户购买时间预测
my solution for JDATA 2018
rank:30
## How to run
- extrac features from the data
  extract_fea.py
- train xgboost to predict S1 and S2
  model_xgb.py
- ensemble models using stacking method
 model_stacking.py
- ensemble the results of xgb and stacking with weighted average
 model weighted.py
