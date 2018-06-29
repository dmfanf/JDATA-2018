# JDATA-2018 用户购买时间预测
my solution for JDATA 2018
rank: A:171/5182  B:30    
## How to run
- extract feature from data<br>
    --extract_fea.py   
- train xgboost   
   -- model_xgb.py   
- ensemble mutiple models using stacking method    
   -- model_stacking.py   
- ensemble the results of xgboost and stacking through weighted average   
   -- model_weighted.py
