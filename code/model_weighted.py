import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# s1_pred = pd.read_csv('s1_pred_stack.csv')
# s2_pred = pd.read_csv('s2_pred_stack.csv')

pred_xgb = pd.read_csv('predict_xgb.csv')
pred_stack = pd.read_csv('predict_stack.csv')

pred_xgb['s1_pred'] = 0.3 * pred_xgb.s1_pred + 0.7 * pred_stack.s1_pred
pred_xgb['pred_date'] = 0.3 * pred_xgb.pred_date +0.7 *pred_stack.pred_date
out_submit = pred_xgb
out_submit = out_submit.sort_values(['s1_pred'],ascending=False)
out_submit['pred_date'] = out_submit['pred_date'].map(lambda day: datetime(2017, 9, 1)+timedelta(days=int(day+0.49-1)))

# out_submit.to_csv('predict_stack.csv',index=False)

out_submit = out_submit.drop(['s1_pred'],axis=1)
out_submit.head(50000).to_csv('../submit/stack_xgb_weighted.csv',index=False,header=True)
# print(out_submit)