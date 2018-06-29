# -*- coding: utf-8 -*-
# author：Cookly
from util import DataLoader, Features
import pandas as pd 
import numpy as np 
from datetime import datetime, timedelta
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# test code
Data = DataLoader(
				FILE_jdata_sku_basic_info='../data/jdata_sku_basic_info.csv',
				FILE_jdata_user_action='../data/jdata_user_action.csv',
				FILE_jdata_user_basic_info='../data/jdata_user_basic_info.csv',
				FILE_jdata_user_comment_score='../data/jdata_user_comment_score.csv',
				FILE_jdata_user_order='../data/jdata_user_order.csv'
			)


###  1,3,9
dataset1 = Features(
						DataLoader=Data,
						PredMonthBegin = datetime(2017, 6, 1),
						PredMonthEnd = datetime(2017,6, 30),
					FeatureMonthList = [(datetime(2017, 5, 1), datetime(2017, 5,31), 1),\
								(datetime(2017, 3, 1), datetime(2017, 5, 31), 3),\
								(datetime(2016, 9, 1), datetime(2017, 5, 31), 9)],
					MakeLabel = True
				)
dataset2 = Features(
						DataLoader=Data,
						PredMonthBegin = datetime(2017, 7, 1),
						PredMonthEnd = datetime(2017,7, 31),
						FeatureMonthList = [(datetime(2017, 6, 1), datetime(2017, 6, 30), 1),\
									(datetime(2017, 4, 1), datetime(2017, 6, 30), 3),\
									(datetime(2016, 10, 1), datetime(2017, 6, 30), 9)],
						MakeLabel = True
						)
dataset3 = Features(
						DataLoader=Data,
						PredMonthBegin = datetime(2017, 8, 1),
						PredMonthEnd = datetime(2017,8, 31),
					FeatureMonthList = [(datetime(2017, 7, 1), datetime(2017, 7, 31), 1),\
								(datetime(2017, 5, 1), datetime(2017, 7, 31), 3),\
								(datetime(2016, 11, 1), datetime(2017, 7, 31), 9)],
					MakeLabel = True
				)

# pred data
testset = Features(
					DataLoader=Data,
					PredMonthBegin = datetime(2017, 9, 1),
					PredMonthEnd = datetime(2017, 9, 30),
					FeatureMonthList = [(datetime(2017, 8, 1), datetime(2017, 8, 31), 1),\
									(datetime(2017, 6, 1), datetime(2017, 8, 31), 3),\
									(datetime(2016, 12, 1), datetime(2017, 8, 31), 9)],
					MakeLabel = False
				)
dataset1.data_BuyOrNot_FirstTime = dataset1.data_BuyOrNot_FirstTime.fillna(0)
dataset2.data_BuyOrNot_FirstTime = dataset2.data_BuyOrNot_FirstTime.fillna(0)
dataset3.data_BuyOrNot_FirstTime = dataset3.data_BuyOrNot_FirstTime.fillna(0)
testset.data_BuyOrNot_FirstTime = testset.data_BuyOrNot_FirstTime.fillna(0)


dataset1.data_BuyOrNot_FirstTime.to_csv('../data/month6.csv',index=False)
dataset2.data_BuyOrNot_FirstTime.to_csv('../data/month7.csv',index=False)
dataset3.data_BuyOrNot_FirstTime.to_csv('../data/month8.csv',index=False)
testset.data_BuyOrNot_FirstTime.to_csv('../data/month9.csv',index=False)
print('done!!')