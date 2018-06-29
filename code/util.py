# -*- coding: utf-8 -*-
# author：Cookly
import pandas as pd 
import numpy as np 
from datetime import datetime
import lightgbm as lgb

class DataLoader(object):
	def __init__(self,
				FILE_jdata_sku_basic_info,
				FILE_jdata_user_action,
				FILE_jdata_user_basic_info,
				FILE_jdata_user_comment_score,
				FILE_jdata_user_order
				):
		self.FILE_jdata_sku_basic_info = FILE_jdata_sku_basic_info
		self.FILE_jdata_user_action = FILE_jdata_user_action
		self.FILE_jdata_user_basic_info = FILE_jdata_user_basic_info
		self.FILE_jdata_user_comment_score = FILE_jdata_user_comment_score
		self.FILE_jdata_user_order = FILE_jdata_user_order

		self.df_sku_info = pd.read_csv(self.FILE_jdata_sku_basic_info)
		self.df_user_action = pd.read_csv(self.FILE_jdata_user_action)
		self.df_user_info = pd.read_csv(self.FILE_jdata_user_basic_info)
		self.df_user_comment = pd.read_csv(self.FILE_jdata_user_comment_score)
		self.df_user_order = pd.read_csv(self.FILE_jdata_user_order)

		# change date2datetime
		self.df_user_action['a_date'] = pd.to_datetime(self.df_user_action['a_date'])
		self.df_user_order['o_date'] = pd.to_datetime(self.df_user_order['o_date'])
		self.df_user_comment['comment_create_tm'] = pd.to_datetime(self.df_user_comment['comment_create_tm'])

		#### delete outliers
		self.df_user_action = self.df_user_action[self.df_user_action.a_num<4]
		self.df_user_order = self.df_user_order[self.df_user_order.o_sku_num<20]

		# sort by datetime
		self.df_user_action = self.df_user_action.sort_values(['user_id','a_date'])
		self.df_user_order = self.df_user_order.sort_values(['user_id','o_date'])
		self.df_user_comment = self.df_user_comment.sort_values(['user_id','comment_create_tm'])

		# year month day
		self.df_user_order['year'] = self.df_user_order['o_date'].dt.year
		self.df_user_order['month'] = self.df_user_order['o_date'].dt.month
		self.df_user_order['day'] = self.df_user_order['o_date'].dt.day

		self.df_user_action['year'] = self.df_user_action['a_date'].dt.year
		self.df_user_action['month'] = self.df_user_action['a_date'].dt.month
		self.df_user_action['day'] = self.df_user_action['a_date'].dt.day



class Features(object):
	def __init__(self,
				DataLoader,
				PredMonthBegin,
				PredMonthEnd,
				FeatureMonthList,
				MakeLabel = True
				):
		self.DataLoader = DataLoader
		self.PredMonthBegin = PredMonthBegin
		self.PredMonthEnd = PredMonthEnd
		self.FeatureMonthList = FeatureMonthList
		self.MakeLabel = MakeLabel

		# label columns
		self.LabelColumns = ['label1','label2']
		self.IDColumns = ['user_id']

		# merge feature table
		# Order Comment User Sku
		# print(self.DataLoader.df_user_order.head())
		# print(self.DataLoader.df_user_info.head())
		self.df_Order_Comment_User_Sku = self.DataLoader.df_user_order.\
												merge(self.DataLoader.df_user_comment,on=['user_id','o_id'],how='left').\
												merge(self.DataLoader.df_user_info,on='user_id',how='left').\
												merge(self.DataLoader.df_sku_info,on='sku_id',how='left')
		# Action User Sku
		self.df_Action_User_Sku = self.DataLoader.df_user_action.\
												merge(self.DataLoader.df_user_info,on='user_id',how='left').\
												merge(self.DataLoader.df_sku_info,on='sku_id',how='left')

		# Make Label 
		self.data_BuyOrNot_FirstTim = self.MakeLabel_()

		# MakeFeature_Order_Comment_
		for FeatureMonthBegin, FeatureMonthEnd, month in FeatureMonthList:
			self.MakeFeature_Order_Comment_(
									FeatureMonthBegin = FeatureMonthBegin,
									FeatureMonthEnd = FeatureMonthEnd,
									month=month
				)		

		# MakeFeature_Action_
		for FeatureMonthBegin, FeatureMonthEnd, month in FeatureMonthList:
			self.MakeFeature_Action_(
									FeatureMonthBegin = FeatureMonthBegin,
									FeatureMonthEnd = FeatureMonthEnd,
									month=month
				)
		# MakeFeature_Action_Order_
		for FeatureMonthBegin, FeatureMonthEnd, month in FeatureMonthList:
			self.MakeFeature_Action_Order_(
									FeatureMonthBegin = FeatureMonthBegin,
									FeatureMonthEnd = FeatureMonthEnd,
									month=month
				)
		self.TrainColumns = [col for col in self.data_BuyOrNot_FirstTime.columns if col not in self.IDColumns + self.LabelColumns]


	def MakeLabel_(self):
		self.data_BuyOrNot_FirstTime = self.DataLoader.df_user_info

		if self.MakeLabel:
			df_user_order_sku = self.DataLoader.df_user_order.merge(self.DataLoader.df_sku_info,on='sku_id',how='left')

			label_temp_ = df_user_order_sku[(df_user_order_sku['o_date']>=self.PredMonthBegin) &\
										(df_user_order_sku['o_date']<=self.PredMonthEnd)]

			label_temp_30_101 = label_temp_[(label_temp_['cate']==30) | (label_temp_['cate']==101)]
			# label_temp_30_101.to_csv('label.csv',index=False)

			# 统计用户当月下单数 回归建模
			BuyOrNotLabel_30_101 = label_temp_30_101.groupby(['user_id'])['o_id'].\
										nunique().\
										reset_index().\
										rename(columns={'user_id':'user_id','o_id':'label1'})
			
			# ############################# 修改 ##########################
			# # 所有已经下单的用户
			BuyOrNotLabel_30_101.label1=1

				
			# 用户首次下单时间 回归建模
			# keep first 获得首次下单时间 - 月初时间 = 下单在当月第几天购买
			FirstTimeLabel_30_101 = label_temp_30_101.\
			 						drop_duplicates('user_id', keep='first')[['user_id','o_date']].\
			 						rename(columns={'user_id':'user_id','o_date':'label2'})
			FirstTimeLabel_30_101['label2'] = (FirstTimeLabel_30_101['label2'] - self.PredMonthBegin).dt.days

			# 是否购买：1/0
			self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(BuyOrNotLabel_30_101,on='user_id', how='left').fillna(0)
			# 购买日期：-1 表示当月没有购买														
			self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(FirstTimeLabel_30_101, on='user_id', how='left').fillna(-1)
		else:
			self.data_BuyOrNot_FirstTime['label1'] = -1
			self.data_BuyOrNot_FirstTime['label2'] = -1

		return self.data_BuyOrNot_FirstTime

	def MakeFeature_Order_Comment_(self, 
								FeatureMonthBegin,
								FeatureMonthEnd,
								month
									):
		# Order Comment User Sku
		'''
		self.df_Order_Comment_User_Sku
		user_id sku_id o_id o_date o_area o_sku_num comment_create_tm score_level age sex user_lv_cd price cate para_1 para_2 para_3
		'''
		BetweenFlag = 'M'+str(month)+'_'
		features_temp_Order_ = self.df_Order_Comment_User_Sku[(self.df_Order_Comment_User_Sku['o_date']>=FeatureMonthBegin) &\
															(self.df_Order_Comment_User_Sku['o_date']<=FeatureMonthEnd)]

		# make features	
		#################################################################################################################												
		# 目标品类 30 101 订单数
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']==30) | (features_temp_Order_['cate']==101)].\
											groupby(['user_id'])['o_id'].\
											nunique().\
											reset_index().\
											rename(columns={'user_id':'user_id','o_id':BetweenFlag+'o_id_cate_30_101_cnt'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')

		# 目标品类 30 订单数
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']==30)].\
											groupby(['user_id'])['o_id'].\
											nunique().\
											reset_index().\
											rename(columns={'user_id':'user_id','o_id':BetweenFlag+'o_id_cate_30_cnt'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')

		# 目标品类 101 订单数
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']==101)].\
											groupby(['user_id'])['o_id'].\
											nunique().\
											reset_index().\
											rename(columns={'user_id':'user_id','o_id':BetweenFlag+'o_id_cate_101_cnt'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')

		#  非目标品类 订单数
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']!=30) & (features_temp_Order_['cate']!=101)].\
											groupby(['user_id'])['o_id'].\
											nunique().\
											reset_index().\
											rename(columns={'user_id':'user_id','o_id':BetweenFlag+'o_id_cate_other_cnt'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')


		#################################################################################################################	
		# sku_id cate 30 101 购买商品次数，可能不同的商品是一个订单
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']==30) | (features_temp_Order_['cate']==101)].\
											groupby(['user_id'])['sku_id'].\
											count().\
											reset_index().\
											rename(columns={'user_id':'user_id','sku_id':BetweenFlag+'sku_id_cate_30_101_cnt'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')

		# sku_id cate 30 购买商品次数
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']==30)].\
											groupby(['user_id'])['sku_id'].\
											count().\
											reset_index().\
											rename(columns={'user_id':'user_id','sku_id':BetweenFlag+'sku_id_cate_30_cnt'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')		

		# sku_id cate 101 购买商品次数
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']==101)].\
											groupby(['user_id'])['sku_id'].\
											count().\
											reset_index().\
											rename(columns={'user_id':'user_id','sku_id':BetweenFlag+'sku_id_cate_101_cnt'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')

		# sku_id cate 非 30 101 购买商品次数
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']!=30) & (features_temp_Order_['cate']!=101)].\
											groupby(['user_id'])['sku_id'].\
											count().\
											reset_index().\
											rename(columns={'user_id':'user_id','sku_id':BetweenFlag+'sku_id_cate_other_cnt'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')


		#################################################################################################################	
		# o_date cate 30 101 购买天数
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']==30) | (features_temp_Order_['cate']==101)].\
											groupby(['user_id'])['o_date'].\
											nunique().\
											reset_index().\
											rename(columns={'user_id':'user_id','o_date':BetweenFlag+'o_date_cate_30_101_nuique'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')	

		# 其他品类 购买天数
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']!=30) & (features_temp_Order_['cate']!=101)].\
											groupby(['user_id'])['o_date'].\
											nunique().\
											reset_index().\
											rename(columns={'user_id':'user_id','o_date':BetweenFlag+'o_date_cate_other_nuique'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')											


		#################################################################################################################	
		# o_sku_num cate 30 101 用户购买件数
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']==30) | (features_temp_Order_['cate']==101)].\
											groupby(['user_id'])['o_sku_num'].\
											sum().\
											reset_index().\
											rename(columns={'user_id':'user_id','o_sku_num':BetweenFlag+'o_sku_num_cate_30_101_sum'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')

		'''
		继续添加
		'''
		# 目标品类 平均每次订单购买件数
		self.data_BuyOrNot_FirstTime[BetweenFlag+'num_per_order_target']= \
									self.data_BuyOrNot_FirstTime[BetweenFlag+'o_sku_num_cate_30_101_sum'] / \
									self.data_BuyOrNot_FirstTime[BetweenFlag+'o_date_cate_30_101_nuique']
		# 目标品类 平均每月购买件数
		self.data_BuyOrNot_FirstTime[BetweenFlag+'num_per_month_target']= \
									self.data_BuyOrNot_FirstTime[BetweenFlag+'o_sku_num_cate_30_101_sum'] / month
		
		# 目标品类与非目标品类 总订单数
		features_temp_ = features_temp_Order_.groupby(['user_id'])['o_id'].\
											nunique().\
											reset_index().\
											rename(columns={'user_id':'user_id','o_id':BetweenFlag+'o_id_cnt'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')

		# 目标品类与非目标品类 总件数
		features_temp_ = features_temp_Order_.groupby(['user_id'])['o_sku_num'].\
											sum().\
											reset_index().\
											rename(columns={'user_id':'user_id','o_sku_num':BetweenFlag+'o_sku_num_sum'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left') 
		# print(self.data_BuyOrNot_FirstTime[[BetweenFlag+'o_sku_num_sum']])

		# 目标品类与非目标品类 购买天数
		features_temp_ = features_temp_Order_.groupby(['user_id'])['o_date'].\
											nunique().\
											reset_index().\
											rename(columns={'user_id':'user_id','o_date':BetweenFlag+'o_date_nuique'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')

		# 所有品类 平均每次订单购买件数
		self.data_BuyOrNot_FirstTime[BetweenFlag+'num_per_order']= \
									self.data_BuyOrNot_FirstTime[BetweenFlag+'o_sku_num_sum'] / \
									self.data_BuyOrNot_FirstTime[BetweenFlag+'o_date_nuique']										
							
		# 所有品类 平均每月购买件数
		self.data_BuyOrNot_FirstTime[BetweenFlag+'num_per_month']= \
									self.data_BuyOrNot_FirstTime[BetweenFlag+'o_sku_num_sum'] / month

		# 需不需加入平均开销？？？
		

		# 时间特征,购买时间属性，不属于时间间隔特征
		#################################################################################################################	
		# 第一次购买时间
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']==30) | (features_temp_Order_['cate']==101)].\
											groupby(['user_id'])['day'].\
											min().\
											reset_index().\
											rename(columns={'user_id':'user_id','day':BetweenFlag+'day_cate_30_101_firstday'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')
		# cate30 第一次购买时间
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']==30)].\
											groupby(['user_id'])['day'].\
											min().\
											reset_index().\
											rename(columns={'user_id':'user_id','day':BetweenFlag+'day_cate_30_firstday'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')
		
		#cate 101 第一次购买时间
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']==101)].\
											groupby(['user_id'])['day'].\
											min().\
											reset_index().\
											rename(columns={'user_id':'user_id','day':BetweenFlag+'day_cate_101_firstday'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')
		#################################################################################################################	
		# 最后一次购买时间
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']==30) | (features_temp_Order_['cate']==101)].\
											groupby(['user_id'])['day'].\
											max().\
											reset_index().\
											rename(columns={'user_id':'user_id','day':BetweenFlag+'day_cate_30_101_lastday'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')
		

		#################################################################################################################	
		# 购买当月平均第几天
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']==30) | (features_temp_Order_['cate']==101)].\
											groupby(['user_id'])['day'].\
											mean().\
											reset_index().\
											rename(columns={'user_id':'user_id','day':BetweenFlag+'day_cate_30_101_meanday'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')


		# 购买月份数
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']==30) | (features_temp_Order_['cate']==101)].\
											groupby(['user_id'])['month'].\
											nunique().\
											reset_index().\
											rename(columns={'user_id':'user_id','month':BetweenFlag+'month_cate_30_101_monthnum'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')

		'''
		继续添加
		'''
		# 其他品类 第一次购买时间
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']!=30) & (features_temp_Order_['cate']!=101)].\
											groupby(['user_id'])['day'].\
											min().\
											reset_index().\
											rename(columns={'user_id':'user_id','day':BetweenFlag+'day_cate_other_firstday'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')
		# 其他品类 最后一次购买时间
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']!=30) & (features_temp_Order_['cate']!=101)].\
											groupby(['user_id'])['day'].\
											max().\
											reset_index().\
											rename(columns={'user_id':'user_id','day':BetweenFlag+'day_cate_other_lastday'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')

		# 其他品类 购买当月平均第几天
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']!=30) & (features_temp_Order_['cate']!=101)].\
											groupby(['user_id'])['day'].\
											mean().\
											reset_index().\
											rename(columns={'user_id':'user_id','day':BetweenFlag+'day_cate_other_meanday'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')

		# 其它品类 购买月份数
		features_temp_ = features_temp_Order_[(features_temp_Order_['cate']!=30) & (features_temp_Order_['cate']!=101)].\
											groupby(['user_id'])['month'].\
											nunique().\
											reset_index().\
											rename(columns={'user_id':'user_id','month':BetweenFlag+'month_cate_other_monthnum'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')

		###########################################################################

		# 目标品类 购买时间间隔
		features_temp_ = features_temp_Order_[features_temp_Order_.cate.isin((30,101))].\
											loc[:,['user_id','o_date','o_sku_num']].\
											sort_values(by=['user_id','o_date'])
		features_temp_diff = features_temp_.diff().rename(columns={'user_id':'user_id_diff'})
		features_temp_diff['user_id'] = features_temp_.user_id
		features_temp_diff.o_date=features_temp_diff.o_date.dt.days

		# o_sku_num 上次订单购买的商品个数
		features_temp_diff.o_sku_num = features_temp_[['o_sku_num']].shift(periods=1)
		###  过滤掉不同用户之间的差值
		features_temp_diff = features_temp_diff[features_temp_diff.user_id_diff == 0].\
												reset_index(drop=True)
		##  特征：day1，名称：BetweenFlag_o_date_diff_sum  目标品类 最后一次购买 距 第一次购买 间隔的天数
		day1 = features_temp_diff.groupby(['user_id']).o_date.sum().reset_index().\
											rename(columns={'o_date':BetweenFlag+'o_date_diff_sum'})

		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(day1,on=['user_id'],how='left')

		# 目标品类 最后一次购买之前 购买的总数 sum1 
		# 特征名称：last_o_sku_num_sum
		sum1 = features_temp_diff.groupby(['user_id']).o_sku_num.sum().reset_index().\
												rename(columns={'o_sku_num':BetweenFlag+'last_o_sku_num_sum'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(sum1,on=['user_id'],how='left')

		# sum1 / day1  平均每个商品用多少天
		self.data_BuyOrNot_FirstTime[BetweenFlag+'day_per_sku_target'] = \
											self.data_BuyOrNot_FirstTime[BetweenFlag+'o_date_diff_sum'] / \
											self.data_BuyOrNot_FirstTime[BetweenFlag+'last_o_sku_num_sum']
		# 目标品类 购买时间间隔的方差
		o_day_diff_var = features_temp_diff.groupby(['user_id']).o_date.var().reset_index().\
											rename(columns={'o_date':BetweenFlag+'o_date_diff_var'})

		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(o_day_diff_var,on=['user_id'],how='left')
		# 目标品类 购买时间间隔的均值
		o_day_diff_var = features_temp_diff.groupby(['user_id']).o_date.mean().reset_index().\
											rename(columns={'o_date':BetweenFlag+'o_date_diff_mean'})

		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(o_day_diff_var,on=['user_id'],how='left')
		###################################################################################################
		
		# 其他品类 最后一次购买 距 第一次购买 间隔的天数 day2
		features_temp_ = features_temp_Order_[(features_temp_Order_.cate!=30) & (features_temp_Order_.cate!=101)].\
											loc[:,['user_id','o_date','o_sku_num']].\
											sort_values(by=['user_id','o_date'])
		features_temp_diff = features_temp_.diff().rename(columns={'user_id':'user_id_diff'})
		features_temp_diff['user_id'] = features_temp_.user_id
		features_temp_diff.o_date=features_temp_diff.o_date.dt.days

		# o_sku_num 上次订单购买的商品个数
		features_temp_diff.o_sku_num = features_temp_[['o_sku_num']].shift(periods=1)

		###  过滤掉不同用户之间的差值
		features_temp_diff = features_temp_diff[features_temp_diff.user_id_diff == 0].\
												reset_index(drop=True)
		
		##  特征：day2，名称：BetweenFlag_o_date_diff_other
		day2 = features_temp_diff.groupby(['user_id']).o_date.sum().reset_index().\
											rename(columns={'o_date':BetweenFlag+'o_date_diff_sum_other'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(day2,on=['user_id'],how='left')

		# 其他品类 最后一次购买之前 购买的总数 sum2
		sum2 = features_temp_diff.groupby(['user_id']).o_sku_num.sum().reset_index( ).\
												rename(columns={'o_sku_num':BetweenFlag+'last_o_sku_num_sum_other'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(sum2,on=['user_id'],how='left')

		# sum2 / day2
		self.data_BuyOrNot_FirstTime[BetweenFlag+'day_per_sku_other'] = \
											self.data_BuyOrNot_FirstTime[BetweenFlag+'o_date_diff_sum_other'] / \
											self.data_BuyOrNot_FirstTime[BetweenFlag+'last_o_sku_num_sum_other']
		# 其它品类 购买日期间隔的方差
		o_day_diff_var = features_temp_diff.groupby(['user_id']).o_date.var().reset_index().\
											rename(columns={'o_date':BetweenFlag+'o_date_diff_var_other'})

		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(o_day_diff_var,on=['user_id'],how='left')
		# 其它品类 购买日期间隔的均值
		o_day_diff_var = features_temp_diff.groupby(['user_id']).o_date.mean().reset_index().\
											rename(columns={'o_date':BetweenFlag+'o_date_diff_mean_other'})

		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(o_day_diff_var,on=['user_id'],how='left')
		
		# 目标品类 最后一次购买的个数
		features_temp_ = features_temp_Order_[features_temp_Order_.cate.isin((30,101))].drop_duplicates('user_id',keep='last').\
											loc[:,['user_id','o_sku_num']].\
											reset_index(drop=True).\
											rename(columns={'o_sku_num':BetweenFlag+'last_time_o_sku_num'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')
		
		# 其他品类 最后一次购买的个数
		features_temp_ = features_temp_Order_[(features_temp_Order_.cate!=30) & (features_temp_Order_.cate!=101)].\
											drop_duplicates('user_id',keep='last').\
											loc[:,['user_id','o_sku_num']].\
											reset_index(drop=True).\
											rename(columns={'o_sku_num':BetweenFlag+'last_time_o_sku_num_other'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')
		


	def MakeFeature_Action_(self,
							FeatureMonthBegin,
							FeatureMonthEnd,
							month
							):
		# Action User Sku
		'''
		self.df_Action_User_Sku
		user_id sku_id a_date a_num a_type age sex user_lv_cd price cate para_1 para_2 para_3
		'''
		BetweenFlag = 'M'+str(month)+'_'
		features_temp_Action_ = self.df_Action_User_Sku[(self.df_Action_User_Sku['a_date']>=FeatureMonthBegin) &\
														(self.df_Action_User_Sku['a_date']<=FeatureMonthEnd)]
		
		# 目标品类浏览+关注商品种类数
		features_temp_ = features_temp_Action_[(features_temp_Action_['cate']==30) | (features_temp_Action_['cate']==101)].\
											groupby(['user_id'])['sku_id'].\
											nunique().\
											reset_index().\
											rename(columns={'user_id':'user_id','sku_id':BetweenFlag+'sku_id_cate_30_101_cnt'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')

		# 对目标品类浏览+关注的天数
		features_temp_ = features_temp_Action_[(features_temp_Action_['cate']==30) | (features_temp_Action_['cate']==101)].\
											groupby(['user_id'])['a_date'].\
											nunique().\
											reset_index().\
											rename(columns={'user_id':'user_id','a_date':BetweenFlag+'num_of_day_action'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')	

		'''
		# 继续整加特征
		'''
		#  目标品类 action 总次数
		features_temp_ = features_temp_Action_[(features_temp_Action_['cate']==30) | (features_temp_Action_['cate']==101)].\
											groupby(['user_id']).a_num.agg('sum').\
											reset_index().\
											rename(columns={'a_num':BetweenFlag+'num_of_action'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')
		
		# 目标品类平均每天action的次数
		self.data_BuyOrNot_FirstTime[BetweenFlag+'num_of_action_perday']= \
			self.data_BuyOrNot_FirstTime[BetweenFlag+'num_of_action']/ \
			self.data_BuyOrNot_FirstTime[BetweenFlag+'num_of_day_action']

		# 其他品类action总次数
		features_temp_ = features_temp_Action_[(features_temp_Action_['cate']!=30) & (features_temp_Action_['cate']!=101)].\
											groupby(['user_id']).a_num.agg('sum').\
											reset_index().\
											rename(columns={'a_num':BetweenFlag+'num_of_action_other'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')

		# 其他品类浏览商品种类数
		features_temp_ = features_temp_Action_[(features_temp_Action_['cate']!=30) & (features_temp_Action_['cate']!=101)].\
											groupby(['user_id'])['sku_id'].\
											nunique().\
											reset_index().\
											rename(columns={'user_id':'user_id','sku_id':BetweenFlag+'sku_id_cate_other_cnt'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')

		# 浏览其他品类的天数
		features_temp_ = features_temp_Action_[(features_temp_Action_['cate']!=30) & (features_temp_Action_['cate']!=101)].\
											groupby(['user_id'])['a_date'].\
											nunique().\
											reset_index().\
											rename(columns={'user_id':'user_id','a_date':BetweenFlag+'num_of_day_action_other'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')	
		# 其他品类平均每天action的次数
		self.data_BuyOrNot_FirstTime[BetweenFlag+'num_of_action_perday_other']= \
			self.data_BuyOrNot_FirstTime[BetweenFlag+'num_of_action_other']/ \
			self.data_BuyOrNot_FirstTime[BetweenFlag+'num_of_day_action_other']

		# 浏览所有品类的总次数
		features_temp_ = features_temp_Action_.groupby(['user_id']).a_num.agg('sum').\
											reset_index().\
											rename(columns={'a_num':BetweenFlag+'num_of_action_all'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')

		# 浏览所有品类的天数
		features_temp_ = features_temp_Action_.groupby(['user_id'])['a_date'].\
											nunique().\
											reset_index().\
											rename(columns={'user_id':'user_id','a_date':BetweenFlag+'num_of_day_action_all'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')

		# 平均每天浏览所有品类的次数
		self.data_BuyOrNot_FirstTime[BetweenFlag+'num_of_action_perday_all']= \
			self.data_BuyOrNot_FirstTime[BetweenFlag+'num_of_action_all']/ \
			self.data_BuyOrNot_FirstTime[BetweenFlag+'num_of_day_action_all']

	def MakeFeature_Action_Order_(self, 
								FeatureMonthBegin,
								FeatureMonthEnd,
								month):
		# BetweenFlag = 'A_O'+str(month)+'_'
		BetweenFlag = 'M'+str(month)+'_'
		
		# 目标品类 总的action 天数和下单天数 之比
		self.data_BuyOrNot_FirstTime[BetweenFlag+'action_order_day_ratio_target'] = \
								self.data_BuyOrNot_FirstTime[BetweenFlag+'num_of_day_action'] / \
								self.data_BuyOrNot_FirstTime[BetweenFlag+'o_date_cate_30_101_nuique']
		
		# 目标品类 action次数和下单数之比
		self.data_BuyOrNot_FirstTime[BetweenFlag+'action_order_num_ratio_target']=\
				self.data_BuyOrNot_FirstTime[BetweenFlag+'num_of_action']/\
				self.data_BuyOrNot_FirstTime[BetweenFlag+'o_id_cate_30_101_cnt']

		# 非目标品类 总的action 天数和下单天数 之比
		self.data_BuyOrNot_FirstTime[BetweenFlag+'action_order_day_ratio_other']=\
				self.data_BuyOrNot_FirstTime[BetweenFlag+'num_of_day_action_other']/\
				self.data_BuyOrNot_FirstTime[BetweenFlag+'o_date_cate_other_nuique']
		
		# 非目标品类 action次数和下单数之比
		self.data_BuyOrNot_FirstTime[BetweenFlag+'action_order_num_ratio_other']=\
			self.data_BuyOrNot_FirstTime[BetweenFlag+'num_of_action_other']/\
			self.data_BuyOrNot_FirstTime[BetweenFlag+'o_id_cate_other_cnt']

		# 所有品类  总的action 天数和下单天数 之比
		self.data_BuyOrNot_FirstTime[BetweenFlag+'action_order_day_ratio_all']=\
				self.data_BuyOrNot_FirstTime[BetweenFlag+'num_of_day_action_all']/\
				self.data_BuyOrNot_FirstTime[BetweenFlag+'o_date_nuique']
		
		# 所有品类 action次数和下单数之比
		self.data_BuyOrNot_FirstTime[BetweenFlag+'action_order_num_ratio_all']=\
				self.data_BuyOrNot_FirstTime[BetweenFlag+'num_of_action_all']/ \
				self.data_BuyOrNot_FirstTime[BetweenFlag+'o_id_cnt']

		###########################################################################################
		action = self.df_Action_User_Sku[(self.df_Action_User_Sku.a_date>=FeatureMonthBegin) & (self.df_Action_User_Sku.a_date<=FeatureMonthEnd)]
		order = self.df_Order_Comment_User_Sku[(self.df_Order_Comment_User_Sku.o_date>=FeatureMonthBegin) & (self.df_Order_Comment_User_Sku.o_date<=FeatureMonthEnd)]
		action_order = action[['user_id','a_date','cate']].merge(order[['user_id','cate','o_date']],on=['user_id','cate'],how='left')
		action_order = action_order[action_order.o_date>action_order.a_date].\
					sort_values(by=['user_id','a_date','o_date']).\
					drop_duplicates(['user_id','cate','o_date'],keep='last')
		action_order['ao_day_gap']=(action_order.o_date-action_order.a_date).dt.days
		
		# 目标品类 下单时间与之前最近一次action的时间间隔
		features_temp_ = action_order[action_order.cate.isin([30,101])].\
					groupby(['user_id']).ao_day_gap.mean().\
					reset_index().rename(columns={'ao_day_gap':BetweenFlag+'ao_day_gap_target'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')
		
		# 非目标品类 下单时间与之前最近一次action的时间间隔
		features_temp_ = action_order[(action_order.cate!=30) & (action_order.cate!=101)].\
					groupby(['user_id']).ao_day_gap.mean().\
					reset_index().rename(columns={'ao_day_gap':BetweenFlag+'ao_day_gap_other'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')
		# 所有品类 下单时间与之前最近一次action的时间间隔
		features_temp_ = action_order.groupby(['user_id']).ao_day_gap.mean().\
					reset_index().rename(columns={'ao_day_gap':BetweenFlag+'ao_day_gap_all'})
		self.data_BuyOrNot_FirstTime = self.data_BuyOrNot_FirstTime.merge(features_temp_,on=['user_id'],how='left')










