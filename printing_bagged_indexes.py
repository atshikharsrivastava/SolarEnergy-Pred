##################################################
# This code saves indexes for cross validation which are used for running GBM grid search
# Startified cross validation included
##################################################


from __future__ import print_function
import pandas
import os

import sys
station_name = sys.argv[1]

import numpy
import pandas
import math
import os
numpy.random.seed(23)
os.environ['KERAS_BACKEND']='theano'

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Dropout
from keras.layers import TimeDistributed
from keras.models import model_from_json
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn import metrics 
import sklearn
import theano
import copy
import keras.backend as K
from keras import optimizers
from itertools import chain
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import StratifiedShuffleSplit

THEANO_FLAGS='floatX=float32,openmp=True'
OMP_NUM_THREADS=4

from joblib import Parallel, delayed
import multiprocessing




import time

os.chdir("/home/shikhar/USA/"+str(station_name)+"/")

timesave_final_all = []
data_parent = pandas.read_csv('datasets/dataset_all_ts.csv', engine='python')


if not os.path.exists("dates_train_test/"):
	os.makedirs("dates_train_test/")

os.chdir("dates_train_test/")



hidden_neurons_parent = sys.argv[2]#80, 90,100,110, 120, 140
hidden_neurons_2L = [40 ,60, 80, 100,120]

first_arg = sys.argv[2]


neurons1 = int(first_arg)
days_to_lookback = [1]
# for i in [110,90]:
i=1

per_day_obs=5
time_Steps=int(i)*per_day_obs

offset=1+per_day_obs*(time_Steps/per_day_obs-1)

def create_dataset(dataset, time_Steps=time_Steps,offset=offset):
	dataX , dataY = [], []
	for i in range(0,len(dataset)-offset,5):
		a = dataset[i:(i+time_Steps), 0:(dataset.shape[1]-1)]   
		b = dataset[i+time_Steps-1,-1:]
		# c = dataset[i+time_Steps-1,0]
		
		dataX.append(a)
		dataY.append(b)
		# dataZ.append(c)
	return numpy.array(dataX), numpy.array(dataY)

def create_dataset_req(dataset, time_Steps=time_Steps,offset=offset):
	dataZ = []
	for i in range(0,len(dataset)-offset,5):
		c = dataset[i+time_Steps-1,0]
		dataZ.append(c)
	return  numpy.array(dataZ)


def build_data(data_parent) :
	cols_to_del = ["stid","m1_elon","m2_elon","m3_elon","m4_elon","m1_nlat","m2_nlat","m3_nlat","m4_nlat"]
	# data_parent2 = data_parent.copy() 
	for col in cols_to_del:
		data_parent = data_parent.drop(col, 1)
	
	data_parent = data_parent.loc[data_parent['date'] < 20150000]
	dataframe1 = data_parent.loc[data_parent['date'] < 20120000]
	dataframe2 = data_parent.loc[data_parent['date'] > 20120000]
	
	dataframe3 = dataframe2.loc[dataframe2['date'] < 20140000]
	dataframe4 = dataframe2.loc[dataframe2['date'] > 20140000]
	# testdataframe = testdataframe.drop('stid', 1)
	columns = (dataframe1.columns.values).tolist()
	# fix random seed for reproducibility
	numpy.random.seed(7)
	start = columns.index("month")
	end = columns.index("energy") + 1
	
	data_framesub21 =  dataframe1.ix[:,start:end].copy()
	dataset1 = data_framesub21.values
	dataset1 = dataset1.astype('float32')
	data_framesub31 =  dataframe1[["date"]].copy()
	dataset_req1 = data_framesub31.values
	
	data_framesub22 =  dataframe3.ix[:,start:end].copy()
	dataset2 = data_framesub22.values
	dataset2 = dataset2.astype('float32')
	data_framesub32 =  dataframe3[["date"]].copy()
	dataset_req2 = data_framesub32.values
	
	data_framesub23 =  dataframe4.ix[:,start:end].copy()
	dataset3 = data_framesub23.values
	dataset3 = dataset3.astype('float32')
	data_framesub33 =  dataframe4[["date"]].copy()
	dataset_req3 = data_framesub33.values
	
	# train, val and test sets
	train_size = len(dataset1)
	val_size = len(dataset2)
	test_size = len(dataset3)
	
	train, val,test= dataset1[0:(train_size),:], dataset2[0:(val_size),:], dataset3[0:(test_size),:]
	train_req, val_req,test_req =  dataset_req1[0:(train_size),:], dataset_req2[0:(val_size),:], dataset_req3[0:(test_size),:]
	
	# test[:,-1:] = 0
	idvs = end - start -1
	# normalize the dataset
	full_data_idv = numpy.vstack((train[:,:idvs],val[:,:idvs],test[:,:idvs]))
	idv_scaler = MinMaxScaler(feature_range=(0, 1))
	idv_scaler.fit(full_data_idv)
	train[:,:idvs] = idv_scaler.transform(train[:,:idvs])
	val[:,:idvs] = idv_scaler.transform(val[:,:idvs])
	test[:,:idvs] = idv_scaler.transform(test[:,:idvs])
	
	
	scaler = MinMaxScaler(feature_range=(0, 1))
	scaler.fit(train[:,-1:])
	train[:,-1:] = scaler.transform(train[:,-1:])
	val[:,-1:] = scaler.transform(val[:,-1:])
	test[:,-1:] = scaler.transform(test[:,-1:])
	
	# create datasets
	trainX, trainY = create_dataset(train)
	valX, valY = create_dataset(val)
	testX, testY = create_dataset(test)
	
	trainDate = create_dataset_req(train_req)
	valDate = create_dataset_req(val_req)
	testDate = create_dataset_req(test_req)
	
	#reshape
	trainX = numpy.reshape(trainX, (trainX.shape[0], time_Steps, trainX.shape[2]))
	valX = numpy.reshape(valX, (valX.shape[0], time_Steps, valX.shape[2]))
	testX = numpy.reshape(testX, (testX.shape[0], time_Steps, testX.shape[2]))
	
	return trainX, trainY, trainDate, valX, valY, valDate, testX, testY, testDate, scaler

trainX, trainY, trainDate, valX, valY, valDate, testX, testY, testDate, scaler = build_data(data_parent)

	### Running Cross validations
	
def cross_val(trainY):
	data = trainY[:,0]
	bins = numpy.linspace(0, 1, 9)
	digitized = numpy.digitize(data, bins)
	ps = pandas.Series([i for i in digitized])
	counts = ps.value_counts()
	counts_name = numpy.array(counts.index)
	counts_value = numpy.array(counts.values)
	for i in range(0,len(counts_value)):
		if counts_value[i] < 2:
			a = counts_name[i]
			# print(str(i))
			if a < 2:
				b = counts_name[i+1]
			else:
				b = counts_name[i-1]
			digitized[digitized == a] = b		
	sss = StratifiedShuffleSplit(digitized, 2, test_size=0.20, random_state=7)
	return sss

def get_folds(trainY):
	sss = cross_val(trainY)
	return sss

sss = get_folds(trainY)

t_start = time.time()

# def run_parallel(neurons1):
# for neurons1 in hidden_neurons_parent:


def build_with_crossValidations( i):
	# print ("\n\n\n - - # # # # - Starting for: "+ stid +" ("+str(num_stid+1)+")  for lookback of "+str(i)+" day & neurons="+str(neurons)+" - # # # # - - \n\n")
	for fold, (train_index, test_index) in enumerate(sss):
		numpy.random.seed(7)
		# train_index = numpy.append(train_index,(numpy.random.choice(train_index, size=1500, replace=True, p=None)),axis=None)
		df_trainindex = pandas.DataFrame(train_index)
		df_trainindex.columns = ["rows_fold_"+str(fold+1)]
		filename_train= "trainIndexes.csv"
		if fold+1==1:
			df_trainindex.to_csv(filename_train, sep=',',index=False)
		else:
			f = pandas.read_csv(filename_train, engine='python')
			df4 = pandas.concat([f,df_trainindex], axis=1,ignore_index=False)	
			(df4).to_csv(filename_train, sep=',',index=False,header=True)
		
		df_valindex = pandas.DataFrame(test_index)
		df_valindex.columns = ["rows_fold_"+str(fold+1)]
		filename_val= "valIndexes.csv"
		if fold+1==1:
			df_valindex.to_csv(filename_val, sep=',',index=False)
		else:
			f = pandas.read_csv(filename_val, engine='python')
			df4 = pandas.concat([f,df_valindex], axis=1,ignore_index=False)	
			(df4).to_csv(filename_val, sep=',',index=False,header=True)
		
	t1=time.time()
	timesave = str(int(round((t1-t0)/60,0)))
	# print ("\n\n\n - - # # # # - Built Complete for: "+ stid +" ("+str(num_stid+1)+") in: "+ timesave +" mins- # # # # - - \n\n")

build_with_crossValidations(i)

t_over_neurons = time.time()
timesave_neurons = str(round((t_over_neurons-t_start_neurons)/3600,1))

print ("\n\n\n - - # # # # - Completed - # # # # - - \n\n")
print ("\n\n\n - - # # # # - Took : "+ timesave_neurons+" hours - # # # # - - \n\n")

