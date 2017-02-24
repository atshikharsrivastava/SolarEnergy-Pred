##################################################
#run from shell->  python 6_LSTM_finalModels_eu.py station_number(EU: 1 to 16 or US:1 to 5)
##################################################



from __future__ import print_function
import pandas
import os

import sys
input = sys.argv[1]

station_dat= pandas.read_csv("/home/shikhar/ground_data_correctelev_eu.csv", engine='python')
all_stid = station_dat["stid"].tolist()
station_name = all_stid[int(input)-1]


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

os.chdir("/home/shikhar/Europe/"+str(station_name)+"/")

print ("\n Building for "+station_name+"\n")
timesave_final_all = []
data_parent = pandas.read_csv('datasets/dataset_all_ts.csv', engine='python')


hidden_neurons_parent = 120
hidden_neurons_2L = 120

neurons1 = hidden_neurons_parent
neurons2 = hidden_neurons_2L
days_to_lookback = [1]
# for i in [110,90]:
i=1

per_day_obs=7
time_Steps=int(i)*per_day_obs

offset=1+per_day_obs*(time_Steps/per_day_obs-1)

def create_dataset(dataset, time_Steps=time_Steps,offset=offset):#Restructuring function: Grouping of intra-day timesteps together for per day target value
	dataX , dataY = [], []
	for i in range(0,len(dataset)-offset,per_day_obs):
		a = dataset[i:(i+time_Steps), 0:(dataset.shape[1]-1)]   
		b = dataset[i+time_Steps-1,-1:]

		
		dataX.append(a)
		dataY.append(b)

	return numpy.array(dataX), numpy.array(dataY)

def create_dataset_req(dataset, time_Steps=time_Steps,offset=offset):#function to extract dates
	dataZ = []
	for i in range(0,len(dataset)-offset,per_day_obs):
		c = dataset[i+time_Steps-1,0]
		dataZ.append(c)
	return  numpy.array(dataZ)


train_begindate = 20050000
train_enddate = 20120000
val_enddate = 20140000
test_enddate = 20150000



def build_data(data_parent) :

	
	data_parent = data_parent.loc[data_parent['date'] > train_begindate]
	data_parent = data_parent.loc[data_parent['date'] < test_enddate]
	data_parent_focused = data_parent.loc[data_parent['stid'] == str(station_name)].copy() 
	data_parent = data_parent.drop("stid", 1)
	data_parent_focused = data_parent_focused.drop("stid", 1)
	
	##### defining train, test and val pandas dataframes
	dataframe1 = data_parent.loc[data_parent['date'] < train_enddate]
	dataframe2 = data_parent.loc[data_parent['date'] > train_enddate]
	dataframe3 = dataframe2.loc[dataframe2['date'] < val_enddate]
	dataframe4 = dataframe2.loc[dataframe2['date'] > val_enddate]
	
	##### Station in focus : defining train, test and val pandas dataframes
	dataframe1_focused = data_parent_focused.loc[data_parent_focused['date'] < train_enddate]
	dataframe2_focused = data_parent_focused.loc[data_parent_focused['date'] > train_enddate]
	dataframe3_focused = dataframe2_focused.loc[dataframe2_focused['date'] < val_enddate]
	dataframe4_focused = dataframe2_focused.loc[dataframe2_focused['date'] > val_enddate]
	
	
	columns = (dataframe1.columns.values).tolist()
	# fix random seed for reproducibility
	numpy.random.seed(7)
	start = columns.index("month")
	end = columns.index("energy") + 1
	
	##### changing to numpy arrays
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
	
	##### Station in focus : changing to numpy arrays
	data_framesub21_focused =  dataframe1_focused.ix[:,start:end].copy()
	dataset1_focused = data_framesub21_focused.values
	dataset1_focused = dataset1_focused.astype('float32')
	data_framesub31_focused =  dataframe1_focused[["date"]].copy()
	dataset_req1_focused = data_framesub31_focused.values
	
	data_framesub22_focused =  dataframe3_focused.ix[:,start:end].copy()
	dataset2_focused = data_framesub22_focused.values
	dataset2_focused = dataset2_focused.astype('float32')
	data_framesub32_focused =  dataframe3_focused[["date"]].copy()
	dataset_req2_focused = data_framesub32_focused.values
	
	data_framesub23_focused =  dataframe4_focused.ix[:,start:end].copy()
	dataset3_focused = data_framesub23_focused.values
	dataset3_focused = dataset3_focused.astype('float32')
	data_framesub33_focused =  dataframe4_focused[["date"]].copy()
	dataset_req3_focused = data_framesub33_focused.values
	
	# train, val and test sets
	train_size = len(dataset1)
	val_size = len(dataset2)
	test_size = len(dataset3)
	
	train, val,test= dataset1[0:(train_size),:], dataset2[0:(val_size),:], dataset3[0:(test_size),:]
	train_req, val_req,test_req =  dataset_req1[0:(train_size),:], dataset_req2[0:(val_size),:], dataset_req3[0:(test_size),:]
	
	# Station in focus :  train, val and test sets
	train_size_focused = len(dataset1_focused)
	val_size_focused = len(dataset2_focused)
	test_size_focused = len(dataset3_focused)
	
	train_focused, val_focused,test_focused= dataset1_focused[0:(train_size_focused),:], dataset2_focused[0:(val_size_focused),:], dataset3_focused[0:(test_size_focused),:]
	train_req_focused, val_req_focused,test_req_focused =  dataset_req1_focused[0:(train_size_focused),:], dataset_req2_focused[0:(val_size_focused),:], dataset_req3_focused[0:(test_size_focused),:]
	
	# test[:,-1:] = 0
	idvs = end - start -1
	# normalize the dataset
	full_data_idv = numpy.vstack((train[:,:idvs],val[:,:idvs],test[:,:idvs]))
	idv_scaler = MinMaxScaler(feature_range=(0, 1))
	idv_scaler.fit(full_data_idv)
	train[:,:idvs] = idv_scaler.transform(train[:,:idvs])
	val[:,:idvs] = idv_scaler.transform(val[:,:idvs])
	test[:,:idvs] = idv_scaler.transform(test[:,:idvs])
	
	train_focused[:,:idvs] = idv_scaler.transform(train_focused[:,:idvs])
	val_focused[:,:idvs] = idv_scaler.transform(val_focused[:,:idvs])
	test_focused[:,:idvs] = idv_scaler.transform(test_focused[:,:idvs])
	
	
	scaler = MinMaxScaler(feature_range=(0, 1))
	scaler.fit(train[:,-1:])
	train[:,-1:] = scaler.transform(train[:,-1:])
	val[:,-1:] = scaler.transform(val[:,-1:])
	test[:,-1:] = scaler.transform(test[:,-1:])
	
	train_focused[:,-1:] = scaler.transform(train_focused[:,-1:])
	val_focused[:,-1:] = scaler.transform(val_focused[:,-1:])
	test_focused[:,-1:] = scaler.transform(test_focused[:,-1:])
	
	# create datasets
	trainX, trainY = create_dataset(train)
	valX, valY = create_dataset(val)
	testX, testY = create_dataset(test)
	
	trainDate = create_dataset_req(train_req)
	valDate = create_dataset_req(val_req)
	testDate = create_dataset_req(test_req)
	
	trainX_focused, trainY_focused = create_dataset(train_focused)
	valX_focused, valY_focused = create_dataset(val_focused)
	testX_focused, testY_focused = create_dataset(test_focused)
	
	trainDate_focused = create_dataset_req(train_req_focused)
	valDate_focused = create_dataset_req(val_req_focused)
	testDate_focused = create_dataset_req(test_req_focused)
	
	#reshape for LSTMs input (train_size,time_steps,features)
	trainX = numpy.reshape(trainX, (trainX.shape[0], time_Steps, trainX.shape[2]))
	valX = numpy.reshape(valX, (valX.shape[0], time_Steps, valX.shape[2]))
	testX = numpy.reshape(testX, (testX.shape[0], time_Steps, testX.shape[2]))
	
	trainX_focused = numpy.reshape(trainX_focused, (trainX_focused.shape[0], time_Steps, trainX_focused.shape[2]))
	valX_focused = numpy.reshape(valX_focused, (valX_focused.shape[0], time_Steps, valX_focused.shape[2]))
	testX_focused = numpy.reshape(testX_focused, (testX_focused.shape[0], time_Steps, testX_focused.shape[2]))
	
	return trainX, trainY, trainDate, valX, valY, valDate, testX, testY, testDate,trainX_focused, trainY_focused, trainDate_focused, valX_focused, valY_focused, valDate_focused, testX_focused, testY_focused, testDate_focused, scaler

trainX, trainY, trainDate, valX, valY, valDate, testX, testY, testDate,trainX_focused, trainY_focused, trainDate_focused, valX_focused, valY_focused, valDate_focused, testX_focused, testY_focused, testDate_focused, scaler = build_data(data_parent)



epoch_global=100

### creating sub folders
t_start_neurons = time.time()
if not os.path.exists("/home/shikhar/Europe/"+str(station_name)+"/lstm/"):
	os.makedirs("/home/shikhar/Europe/"+str(station_name)+"/lstm/")

os.chdir("/home/shikhar/Europe/"+str(station_name)+"/lstm/")
if not os.path.exists("RMSE/"):
	os.makedirs("RMSE/")

if not os.path.exists("scores/"):
	os.makedirs("scores/")

directory_name=str(i)+"_day/"
if not os.path.exists(directory_name):
	os.makedirs(directory_name)

os.chdir("/home/shikhar/Europe/"+str(station_name)+"/lstm/"+directory_name)

directory_name_cells1 = str(neurons1)+"_neurons/"
directory_name_cells = str(neurons1)+"_neurons/"+str(neurons2)+"_neurons_2L/"
if not os.path.exists(directory_name_cells1):
	os.makedirs(directory_name_cells1)	

if not os.path.exists(directory_name_cells):
	os.makedirs(directory_name_cells)
	os.chdir("/home/shikhar/Europe/"+str(station_name)+"/lstm/"+directory_name+directory_name_cells)
	os.makedirs("full_models")
	os.makedirs("models")
	os.makedirs("weights")
	os.makedirs("run")

os.chdir("/home/shikhar/Europe/"+str(station_name)+"/lstm/"+directory_name)

batch_size = 256 ###this value directly affects number of epochs required for efficient learning

# create and fit the LSTM network
def create_model(time_Steps,trainX,neurons1,neurons2):
	time_Steps = time_Steps
	model = Sequential()
	model.add(LSTM(neurons1,
				batch_input_shape=(batch_size, time_Steps, trainX.shape[2]),
				stateful=False,
				activation='relu', 
				return_sequences=True))
	model.add(LSTM(neurons2,
				batch_input_shape=(batch_size, time_Steps, neurons1),
				stateful=False,
				activation='relu',
				return_sequences=False))
	model.add(Dropout(0.3))
	model.add(Dense(1)) 
	adam=optimizers.Adam(lr=0.0001)
	model.compile(loss='mean_squared_error',optimizer=adam,metrics=['accuracy'])
	return  model

### Model training and evaluating function
def train_and_evaluate_model( train_data, train_target, trainDate,  val_localdata, val_localtarget, val_localDate, test_localdata, test_localtarget, test_localDate,  scaler ,directory_name_cells,time_Steps,neurons1,neurons2,epoch_global):
	
	model = create_model(time_Steps,train_data,neurons1,neurons2)	
	
	epochs_value = epoch_global
	
	t0=time.time()
	epochs = epochs_value
	modelloss=[]
	modelval_loss=[]
	for iter in range(epochs):
		model_name = "all_model_iter"+str(iter+1)
		model_weights = model_name+".h5"
		print('Test_level Epoch: ' + str(iter+1) +"/"+str(epochs) +" Neurons: "+str(neurons1)+"_"+str(neurons2))
		x=model.fit(train_data, train_target, nb_epoch=1, batch_size=batch_size, validation_data=(val_localdata,val_localtarget),verbose=2, shuffle=True)
		##Epoch optimizations
		modelloss.extend(x.history['loss'])
		modelval_loss.extend(x.history['val_loss'])
		if iter+1 == epochs:
			model.save(directory_name_cells+"full_models/"+model_weights)
		
		# serialize model to JSON
		model_json = model.to_json()
		with open(directory_name_cells+"models/"+model_name, "w") as json_file:
			json_file.write(model_json)
		# serialize weights to HDF5
		model.save_weights(directory_name_cells+"weights/"+model_weights)
		model_final_name = model_name
		model_final_weight = model_weights
		print ("\n Building for "+station_name+"\n")
		
		#predictions and performance on whole region
		valPredict = model.predict(val_localdata,batch_size=batch_size)
		valPredict = (valPredict * scaler.data_range_[scaler.data_range_.shape[0]-1,]) + scaler.min_[scaler.min_.shape[0]-1,]
		y_val = (val_localtarget * scaler.data_range_[scaler.data_range_.shape[0]-1,]) + scaler.min_[scaler.min_.shape[0]-1,]
		valScore = math.sqrt(sklearn.metrics.mean_squared_error(y_val[:,0], valPredict[:,0]))  ## RMSE
		valScore_rel = valScore/(numpy.mean(y_val[:,0]))*100 ##relative RMSE
		
		test_localPredict = model.predict(test_localdata,batch_size=batch_size)
		test_localPredict = (test_localPredict * scaler.data_range_[scaler.data_range_.shape[0]-1,]) + scaler.min_[scaler.min_.shape[0]-1,]
		y_test_local = (test_localtarget * scaler.data_range_[scaler.data_range_.shape[0]-1,]) + scaler.min_[scaler.min_.shape[0]-1,]
		test_localScore = math.sqrt(sklearn.metrics.mean_squared_error(y_test_local[:,0], test_localPredict[:,0]))## RMSE
		test_localScore_rel = test_localScore/(numpy.mean(y_test_local[:,0]))*100
		
		trainPredict = model.predict(train_data,batch_size=batch_size)
		trainPredict = (trainPredict * scaler.data_range_[scaler.data_range_.shape[0]-1,]) + scaler.min_[scaler.min_.shape[0]-1,]
		y_train = (train_target * scaler.data_range_[scaler.data_range_.shape[0]-1,]) + scaler.min_[scaler.min_.shape[0]-1,]
		trainScore = math.sqrt(sklearn.metrics.mean_squared_error(y_train[:,0], trainPredict[:,0]))## RMSE
		trainScore_rel = trainScore/(numpy.mean(y_train[:,0]))*100
		
		df_tosave = pandas.DataFrame({'iterations' : [int(iter+1)], 'train_score' : [(trainScore)], 'val_score' : [(valScore)],'test_score' : [(test_localScore)], 'train_score_rel' : [(trainScore_rel)], 'val_score_rel' : [(valScore_rel)],'test_score_rel' : [(test_localScore_rel)]})
		cols=['iterations', 'train_score', 'val_score','test_score', 'train_score_rel', 'val_score_rel','test_score_rel']
		df_tosave=df_tosave[cols]
		
		##saving performance scores for each iteration
		filename2 = "regional_RMSE.csv"
		
		if  (iter+1) == 1:
			os.chdir("/home/shikhar/Europe/"+str(station_name)+"/lstm/RMSE/")
			df_tosave.to_csv(filename2, sep=',',index=False)
			os.chdir("/home/shikhar/Europe/"+str(station_name)+"/lstm/"+directory_name)
		else:
			os.chdir("/home/shikhar/Europe/"+str(station_name)+"/lstm/RMSE/")
			with open(filename2, 'a') as f:
				(df_tosave).to_csv(f, sep=',',index=False,header=False)
			os.chdir("/home/shikhar/Europe/"+str(station_name)+"/lstm/"+directory_name)
		
		#predictions and performance on single location
		valPredict_focused = model.predict(valX_focused,batch_size=batch_size)
		valPredict_focused = (valPredict_focused * scaler.data_range_[scaler.data_range_.shape[0]-1,]) + scaler.min_[scaler.min_.shape[0]-1,]
		y_val_focused = (valY_focused * scaler.data_range_[scaler.data_range_.shape[0]-1,]) + scaler.min_[scaler.min_.shape[0]-1,]
		valScore_focused = math.sqrt(sklearn.metrics.mean_squared_error(y_val_focused[:,0], valPredict_focused[:,0]))## RMSE
		valScore_focused_rel = valScore_focused/(numpy.mean(y_val_focused[:,0]))*100
		
		test_localPredict_focused = model.predict(testX_focused,batch_size=batch_size)
		test_localPredict_focused = (test_localPredict_focused * scaler.data_range_[scaler.data_range_.shape[0]-1,]) + scaler.min_[scaler.min_.shape[0]-1,]
		y_test_local_focused = (testY_focused * scaler.data_range_[scaler.data_range_.shape[0]-1,]) + scaler.min_[scaler.min_.shape[0]-1,]
		test_localScore_focused = math.sqrt(sklearn.metrics.mean_squared_error(y_test_local_focused[:,0], test_localPredict_focused[:,0]))## RMSE
		test_localScore_focused_rel = test_localScore_focused/(numpy.mean(y_test_local_focused[:,0]))*100
		
		trainPredict_focused = model.predict(trainX_focused,batch_size=batch_size)
		trainPredict_focused = (trainPredict_focused * scaler.data_range_[scaler.data_range_.shape[0]-1,]) + scaler.min_[scaler.min_.shape[0]-1,]
		y_train_focused = (trainY_focused * scaler.data_range_[scaler.data_range_.shape[0]-1,]) + scaler.min_[scaler.min_.shape[0]-1,]
		trainScore_focused = math.sqrt(sklearn.metrics.mean_squared_error(y_train_focused[:,0], trainPredict_focused[:,0]))## RMSE
		trainScore_focused_rel = trainScore_focused/(numpy.mean(y_train_focused[:,0]))*100
		
		df_tosave2 = pandas.DataFrame({'iterations' : [int(iter+1)], 'train_score' : [(trainScore_focused)], 'val_score' : [(valScore_focused)],'test_score' : [(test_localScore_focused)], 'train_score_rel' : [(trainScore_focused_rel)], 'val_score_rel' : [(valScore_focused_rel)],'test_score_rel' : [(test_localScore_focused_rel)]})
		cols=['iterations', 'train_score', 'val_score','test_score', 'train_score_rel', 'val_score_rel','test_score_rel']
		df_tosave2=df_tosave2[cols]
		
		###saving performance scores for each iteration
		filename3 = "grnd_station_RMSE.csv"
		
		if  (iter+1) == 1:
			os.chdir("/home/shikhar/Europe/"+str(station_name)+"/lstm/RMSE/")
			df_tosave2.to_csv(filename3, sep=',',index=False)
			os.chdir("/home/shikhar/Europe/"+str(station_name)+"/lstm/"+directory_name)
		else:
			os.chdir("/home/shikhar/Europe/"+str(station_name)+"/lstm/RMSE/")
			with open(filename3, 'a') as f:
				(df_tosave2).to_csv(f, sep=',',index=False,header=False)
			os.chdir("/home/shikhar/Europe/"+str(station_name)+"/lstm/"+directory_name)
		
		
		df_valdate = pandas.DataFrame(valDate_focused)
		df_val = pandas.DataFrame(valPredict_focused[:,0])
		dfval_save = pandas.concat([df_valdate,df_val], axis=1)
		dfval_save.columns = ['date', "iter_"+str(iter+1)]
		
		###saving predictions for each iteration on validation set
		filename_val= "val_iteration_scores.csv"
		
		if iter+1 == 1 :
			df_testdate = pandas.DataFrame(testDate_focused)
			df_test = pandas.DataFrame(test_localPredict_focused[:,0])
			df_test_real =  pandas.DataFrame(y_test_local_focused[:,0])
			dftest_save = pandas.concat([df_testdate,df_test_real,df_test], axis=1)
			dftest_save.columns = ['date',  'actual',"iter_"+str(iter+1)]
			os.chdir("/home/shikhar/Europe/"+str(station_name)+"/lstm/scores/")
			dftest_save.to_csv(filename_test, sep=',',index=False)
			os.chdir("/home/shikhar/Europe/"+str(station_name)+"/lstm/"+directory_name)
		else :
			df_testdate = pandas.DataFrame(testDate_focused)
			df_test = pandas.DataFrame(test_localPredict_focused[:,0])
			dftest_save = pandas.concat([df_testdate,df_test], axis=1)
			dftest_save.columns = ['date', "iter_"+str(iter+1)]
			os.chdir("/home/shikhar/Europe/"+str(station_name)+"/lstm/scores/")
			test_file = pandas.read_csv(filename_test, engine='python')
			test_file2 = pandas.merge(test_file, dftest_save,how='left', on='date')
			test_file2 = test_file2.sort_values(by='date')
			test_file2.to_csv(filename_test, sep=',',index=False)
			os.chdir("/home/shikhar/Europe/"+str(station_name)+"/lstm/"+directory_name)
		
		df_testdate = pandas.DataFrame(testDate_focused)
		df_test = pandas.DataFrame(test_localPredict_focused[:,0])
		dftest_save = pandas.concat([df_testdate,df_test], axis=1)
		dftest_save.columns = ['date', "iter_"+str(iter+1)]
		
		###saving predictions for each iteration on test set
		filename_test= "iteration_scores.csv"
		
		if iter+1 == 1 :
			df_testdate = pandas.DataFrame(testDate_focused)
			df_test = pandas.DataFrame(test_localPredict_focused[:,0])
			df_test_real =  pandas.DataFrame(y_test_local_focused[:,0])
			dftest_save = pandas.concat([df_testdate,df_test_real,df_test], axis=1)
			dftest_save.columns = ['date',  'actual',"iter_"+str(iter+1)]
			os.chdir("/home/shikhar/Europe/"+str(station_name)+"/lstm/scores/")
			dftest_save.to_csv(filename_test, sep=',',index=False)
			os.chdir("/home/shikhar/Europe/"+str(station_name)+"/lstm/"+directory_name)
		else :
			df_testdate = pandas.DataFrame(testDate_focused)
			df_test = pandas.DataFrame(test_localPredict_focused[:,0])
			dftest_save = pandas.concat([df_testdate,df_test], axis=1)
			dftest_save.columns = ['date', "iter_"+str(iter+1)]
			os.chdir("/home/shikhar/Europe/"+str(station_name)+"/lstm/scores/")
			test_file = pandas.read_csv(filename_test, engine='python')
			test_file2 = pandas.merge(test_file, dftest_save,how='left', on='date')
			test_file2 = test_file2.sort_values(by='date')
			test_file2.to_csv(filename_test, sep=',',index=False)
			os.chdir("/home/shikhar/Europe/"+str(station_name)+"/lstm/"+directory_name)

#running the function
train_and_evaluate_model(trainX, trainY, trainDate,  valX, valY, valDate, testX, testY, testDate,  scaler ,directory_name_cells,time_Steps,neurons1,neurons2,epoch_global)


