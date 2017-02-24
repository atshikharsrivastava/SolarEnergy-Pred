##################################################
#run from shell->  python ann_gridsearch.py hidden_neurons_1stlayer(10,20,40,60,80,100,110,120,140,150)
##################################################



from __future__ import print_function
import pandas
import os

import sys
station_name = "penn_state"

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
data_parent = pandas.read_csv('datasets/dataset_all_ml.csv', engine='python')


hidden_neurons_parent = sys.argv[1]#10,20,40,60,80,100,110,120,140,150
hidden_neurons_2L = [10,20,40,60,80,100,110,120,140,150]


first_arg = sys.argv[1]


neurons1 = int(first_arg)
days_to_lookback = [1]
# for i in [110,90]:
i=1

per_day_obs=1
time_Steps=1

offset=1
def create_dataset(dataset, time_Steps=time_Steps,offset=offset):
	dataX , dataY = [], []
	for i in range(0,len(dataset),1):
		a = dataset[i+time_Steps-1, 0:(dataset.shape[1]-1)]   
		b = dataset[i+time_Steps-1,-1:]
		# c = dataset[i+time_Steps-1,0]
		
		dataX.append(a)
		dataY.append(b)
		# dataZ.append(c)
	return numpy.array(dataX), numpy.array(dataY)

def create_dataset_req(dataset, time_Steps=time_Steps,offset=offset):
	dataZ = []
	for i in range(0,len(dataset),1):
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
	
	
	return trainX, trainY, trainDate, valX, valY, valDate, testX, testY, testDate,trainX_focused, trainY_focused, trainDate_focused, valX_focused, valY_focused, valDate_focused, testX_focused, testY_focused, testDate_focused, scaler

trainX, trainY, trainDate, valX, valY, valDate, testX, testY, testDate,trainX_focused, trainY_focused, trainDate_focused, valX_focused, valY_focused, valDate_focused, testX_focused, testY_focused, testDate_focused, scaler = build_data(data_parent)

	### Running Cross validations

t_start = time.time()


for neurons2 in hidden_neurons_2L:

	if neurons1 <= 70:
		epoch_global = 200
	else :
		epoch_global = 200
	
	t_start_neurons = time.time()
	
	
	if not os.path.exists("/home/shikhar/USA/"+str(station_name)+"/ann_config_selection/"):
		os.makedirs("/home/shikhar/USA/"+str(station_name)+"/ann_config_selection/")
	
	os.chdir("/home/shikhar/USA/"+str(station_name)+"/ann_config_selection/")
	if not os.path.exists("RMSE/"):
		os.makedirs("RMSE/")
	
	if not os.path.exists("scores/"):
		os.makedirs("scores/")
	
	directory_name=str(i)+"_day/"
	if not os.path.exists(directory_name):
		os.makedirs(directory_name)
	
	os.chdir("/home/shikhar/USA/"+str(station_name)+"/ann_config_selection/"+directory_name)
	
	directory_name_cells1 = str(neurons1)+"_neurons/"
	directory_name_cells = str(neurons1)+"_neurons/"+str(neurons2)+"_neurons_2L/"
	if not os.path.exists(directory_name_cells1):
		os.makedirs(directory_name_cells1)	
	
	if not os.path.exists(directory_name_cells):
		os.makedirs(directory_name_cells)
		os.chdir("/home/shikhar/USA/"+str(station_name)+"/ann_config_selection/"+directory_name+directory_name_cells)
		os.makedirs("full_models")
		os.makedirs("models")
		os.makedirs("weights")
		os.makedirs("run")
	
	os.chdir("/home/shikhar/USA/"+str(station_name)+"/ann_config_selection/"+directory_name)
	
	batch_size = 256
	
	# create and fit the ann_config_selection network
	def create_model(time_Steps,trainX,neurons1,neurons2):
		time_Steps = time_Steps
		model = Sequential()
		model.add(Dense(neurons1,
					input_dim=(trainX.shape[1]),
					activation='relu'))
				#model.add(Dropout(0.2))
		model.add(Dense(neurons2,
					input_dim=(neurons1),
					activation='relu'))
		model.add(Dropout(0.3))
		model.add(Dense(1)) 
		# Model Compilation
		adam=optimizers.Adam(lr=0.0001)
		model.compile(loss='mean_squared_error',optimizer=adam,metrics=['accuracy'])
		return  model
	
	
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
			print("\n @@ Saved model to disk @@ \n")
			#predictions and score calculations
			valPredict = model.predict(val_localdata,batch_size=batch_size)
			valPredict = (valPredict * scaler.data_range_[scaler.data_range_.shape[0]-1,]) + scaler.min_[scaler.min_.shape[0]-1,]
			y_val = (val_localtarget * scaler.data_range_[scaler.data_range_.shape[0]-1,]) + scaler.min_[scaler.min_.shape[0]-1,]
			valScore = math.sqrt(sklearn.metrics.mean_squared_error(y_val[:,0], valPredict[:,0]))
			valScore_rel = valScore/(numpy.mean(y_val[:,0]))*100
			
			test_localPredict = model.predict(test_localdata,batch_size=batch_size)
			test_localPredict = (test_localPredict * scaler.data_range_[scaler.data_range_.shape[0]-1,]) + scaler.min_[scaler.min_.shape[0]-1,]
			y_test_local = (test_localtarget * scaler.data_range_[scaler.data_range_.shape[0]-1,]) + scaler.min_[scaler.min_.shape[0]-1,]
			test_localScore = math.sqrt(sklearn.metrics.mean_squared_error(y_test_local[:,0], test_localPredict[:,0]))
			test_localScore_rel = test_localScore/(numpy.mean(y_test_local[:,0]))*100
			
			trainPredict = model.predict(train_data,batch_size=batch_size)
			trainPredict = (trainPredict * scaler.data_range_[scaler.data_range_.shape[0]-1,]) + scaler.min_[scaler.min_.shape[0]-1,]
			y_train = (train_target * scaler.data_range_[scaler.data_range_.shape[0]-1,]) + scaler.min_[scaler.min_.shape[0]-1,]
			trainScore = math.sqrt(sklearn.metrics.mean_squared_error(y_train[:,0], trainPredict[:,0]))
			trainScore_rel = trainScore/(numpy.mean(y_train[:,0]))*100
			
			df_tosave = pandas.DataFrame({'neurons_parent' : [int(neurons1)], 'neurons_2L' : [int(neurons2)],'iterations' : [int(iter+1)], 'train_score' : [(trainScore)], 'val_score' : [(valScore)],'test_score' : [(test_localScore)], 'train_score_rel' : [(trainScore_rel)], 'val_score_rel' : [(valScore_rel)],'test_score_rel' : [(test_localScore_rel)]})
			cols=['neurons_parent','neurons_2L','iterations', 'train_score', 'val_score','test_score', 'train_score_rel', 'val_score_rel','test_score_rel']
			df_tosave=df_tosave[cols]
			
			filename2 = "regional_RMSE.csv"
			
			if not os.path.exists("/home/shikhar/USA/"+str(station_name)+"/ann_config_selection/RMSE/"+filename2):
				os.chdir("/home/shikhar/USA/"+str(station_name)+"/ann_config_selection/RMSE/")
				df_tosave.to_csv(filename2, sep=',',index=False)
				os.chdir("/home/shikhar/USA/"+str(station_name)+"/ann_config_selection/"+directory_name)
			else:
				os.chdir("/home/shikhar/USA/"+str(station_name)+"/ann_config_selection/RMSE/")
				with open(filename2, 'a') as f:
					(df_tosave).to_csv(f, sep=',',index=False,header=False)
				os.chdir("/home/shikhar/USA/"+str(station_name)+"/ann_config_selection/"+directory_name)
				
			valPredict_focused = model.predict(valX_focused,batch_size=batch_size)
			valPredict_focused = (valPredict_focused * scaler.data_range_[scaler.data_range_.shape[0]-1,]) + scaler.min_[scaler.min_.shape[0]-1,]
			y_val_focused = (valY_focused * scaler.data_range_[scaler.data_range_.shape[0]-1,]) + scaler.min_[scaler.min_.shape[0]-1,]
			valScore_focused = math.sqrt(sklearn.metrics.mean_squared_error(y_val_focused[:,0], valPredict_focused[:,0]))
			valScore_focused_rel = valScore_focused/(numpy.mean(y_val_focused[:,0]))*100
			
			test_localPredict_focused = model.predict(testX_focused,batch_size=batch_size)
			test_localPredict_focused = (test_localPredict_focused * scaler.data_range_[scaler.data_range_.shape[0]-1,]) + scaler.min_[scaler.min_.shape[0]-1,]
			y_test_local_focused = (testY_focused * scaler.data_range_[scaler.data_range_.shape[0]-1,]) + scaler.min_[scaler.min_.shape[0]-1,]
			test_localScore_focused = math.sqrt(sklearn.metrics.mean_squared_error(y_test_local_focused[:,0], test_localPredict_focused[:,0]))
			test_localScore_focused_rel = test_localScore_focused/(numpy.mean(y_test_local_focused[:,0]))*100
			
			trainPredict_focused = model.predict(trainX_focused,batch_size=batch_size)
			trainPredict_focused = (trainPredict_focused * scaler.data_range_[scaler.data_range_.shape[0]-1,]) + scaler.min_[scaler.min_.shape[0]-1,]
			y_train_focused = (trainY_focused * scaler.data_range_[scaler.data_range_.shape[0]-1,]) + scaler.min_[scaler.min_.shape[0]-1,]
			trainScore_focused = math.sqrt(sklearn.metrics.mean_squared_error(y_train_focused[:,0], trainPredict_focused[:,0]))
			trainScore_focused_rel = trainScore_focused/(numpy.mean(y_train_focused[:,0]))*100
			
			df_tosave2 = pandas.DataFrame({'neurons_parent' : [int(neurons1)], 'neurons_2L' : [int(neurons2)],'iterations' : [int(iter+1)], 'train_score' : [(trainScore_focused)], 'val_score' : [(valScore_focused)],'test_score' : [(test_localScore_focused)], 'train_score_rel' : [(trainScore_focused_rel)], 'val_score_rel' : [(valScore_focused_rel)],'test_score_rel' : [(test_localScore_focused_rel)]})
			cols=['neurons_parent','neurons_2L','iterations', 'train_score', 'val_score','test_score', 'train_score_rel', 'val_score_rel','test_score_rel']
			df_tosave2=df_tosave2[cols]
			
			filename3 = "grnd_station_RMSE.csv"
			
			if not os.path.exists("/home/shikhar/USA/"+str(station_name)+"/ann_config_selection/RMSE/"+filename2):
				os.chdir("/home/shikhar/USA/"+str(station_name)+"/ann_config_selection/RMSE/")
				df_tosave2.to_csv(filename3, sep=',',index=False)
				os.chdir("/home/shikhar/USA/"+str(station_name)+"/ann_config_selection/"+directory_name)
			else:
				os.chdir("/home/shikhar/USA/"+str(station_name)+"/ann_config_selection/RMSE/")
				with open(filename3, 'a') as f:
					(df_tosave2).to_csv(f, sep=',',index=False,header=False)
				os.chdir("/home/shikhar/USA/"+str(station_name)+"/ann_config_selection/"+directory_name)
			
			


	train_and_evaluate_model(trainX, trainY, trainDate,  valX, valY, valDate, testX, testY, testDate,  scaler ,directory_name_cells,time_Steps,neurons1,neurons2,epoch_global)




