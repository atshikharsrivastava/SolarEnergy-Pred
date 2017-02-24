##################################################
#run from shell->  python LSTM_gridsearch.py 5  hidden_neurons_1stlayer(10,20,40,60,80,100,110,120,140,150)
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
data_parent = pandas.read_csv('datasets/dataset_all_ts.csv', engine='python')


hidden_neurons_parent = sys.argv[1]#40, 80, 120, 150
hidden_neurons_2L = [10,20,40,60,80,100,110,120,140,150]

first_arg = sys.argv[1]


neurons1 = int(first_arg)
days_to_lookback = [1]
# for i in [110,90]:
i=1

per_day_obs=7
time_Steps=int(i)*per_day_obs

offset=1+per_day_obs*(time_Steps/per_day_obs-1)

def create_dataset(dataset, time_Steps=time_Steps,offset=offset):#Restructuring function: Grouping of intra-day timesteps together for per day target value
	dataX , dataY = [], []
	for i in range(0,len(dataset)-offset,7):
		a = dataset[i:(i+time_Steps), 0:(dataset.shape[1]-1)]   
		b = dataset[i+time_Steps-1,-1:]
		# c = dataset[i+time_Steps-1,0]
		
		dataX.append(a)
		dataY.append(b)
		# dataZ.append(c)
	return numpy.array(dataX), numpy.array(dataY)

def create_dataset_req(dataset, time_Steps=time_Steps,offset=offset):#function to extract dates
	dataZ = []
	for i in range(0,len(dataset)-offset,7):
		c = dataset[i+time_Steps-1,0]
		dataZ.append(c)
	return  numpy.array(dataZ)


def build_data(data_parent) :
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
	
	#reshape for LSTMs input (train_size,time_steps,features)
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



for neurons2 in hidden_neurons_2L:
	### To save computational resources
	if neurons1 <= 70:
		epoch_global = 100
	else :
		epoch_global = 70
	### creating sub folders
	t_start_neurons = time.time()
	if not os.path.exists("/home/shikhar/USA/"+str(station_name)+"/lstm_config_selection/"):
		os.makedirs("/home/shikhar/USA/"+str(station_name)+"/lstm_config_selection/")
	
	os.chdir("/home/shikhar/USA/"+str(station_name)+"/lstm_config_selection/")
	if not os.path.exists("RMSE/"):
		os.makedirs("RMSE/")
	
	directory_name=str(i)+"_day/"
	if not os.path.exists(directory_name):
		os.makedirs(directory_name)
	
	os.chdir("/home/shikhar/USA/"+str(station_name)+"/lstm_config_selection/"+directory_name)
	
	directory_name_cells1 = str(neurons1)+"_neurons/"
	directory_name_cells = str(neurons1)+"_neurons/"+str(neurons2)+"_neurons_2L/"
	if not os.path.exists(directory_name_cells1):
		os.makedirs(directory_name_cells1)	
	
	if not os.path.exists(directory_name_cells):
		os.makedirs(directory_name_cells)
		os.chdir("/home/shikhar/USA/"+str(station_name)+"/lstm_config_selection/"+directory_name+directory_name_cells)
		os.makedirs("full_models")
		os.makedirs("models")
		os.makedirs("weights")
		os.makedirs("run")
	
	os.chdir("/home/shikhar/USA/"+str(station_name)+"/lstm_config_selection/"+directory_name)

	batch_size = 64
	
	# create and fit the LSTM network
	def create_model(time_Steps,trainX,neurons1,neurons2):
		time_Steps = time_Steps
		model = Sequential()
		model.add(LSTM(neurons1,
					batch_input_shape=(batch_size, time_Steps, trainX.shape[2]),
					stateful=False,
					activation='relu', 
					return_sequences=True))
				#model.add(Dropout(0.2))
		model.add(LSTM(neurons2,
					batch_input_shape=(batch_size, time_Steps, neurons1),
					stateful=False,
					activation='relu',
					return_sequences=False))
		model.add(Dropout(0.3))
		model.add(Dense(1)) #, input_dim= trainX.shape[1:]))
		#model.add(Activation('sigmoid'))
		# Model Compilation
		adam=optimizers.Adam(lr=0.0001)
		model.compile(loss='mean_squared_error',optimizer=adam,metrics=['accuracy'])
		return  model
	
	
	def train_and_evaluate_model( train_data, train_target, trainDate, val_data, val_target, val_localdata, val_localtarget, val_localDate, test_localdata, test_localtarget, test_localDate,  scaler, fold, val_index,directory_name_cells,time_Steps,neurons1,neurons2,epoch_global):
		
		model = create_model(time_Steps,train_data,neurons1,neurons2)	
		
		epochs_value = epoch_global
		
		t0=time.time()
		epochs = epochs_value
		modelloss=[]
		modelval_loss=[]
		saved_models_list=[]
		loss_global = 1
		loss_global1 = 9999
		optimal_epochs2 = 9999
		check_loss_best_mean = 1
		optimal_epochs = 10
		for iter in range(epochs):
			model_name = "all_model_"+str(fold+1)+"CV_iter"+str(iter+1)
			model_weights = model_name+".h5"
			print('Test_level Epoch: ' + str(iter+1) +"/"+str(epochs) +" Fold: "+str(fold+1) +" Neurons: "+str(neurons1)+"_"+str(neurons2))
			x=model.fit(train_data, train_target, nb_epoch=1, batch_size=batch_size, validation_data=(val_data,val_target),verbose=2, shuffle=True)
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
			
			#prediction and score calculation
			valPredict = model.predict(val_localdata,batch_size=batch_size)
			valPredict = (valPredict * scaler.data_range_[scaler.data_range_.shape[0]-1,]) + scaler.min_[scaler.min_.shape[0]-1,]
			y_val = (val_localtarget * scaler.data_range_[scaler.data_range_.shape[0]-1,]) + scaler.min_[scaler.min_.shape[0]-1,]
			valScore = math.sqrt(sklearn.metrics.mean_squared_error(y_val[:,0], valPredict[:,0]))
			
			test_localPredict = model.predict(test_localdata,batch_size=batch_size)
			test_localPredict = (test_localPredict * scaler.data_range_[scaler.data_range_.shape[0]-1,]) + scaler.min_[scaler.min_.shape[0]-1,]
			y_test_local = math.sqrt(test_localtarget * scaler.data_range_[scaler.data_range_.shape[0]-1,]) + scaler.min_[scaler.min_.shape[0]-1,]
			
			test_localScore = (sklearn.metrics.mean_squared_error(y_test_local[:,0], test_localPredict[:,0]))
			
			trainPredict = model.predict(train_data,batch_size=batch_size)
			trainPredict = (trainPredict * scaler.data_range_[scaler.data_range_.shape[0]-1,]) + scaler.min_[scaler.min_.shape[0]-1,]
			y_train = (train_target * scaler.data_range_[scaler.data_range_.shape[0]-1,]) + scaler.min_[scaler.min_.shape[0]-1,]
			trainScore = math.sqrt(sklearn.metrics.mean_squared_error(y_train[:,0], trainPredict[:,0]))
			
			df_tosave = pandas.DataFrame({'neurons_parent' : [int(neurons1)], 'neurons_2L' : [int(neurons2)],'fold' : [int(fold+1)],'iterations' : [int(iter+1)], 'train_score' : [(trainScore)], 'val_score' : [(valScore)],'test_localscore' : [(test_localScore)] })
			cols=['neurons_parent', 'neurons_2L','fold' ,'iterations', 'train_score', 'val_score','test_localscore']
			df_tosave=df_tosave[cols]
			
			filename2 = str(neurons1)+"_"+str(neurons2)+"_"+str(fold+1)+"_CV_RMSE.csv"
			
			if  (iter+1) == 1:
				os.chdir("/home/shikhar/USA/"+str(station_name)+"/lstm_config_selection/RMSE/")
				df_tosave.to_csv(filename2, sep=',',index=False)
				os.chdir("/home/shikhar/USA/"+str(station_name)+"/lstm_config_selection/"+directory_name)
			else:
				os.chdir("/home/shikhar/USA/"+str(station_name)+"/lstm_config_selection/RMSE/")
				with open(filename2, 'a') as f:
					(df_tosave).to_csv(f, sep=',',index=False,header=False)
				os.chdir("/home/shikhar/USA/"+str(station_name)+"/lstm_config_selection/"+directory_name)
		
		t1=time.time()
		timesave = str(int(round((t1-t0)/60,0)))
		filename="all_RMSE"+timesave+"mins_"+str(fold+1)+"CV.csv"
		
		b=modelloss
		c=modelval_loss
		
		newb = [round(item*scaler.data_range_[scaler.data_range_.shape[0]-1,],3) for item in b]
		newc = [round(item*scaler.data_range_[scaler.data_range_.shape[0]-1,],3) for item in c]
		df_epoch_list=range(1,epochs+1,1)
		df_epoch = pandas.DataFrame(df_epoch_list)
		df = pandas.DataFrame(newb)
		df2 = pandas.DataFrame(newc)
		df3 = pandas.concat([df_epoch,df, df2], axis=1)
		df3.columns = ['epoch','train','val']
		df3.to_csv(directory_name_cells+"run/"+filename, sep=',',index=False)
	
	
	
	def build_with_crossValidations (fold, train_index, val_index) :
		
		start_datetime = time.strftime('%l:%M%p %Z on %b %d, %Y')
		print ("\n\n\n - # # - Starting for:  neurons="+str(neurons1)+"_"+str(neurons2)+" at "+str(start_datetime)+" - # # - \n\n")
		t0=time.time()
		trainX, trainY, trainDate, valX, valY, valDate, testX, testY, testDate, scaler = build_data(data_parent)
		numpy.random.seed(7)
		X_train, X_val = trainX[train_index], trainX[val_index]
		y_train, y_val = trainY[train_index], trainY[val_index]
		train_and_evaluate_model(  X_train, y_train, trainDate, X_val, y_val, valX, valY, valDate, testX, testY, testDate, scaler, fold, val_index,directory_name_cells,time_Steps,neurons1, neurons2,epoch_global)
		
		t1=time.time()
		timesave = str(int(round((t1-t0)/60,0)))
		end_datetime = time.strftime('%l:%M%p %Z on %b %d, %Y')
		print ("\n\n\n - # # - Built Complete for Fold: "+str(fold+1)+" in: "+ timesave +" mins, at "+str(end_datetime)+"- # # - \n\n")
	
	
	num_cores2 = 2
	
	Parallel(n_jobs=num_cores2)(delayed(build_with_crossValidations)(fold, train_index, val_index) for fold, (train_index, val_index) in enumerate(sss))
	
	t_over_neurons = time.time()
	timesave_neurons = str(round((t_over_neurons-t_start_neurons)/3600,1))
	
	print ("\n\n\n - - # # # # - Completed for all stids for lookback of "+str(i)+" day & neurons="+str(neurons1)+"_"+str(neurons2)+"- # # # # - - \n\n")
	print ("\n\n\n - - # # # # - Took : "+ timesave_neurons+" hours - # # # # - - \n\n")


t_over = time.time()
timesave_final = str(round((t_over-t_start)/3600,1))
timesave_final_all.extend([timesave_final])
print ("\n\n\n - - # # # # - Completed for all Neurons in the list- # # # # - - \n\n")
print ("\n\n\n - - # # # # - Took : "+ timesave_final+" hours - # # # # - - \n\n")


print (timesave_final_all)



