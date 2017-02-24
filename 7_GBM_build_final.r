##### Building final GBM model
rmse <- function(error)
{
	sqrt(mean(error^2))
}

##Load libraries
library(doSNOW)  
library(foreach)  
library(gbm)
start.time <- Sys.time()

home=paste0("/home/shikhar/")
setwd(home)

grnd_data = read.csv("ground_data_correctelev_eu.csv",as.is=T)



country_parent_dir = "Europe"
setwd(country_parent_dir)


train_begindate = 20050000
train_enddate = 20120000
val_enddate = 20140000
test_enddate = 20150000


cl <- makeCluster(16) #change to your number of CPU cores  
registerDoSNOW(cl)

start.time <- Sys.time()

### Building all models in parallel
foreach (stn = 1:16 , .packages=c("gbm")) %dopar% {
	
	station_name = grnd_data[stn,"stid"]
	home=paste0("/home/shikhar/Europe/",station_name,"/")
	setwd(home)
	dataset = read.csv("datasets/dataset_all_ml.csv",as.is=T)
	stids = unique(dataset$stid)
	
	dataset_todo = dataset[dataset$date>train_begindate,]
	dataset_todo = dataset_todo[dataset_todo$date<test_enddate,] ##2014-2015 is test period
	dataset_todo = dataset_todo[order(dataset_todo$stid,dataset_todo$date),]
	dataset_todo$month <- NULL
	rm(dataset)
	dir.create("gbm")
	dir.create("gbm/RMSE/")
	dir.create("gbm/scores/")
	setwd("gbm/")
	####Creates dataframes for score values
	rmse_all = data.frame( model_treenumber = numeric(),time_taken = numeric(), train_score = numeric(), val_score= numeric(),train_relscore = numeric(), val_relscore= numeric(),stringsAsFactors = FALSE)
	rmse_test = data.frame( model_treenumber = numeric(),time_taken = numeric(), val_score = numeric(), test_score= numeric(),val_relscore= numeric(),test_relscore= numeric(),stringsAsFactors = FALSE)
	to_add<-c(1,2,2,3,4,5)
	
	rmse_all[1,]<-to_add
	rmse_test[1,]<-to_add
	write.csv(rmse_all,"RMSE/train_val.csv",row.names=F)
	write.csv(rmse_all,"RMSE/train_val_regional.csv",row.names=F)
	write.csv(rmse_test,"RMSE/val_test.csv",row.names=F)
	write.csv(rmse_test,"RMSE/val_test_regional.csv",row.names=F)
	
	starttime2 <- Sys.time()
	train = dataset_todo[dataset_todo$date<train_enddate,]
	train_tocheck = train[train$stid==station_name,]
	
	val_test = dataset_todo[dataset_todo$date>train_enddate,]
	
	val = val_test[val_test$date<val_enddate,] ##2012-13 is val set
	val_tocheck = val[val$stid==station_name,]
	
	test = val_test[val_test$date>val_enddate,] ##2014 is test set
	test_tocheck = test[test$stid==station_name,]
	
	train$stid <- NULL
	train_tocheck$stid <- NULL
	val$stid <- NULL
	val_tocheck$stid <- NULL
	test$stid <- NULL
	test_tocheck$stid <- NULL
	
	rm(dataset_todo)
	vars <- setdiff(colnames(train),c("date","energy"))
	all_vars <- paste(vars,collapse="+")
	
	shrinkage = 0.1
	trees = 1
	minobsinnode = 11
	distribution = "laplace"
	int_depth = 11
	model_treenumber = trees
	GBM_model = gbm( as.formula(paste("energy~", all_vars, "")), data = train , shrinkage=shrinkage, n.trees=trees, n.minobsinnode=minobsinnode ,distribution=distribution, interaction.depth=int_depth)
	
	for (add_trees in 1:40){ ### for each 100th tree scores will be calculated
		starttime2 <- Sys.time()
		if (add_trees==1){
			model_treenumber = 100
			GBM_model = gbm.more(GBM_model,n.new.trees = 99)
		}else{
			model_treenumber = add_trees*100
			GBM_model = gbm.more(GBM_model,n.new.trees = 100)
		}
		options(scipen=999)
		gbm_train_regional = predict.gbm(GBM_model,train[,-grep("date|energy",colnames(train))], n.trees=GBM_model$n.trees)
		gbm_val_regional = predict.gbm(GBM_model,val[,-grep("date|energy",colnames(val))], n.trees=GBM_model$n.trees)
		gbm_test_regional = predict.gbm(GBM_model,test[,-grep("date|energy",colnames(test))], n.trees=GBM_model$n.trees)
		
		options(scipen=999)
		rmse_train_regional = rmse(train[,"energy"]-gbm_train_regional)
		rmse_val_regional = rmse(val[,"energy"]-gbm_val_regional)
		rmse_test_regional = rmse(test[,"energy"]-gbm_test_regional)
		rel_rmse_train_regional = rmse(train[,"energy"]-gbm_train_regional)/mean(train[,"energy"])*100
		rel_rmse_val_regional = rmse(val[,"energy"]-gbm_val_regional)/mean(val[,"energy"])*100
		rel_rmse_test_regional = rmse(test[,"energy"]-gbm_test_regional)/mean(test[,"energy"])*100
		
		
		
		options(scipen=999)
		gbm_train = predict.gbm(GBM_model,train_tocheck[,-grep("date|energy",colnames(train_tocheck))], n.trees=GBM_model$n.trees)
		gbm_val = predict.gbm(GBM_model,val_tocheck[,-grep("date|energy",colnames(val_tocheck))], n.trees=GBM_model$n.trees)
		gbm_test = predict.gbm(GBM_model,test_tocheck[,-grep("date|energy",colnames(test_tocheck))], n.trees=GBM_model$n.trees)
		
		##calculating error scores
		options(scipen=999)
		rmse_train = rmse(train_tocheck[,"energy"]-gbm_train)
		rmse_val = rmse(val_tocheck[,"energy"]-gbm_val)
		rmse_test = rmse(test_tocheck[,"energy"]-gbm_test)
		rel_rmse_train = rmse(train_tocheck[,"energy"]-gbm_train)/mean(train_tocheck[,"energy"])*100
		rel_rmse_val = rmse(val_tocheck[,"energy"]-gbm_val)/mean(val_tocheck[,"energy"])*100
		rel_rmse_test = rmse(test_tocheck[,"energy"]-gbm_test)/mean(test_tocheck[,"energy"])*100
		##### saving station predicion
		rmse_all = read.csv("RMSE/train_val.csv",as.is=T)
		rmse_test_save = read.csv("RMSE/val_test.csv",as.is=T)
		endtime2 <- Sys.time()
		
		rmse_all = rbind(rmse_all,c(model_treenumber,as.numeric(round(endtime2 - starttime2,3)),round(rmse_train,4),round(rmse_val,4),round(rel_rmse_train,4),round(rel_rmse_val,4)))
		rmse_test_save = rbind(rmse_test_save,c(model_treenumber,as.numeric(round(endtime2 - starttime2,3)),round(rmse_val,4),round(rmse_test,4),round(rel_rmse_val,4),round(rel_rmse_test,4)))
		
		write.csv(rmse_all,"RMSE/train_val.csv",row.names=F)
		write.csv(rmse_test_save,"RMSE/val_test.csv",row.names=F)
		score_file = data.frame(date=test_tocheck$date,score=gbm_test)
		write.csv(score_file,paste0("scores/tree_",model_treenumber,"_scores.csv"),row.names=F)
		val_score_file = data.frame(date=val_tocheck$date,score=gbm_val)
		write.csv(val_score_file,paste0("scores/tree_",model_treenumber,"_scores_val.csv"),row.names=F)
		####saving regional predicion
		rmse_all_regional = read.csv("RMSE/train_val_regional.csv",as.is=T)
		rmse_test_regional_save = read.csv("RMSE/val_test_regional.csv",as.is=T)	
		
		rmse_all_regional = rbind(rmse_all_regional,c(model_treenumber,as.numeric(round(endtime2 - starttime2,3)),round(rmse_train_regional,4),round(rmse_val_regional,4),round(rel_rmse_train_regional,4),round(rel_rmse_val_regional,4)))
		rmse_test_regional_save = rbind(rmse_test_regional_save,c(model_treenumber,as.numeric(round(endtime2 - starttime2,3)),round(rmse_val_regional,4),round(rmse_test_regional,4),round(rel_rmse_val_regional,4),round(rel_rmse_test_regional,4)))
		
		write.csv(rmse_all_regional,"RMSE/train_val_regional.csv",row.names=F)
		write.csv(rmse_test_regional_save,"RMSE/val_test_regional.csv",row.names=F)		
	}

}


stopCluster(cl)

end.time <- Sys.time()
time.taken <- round(end.time - start.time,3)

print("GBM completed")

print(paste0("Total time taken = ",time.taken," hours"))


