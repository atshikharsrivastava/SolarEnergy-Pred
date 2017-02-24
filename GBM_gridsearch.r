
station_name = 'penn_state'
index_fold_id = 2
home=paste0("/home/shikhar/USA/",station_name,"/")
setwd(home)
rmse <- function(error)
{
	sqrt(mean(error^2))
}

library(doSNOW)  
library(foreach)  
library(gbm)
start.time <- Sys.time()

dataset = read.csv("datasets/dataset_all_ml.csv",as.is=T)
stids = unique(dataset$stid)

dataset_todo = dataset[dataset$date>20050000,]
dataset_todo = dataset_todo[dataset_todo$date<20150000,] ##2014-2015 is test period
dataset_todo = dataset_todo[order(dataset_todo$stid,dataset_todo$date),]
dir.create("gbm_config_select")
dir.create("gbm_config_select/RMSE/")
dir.create("gbm_config_select/models/")
setwd("gbm_config_select/")


# dates = sort(unique(testdataset$date))

train_indexes = read.csv(paste0("/home/shikhar/USA/",station_name,"/dates_train_test/trainIndexes.csv"),as.is=T)
val_indexes = read.csv(paste0("/home/shikhar/USA/",station_name,"/dates_train_test/valIndexes.csv"),as.is=T)
for (i in 1:2){
	train_indexes[,i] <- train_indexes[,i]+1
	val_indexes[,i] <- val_indexes[,i]+1
}

gbm_parameters=expand.grid( shrinkage=c(0.005,0.01,0.05,0.1),minobsinnode=c(8:12),int_depth=c(8:12))

cl <- makeCluster(16) #change the 2 to your number of CPU cores  
registerDoSNOW(cl)

start.time <- Sys.time()

rmse_all = data.frame(fold = numeric(), model_treenumber = numeric(),shrinkage = numeric(),minobsinnode = numeric(), int_depth = numeric(),time_taken = numeric(),train_score = numeric(), val_score= numeric(),stringsAsFactors = FALSE)
to_add<-c(1,2,2,2,3,5,3,4)
rmse_all[1,]<-to_add
write.csv(rmse_all,paste0("RMSE/",index_fold_id,"_CV_rmse_check.csv"),row.names=F)

foreach (combination = 1:nrow(gbm_parameters) , .packages=c("gbm","tictoc")) %dopar% {
	starttime2 <- Sys.time()
	index_fold_id = index_fold_id
	dat = dataset_todo[dataset_todo$date<20120000,]
	val = dataset_todo[dataset_todo$date>20120000,]
	val = val[val$date<20140000,] ##2014 is test set
	dat_train_tocheck = dat[dat$stid==station_name,]
	val_tocheck = val[val$stid==station_name,]
	
	dat$stid <- NULL
	dat_train_tocheck$stid <- NULL
	val$stid <- NULL
	val_tocheck$stid <- NULL

	train_fold_indexes = train_indexes[,index_fold_id]
	train = dat[train_fold_indexes,]
	
	val = val
	train$month <- NULL
	dat_train_tocheck$month <- NULL

	val$month <- NULL
	val_tocheck$month <- NULL

	
	vars <- setdiff(colnames(train),c("date","energy"))
	all_vars <- paste(vars,collapse="+")
	
	shrinkage = gbm_parameters[combination,1]
	trees = 1
	minobsinnode = gbm_parameters[combination,2]
	distribution = "laplace"
	int_depth = gbm_parameters[combination,3]
	
	GBM_model = gbm( as.formula(paste("energy~", all_vars, "")), data = train , shrinkage=shrinkage, n.trees=trees, n.minobsinnode=minobsinnode ,distribution=distribution, interaction.depth=int_depth)

	for (add_trees in 1:15){ ### for each 100th tree scores will be calculated
		if (add_trees==1){
			model_treenumber = 100
			GBM_model = gbm.more(GBM_model,n.new.trees = 99)
		}else{
			model_treenumber = add_trees*100
			GBM_model = gbm.more(GBM_model,n.new.trees = 100)
		}
		options(scipen=999)
		gbm_train = predict.gbm(GBM_model,dat_train_tocheck[,-grep("date|energy",colnames(dat_train_tocheck))], n.trees=GBM_model$n.trees)
		gbm_val = predict.gbm(GBM_model,val_tocheck[,-grep("date|energy",colnames(val_tocheck))], n.trees=GBM_model$n.trees)
		options(scipen=999)
		rmse_train = rmse(dat_train_tocheck[,"energy"]-gbm_train)
		rmse_val = rmse(val_tocheck[,"energy"]-gbm_val)
		
		rmse_all = read.csv(paste0("RMSE/",index_fold_id,"_CV_rmse_check.csv"),as.is=T)
		endtime2 <- Sys.time()
		
		rmse_all = rbind(rmse_all,c(index_fold_id,model_treenumber,shrinkage,minobsinnode,int_depth,as.numeric(round(endtime2 - starttime2,3)),round(rmse_train,4),round(rmse_val,4)))
		
		print(paste0("train rmse:",rmse_train," val rmse:",rmse_val))
		write.csv(rmse_all,paste0("RMSE/",index_fold_id,"_CV_rmse_check.csv"),row.names=F)
		
	}

}


stopCluster(cl)

end.time <- Sys.time()
time.taken <- round(end.time - start.time,3)

print("GBM completed")

print(paste0("Total time taken = ",time.taken," hours"))


