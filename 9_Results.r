rm(list=ls())
mae <- function(error)
{
	mean(abs(error))
}


rmse <- function(error)
{
	sqrt(mean(error^2))
}


#####
##### Move Score and Prediction files to your system


user=(Sys.info()[6])
Desktop=paste("C:/Users/",user,"/Desktop/",sep="")
setwd(Desktop)

home=paste0(Desktop,"MEMS/THESIS/RESEARCH/")
setwd(home)

grnd_data = read.csv("ground_data_correctelev_eu.csv",as.is=T)
results = data.frame(stid=as.character(),ann = as.numeric(),gbm=as.numeric(),lstm=as.numeric(),stringsAsFactors = FALSE)
results_mae = data.frame(stid=as.character(),ann = as.numeric(),gbm=as.numeric(),lstm=as.numeric(),stringsAsFactors = FALSE)
for (stn in 1:nrow(grnd_data)){
	stid = grnd_data[stn,"stid"]
	setwd(paste0(home,"Europe/",grnd_data[stn,"stid"],"/data/input/"))
	train = read.csv("train.csv",as.is=T)
	train = train[train$date>"20140000",]
	train = train[train$date<"20150000",stid]
	for (algo in c("lstm","ann","gbm")){
		
		if (algo=="lstm"){
			setwd(paste0(home,"Europe/FINAL_OUTPUTS/",grnd_data[stn,"stid"],"/",algo,"/"))
			rmse_file = read.csv("regional_RMSE.csv",as.is=T)
			min_iter = rmse_file[rmse_file$val_score==min(rmse_file$val_score),"iterations"] ##the iteration which gave best score at regional level is used to check final RMSE and MAE values of location in focus
			print(paste0(stid," lstm iterations:",rmse_file[rmse_file$val_score==min(rmse_file$val_score),"iterations"]))
			mae_file = read.csv("scores/iteration_scores.csv",as.is=T)
			min_lstm_test_rmse = rmse(mae_file$actual-mae_file[,paste0("iter_",min_iter)])
			min_lstm_test_mae = mae(mae_file$actual-mae_file[,paste0("iter_",min_iter)])
		}else if(algo=="ann"){
			setwd(paste0(home,"Europe/FINAL_OUTPUTS/",grnd_data[stn,"stid"],"/",algo,"/"))
			rmse_file = read.csv("regional_RMSE.csv",as.is=T)
			min_iter = rmse_file[rmse_file$val_score==min(rmse_file$val_score),"iterations"] ##the iteration which gave best score at regional level is used to check final RMSE and MAE values of location in focus
			print(paste0(stid," ann iterations:",rmse_file[rmse_file$val_score==min(rmse_file$val_score),"iterations"]))
			mae_file = read.csv("scores/iteration_scores.csv",as.is=T)
			min_ann_test_rmse = rmse(mae_file$actual-mae_file[,paste0("iter_",min_iter)])
			min_ann_test_mae = mae(mae_file$actual-mae_file[,paste0("iter_",min_iter)])
		}else{
			setwd(paste0(home,"Europe/FINAL_OUTPUTS/",grnd_data[stn,"stid"],"/",algo,"/RMSE/"))
			rmse_file = read.csv("val_test_regional.csv",as.is=T)
			rmse_file = rmse_file[2:nrow(rmse_file),]
			min_tree = rmse_file[rmse_file$val_score==min(rmse_file$val_score),"model_treenumber"] ##the tree_number which gave best score at regional level is used to check final RMSE and MAE values of location in focus
			rmse_local_file = read.csv("val_test.csv",as.is=T)
			min_gbm_test_rmse =  rmse_local_file[rmse_local_file$model_treenumber==min_tree[1],"test_score"]
			print(paste0(stid," trees:",rmse_file[rmse_file$val_score==min(rmse_file$val_score),"model_treenumber"]))
			mae_file = read.csv(paste0(home,"USA/FINAL_OUTPUTS/",grnd_data[stn,"stid"],"/",algo,"/scores/tree_",min_tree[1],"_scores.csv"),as.is=T)
			min_gbm_test_mae = mae(train-mae_file$score)			
		}
	}
	results[stn,"stid"] <- stid
	results[stn,c(2,3,4)] <- c(min_ann_test_rmse,min_gbm_test_rmse,min_lstm_test_rmse) 
	results_mae[stn,"stid"] <- stid
	results_mae[stn,c(2,3,4)] <- c(min_ann_test_mae,min_gbm_test_mae,min_lstm_test_mae) 
}
###checking Improvement of LSTM over gbm and ffnn
results$improv_over_gbm <- (results$gbm-results$lstm)/results$gbm*100
results$improv_over_ann <- (results$ann-results$lstm)/results$ann*100
results_mae$improv_over_gbm <- (results_mae$gbm-results_mae$lstm)/results_mae$gbm*100
results_mae$improv_over_ann <- (results_mae$ann-results_mae$lstm)/results_mae$ann*100

results
results_mae



