##For Europe



rm(list=ls())

user=(Sys.info()[6])
Desktop=paste("C:/Users/",user,"/Desktop/",sep="")
setwd(Desktop)

home=paste(Desktop,"MEMS/THESIS/RESEARCH/",sep="")
setwd(home)

grnd_data = read.csv("ground_data_correctelev_eu.csv",as.is=T)

rmse <- function(error)
{
	sqrt(mean(error^2))
}
mae <- function(error)
{
	mean(abs(error))
}
scores = data.frame(stid=as.character(),RMSE=as.numeric(),MAE=as.numeric(),stringsAsFactors = FALSE) 
for (stn in 1:nrow(grnd_data)){	
	setwd(paste0(home,"Europe/",grnd_data[stn,"stid"],"/data/input/"))
	reg_dv_file = read.csv("train.csv",as.is=T)
	reg_dv_file1 = reg_dv_file[reg_dv_file$date>"20121230",]
	reg_dv_file1 = reg_dv_file1[reg_dv_file1$date<"20140000",]
	stn_dv = reg_dv_file1[,grnd_data[stn,"stid"]]
	stn_dv_persistence = stn_dv[1:(length(stn_dv)-1)] ##day before value
	stn_dv_measured = stn_dv[2:length(stn_dv)] ##present day value
	rmse_val = rmse(stn_dv_measured-stn_dv_persistence)
	mae_val = mae(stn_dv_measured-stn_dv_persistence)
	scores[stn,"stid"] <- grnd_data[stn,"stid"]
	scores[stn,c(2,3)] <- c(rmse_val,mae_val) 
}

