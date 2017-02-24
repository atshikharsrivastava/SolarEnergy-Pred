#### This code will realign data in time series format to use as input for LSTMs
rm(list=ls())

home = "/home/shikhar/"
setwd(home)

grnd_data = read.csv("ground_data_correctelev_eu.csv",as.is=T)

#FOR USA

# grnd_data = grnd_data[grnd_data$country=="USA",]
country_parent_dir = "Europe"
setwd(country_parent_dir)


for (stn in  1:nrow(grnd_data)){

		setwd(paste0(grnd_data[stn,"stid"],"/datasets"))
		dataset=read.csv("dataset_all_ml.csv",as.is=T)
		dataset<-dataset[order(dataset$stid,dataset$date),,]


	stid_data=as.data.frame(sort(unique(dataset$stid)))

	colnames(stid_data)[1]<-paste0("stid")
	stid_data$stid_num<-row.names(stid_data)


	library(doSNOW)  
	library(foreach)
	cl<-makeCluster(15) #change the 2 to your number of CPU cores  
	registerDoSNOW(cl)




	data_per_ob_all <- foreach (stid_num=1:nrow(stid_data),.combine=rbind) %dopar% {
		dat=dataset[dataset$stid==stid_data[stid_num,'stid'],]
		dat$rn<-row.names(dat)
		data_per_ob <- NULL
		##Each variables timesteps which were columns and turned into sequential rows
		for (hr in c(6,9,12,15,18,21,24)) {
		  cols.suffix <- paste0("_", hr)
		  cols.target<-c()
		  for (m in 1:4){
			cols.prefix<- paste0("m", m,"_")
			cols.target.base <- c('downward_long_wave_rad_flux','downward_short_wave_rad_flux','maximum_temperature','minimum_temperature','precipitable_water','pressure','specific_humidity_height_above_ground','temperature_height_above_ground','total_cloud_cover','total_column_integrated_condensate','total_precipitation','upward_long_wave_rad_flux','upward_long_wave_rad_flux_surface','upward_short_wave_rad_flux')
			cols.target.base <- paste0( cols.prefix,cols.target.base)
			cols.target<-c(cols.target,cols.target.base)
			}
		  cols.vars <- paste0( cols.target,cols.suffix)
		  data_per_ob.cur <- data.frame(
			rn = dat$rn,
			hr = factor(hr, levels=c(6,9,12,15,18,21,24)))
		  data_per_ob.cur[,paste0( cols.target)] <- dat[,cols.vars]
		  
		  # for (col.rm in cols.station) {
			# data.station[[col.rm]] <- NULL
		  # }
		  data_per_ob <- rbind(data_per_ob, data_per_ob.cur)
		}
		data_per_ob$rn<-as.numeric(as.character(data_per_ob$rn))
		data_per_ob$hr<-as.numeric(as.character(data_per_ob$hr))

		###Making sure that the order of timesteps is sequential
		data_per_ob=data_per_ob[ order(data_per_ob[,1], data_per_ob[,4]), ]

		dat$rn<-as.numeric(as.character(dat$rn))

		data_per_ob= merge(data_per_ob,dat[,c("rn","date","month","stid","doy_idx","energy",colnames(dat)[grep("nlat|elon|elev|harv",colnames(dat))])],by="rn")

		data_per_ob= data_per_ob[,c("rn","date","month","hr",'stid',"doy_idx",colnames(data_per_ob)[grep("nlat|elon|elev|harv",colnames(data_per_ob))],setdiff(colnames(data_per_ob),c("rn","hr","energy",colnames(data_per_ob)[grep("nlat|elon|elev|harv",colnames(data_per_ob))])),"energy")]
		
		data_per_ob=data_per_ob[ order(data_per_ob[,1], data_per_ob[,2]), ]

		data_per_ob$rn<-NULL
		data_per_ob=data_per_ob[ order(data_per_ob[,"date"], data_per_ob[,"stid"],data_per_ob[,"hr"]), ]
		data_per_ob$hr<-NULL
		data_per_ob
	}


	stopCluster(cl)
	data_per_ob_all$month.1<-NULL
	data_per_ob_all$doy_idx.1<-NULL
	data_per_ob_all$stid.1<-NULL
	data_per_ob_all$date.1<-NULL
	data_per_ob_all$month.1<-NULL


	## data is now ready for LSTM modeling
	write.csv(data_per_ob_all,"dataset_all_ts.csv",row.names=F)
	setwd(home)
	setwd(country_parent_dir)
}

