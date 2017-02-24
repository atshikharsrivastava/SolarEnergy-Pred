##############################################################
## This code is for running on linux server
##############################################################


rm(list=ls())

home = "/home/shikhar/"
setwd(home)

# grnd_data = read.csv("ground_data_stations.csv",as.is=T) #FOR USA
grnd_data = read.csv("ground_data_correctelev_eu.csv",as.is=T)

#FOR USA


country_parent_dir = "Europe" # or "USA"

dir.create(country_parent_dir)
setwd(country_parent_dir)

for (stn in 1:nrow(grnd_data)){
	dir.create(grnd_data[stn,"stid"])
	dir.create(paste0(grnd_data[stn,"stid"],"/datasets"))
	dir.create(paste0(grnd_data[stn,"stid"],"/R_build_data"))
	dir.create(paste0(grnd_data[stn,"stid"],"/lstm"))
	dir.create(paste0(grnd_data[stn,"stid"],"/gbm"))
	dir.create(paste0(grnd_data[stn,"stid"],"/ann"))
	dir.create(paste0(grnd_data[stn,"stid"],"/svr"))

}

print("NOW TRANSFER ALL THE R.DATA FILES")


##############################################################
## build training data --> SHIFT data to server
##############################################################

home = "/home/shikhar/"
setwd(home)

grnd_data = read.csv("ground_data_correctelev_eu.csv",as.is=T)



# grnd_data = grnd_data[grnd_data$country=="USA",]
country_parent_dir = "Europe"
setwd(country_parent_dir)


for (stn in 1:nrow(grnd_data)){

# stn = 3
	setwd(paste0(grnd_data[stn,"stid"]))
	source("winner_func.R")
	date_limit_upper = 20150000
	date_limit_lower = 20050000

	load("R_build_data/data.station.RData")   		###stid,nlat,elon,elev,meso_loc,stid,m,m_nlat,m_elon,m_elev,m_harv_dist,m_elev_dist
	load("R_build_data/data.station.meso.RData")	###stid,m,m_nlat,m_elon,m_elev,m_harv_dist,m_elev_dist
	load("R_build_data/data.tr.out.RData")			###date, (station names)
	load("R_build_data/data.season.RData")			###date,month,year
	load("R_build_data/data.forecast.train.RData")   ###date,gefs,nlat,elon,15x5 characterstics



	data.forecast.train=as.data.frame(data.forecast.train)

	data_forecast_all=as.data.frame(data.forecast.train)
	data_forecast_all$rn = 1:nrow(data_forecast_all)
	data_forecast_all_safe = data_forecast_all

	table(is.na(data_forecast_all[,grep("_15",colnames(data_forecast_all))]))

	
	#####Imputation of NULl values
	for (month in 1:12){
		data_forecast_month = data_forecast_all[as.integer(substring(data_forecast_all$date,5,6))==month,] ##subset MONTH wise data
		columns_fc = colnames(data_forecast_month)[grep("fc_",colnames(data_forecast_month))] 
		atm_var = unique(substring(columns_fc,4,nchar(columns_fc)-3))		### unique atmospheric variable names
		for (var in atm_var){
			req_cols = colnames(data_forecast_month)[substring(colnames(data_forecast_month),4,nchar(colnames(data_forecast_month))-3) == var]
			data_forecast_month_var = data_forecast_month[,c("rn",req_cols)] ## subset for one atmospheric variable
			data_forecast_month_var_train = data_forecast_month_var[complete.cases(data_forecast_month_var),] ## taking complete cases(no NAs) from that subset 
			for (time in c(6,9,12,15,18,21,24)){	##checking for one timestep observation
				all_vars = paste(setdiff(colnames(data_forecast_month_var_train)[-grep(time,colnames(data_forecast_month_var_train))],"rn"),collapse="+") ## formula for regression
				dv = colnames(data_forecast_month_var_train)[grep(time,colnames(data_forecast_month_var_train))]
				# print(paste0("checking for month: ",month," and dv :", dv)) 
				if (length(which(is.na(data_forecast_month_var[,dv]), arr.ind=TRUE))>0)  { ##check if any NA for this timestep in that month
					fit <- lm(as.formula(paste0(dv,"~", all_vars)), data=data_forecast_month_var_train) ## fit regression
					if (anyNA(data_forecast_month_var[,colnames(data_forecast_month_var)[-grep(time,colnames(data_forecast_month_var))]])){ ###check if the row with to-fit NA has any other NAs and replace with month mean
						cont_idvs_toimpute = colnames(data_forecast_month_var_train)[-grep(time,colnames(data_forecast_month_var_train))]
						means = colMeans(data_forecast_month_var[,cont_idvs_toimpute],na.rm=T)
						for(idv in cont_idvs_toimpute) data_forecast_month_var[is.na(data_forecast_month_var[,idv]),idv] =means[idv]
					}
					data_forecast_month_var[is.na(data_forecast_month_var[,dv]),dv]<-predict(fit, data_forecast_month_var[is.na(data_forecast_month_var[,dv]),])
					for (row_num in data_forecast_month_var$rn){
						data_forecast_all[data_forecast_all$rn == row_num,dv] <- data_forecast_month_var[data_forecast_month_var$rn == row_num,dv]
					}
					print(paste0("succesfully imputed for month: ",month," and dv :", dv)) 
				}else{
					next
				}
			}
		}
	}

	#Now Joining the GEFS data to location data
	set.seed(234)

	select.stid= as.character(data.station$stid)

	data.station=data.station[data.station$stid %in% select.stid,]




	data.all.join <- data.frame(data.station)

	data.dates <- sort(data.tr.out$Date)

	data.dates= data.dates[which(data.dates>date_limit_lower)]
	data.dates= data.dates[which(data.dates<date_limit_upper)]

	data.all.join <- data.all.join[rep(1:nrow(data.all.join), 
									   each = length(data.dates)),]
	data.all.join$date <- data.dates


	data.all.join <- merge(data.all.join, data.season,by="date")

	data.tr=data.tr.out[,c("Date",select.stid)]

	colnames(data.tr)[1]<-paste("date")

	for ( m in 1:4){
		m_locs=c(paste0("m",m,"_nlat"),paste0("m",m,"_elon"))
		data.all.join=merge(data.all.join,data_forecast_all,by.x=c("date",m_locs),by.y=c("date","nlat","elon"))

		colnames(data.all.join)[grep("fc_",colnames(data.all.join))]<-paste(gsub("fc",paste0("m",m),colnames(data.all.join)[grep("fc_",colnames(data.all.join))]))

		}

		

	#joining DV values
	data.all.join$energy<-Inf
		
	for (stid in unique(data.all.join$stid)){

		data.all.join[data.all.join$stid==stid,"energy"] <- data.tr[,as.character(stid)]
		
		}
		
	data.all.join=as.data.frame(data.all.join)


	dat=data.all.join

	dat<-dat[,setdiff(colnames(dat),c("year","meso_loc"))]
	dat=dat[,c(setdiff(colnames(dat),"energy"),"energy")]

	dat = dat[,c("date","stid","month",setdiff(colnames(dat),c("date","stid","month")))]

	setwd("datasets")
	#creating extra variable day of the year
	dat = dat[order(dat$stid,dat$date),]
	dat$day_idx <- c(1:length(unique(dat$date)))
	y_idx <- dat$day_idx/(length(unique(dat$date))/length(unique(substring(dat$date,1,4))))

	y_idx <- y_idx-floor(y_idx)
	dat$doy_idx <- y_idx
	dat$day_idx <- NULL
	cols_to_sort = c("date","stid","month","nlat","elon","doy_idx")
	dat = dat[,c(cols_to_sort,setdiff(colnames(dat),cols_to_sort))]
	dat = dat[order(dat$stid,dat$date),]
	
	# file ready for running in GBR and ANN
	write.csv(dat,"dataset_all_ml.csv",row.names=F) ##saved with harv distance
	print(paste0("Finished for ",grnd_data[stn,"stid"])) 
	setwd(home)
	setwd(country_parent_dir)

}


