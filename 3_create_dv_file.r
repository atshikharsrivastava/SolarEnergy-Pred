

rm(list=ls())

user=(Sys.info()[6])
Desktop=paste("C:/Users/",user,"/Desktop/",sep="")
setwd(Desktop)

home=paste(Desktop,"MEMS/THESIS/RESEARCH/",sep="")
setwd(home)

grnd_data = read.csv("ground_data_correctelev_eu.csv",as.is=T)

for (stn in 1:nrow(grnd_data)){
	setwd(paste0(home,"Europe/",grnd_data[stn,"stid"],"/data/input/"))
	stn_file = read.csv("station_info.csv",as.is=T)
	stids = sort(as.character(stn_file$stid))
	setwd(paste0(home,"Europe/",grnd_data[stn,"stid"],"/nsrdb_irr/"))

	for (stid in stids){
		sat_values <- list.files(pattern=paste0(stid,".csv"))
		ghi_file = read.csv(sat_values[1],as.is=T)
		ghi_file$date <- paste0(ghi_file$Year,ifelse(nchar(ghi_file$Month)==1,paste0(0,ghi_file$Month),ghi_file$Month),ifelse(nchar(ghi_file$Day)==1,paste0(0,ghi_file$Day),ghi_file$Day))
		ghi_subs <- ghi_file[,c("date","GHI")]
		ghi_subs$ghi_joules = ghi_subs$GHI*3600
		ghi_subs2 <- aggregate(ghi_subs[,3],list(date=ghi_subs$date),FUN=sum)
		ghi_subs2$ghi = ghi_subs2$x/24/3600   #### converting hourly readings into daily
		ghi_subs2[,2] <- NULL
		colnames(ghi_subs2)[2] <- stid
		if (stid == stids[1]){
			ghi_all = ghi_subs2
		} else {
			ghi_all = merge(ghi_all,ghi_subs2,by="date")
		}
		
		ghi_all = ghi_all[order(ghi_all$date),]
	}
	dir.create(paste0(home,"Europe/",grnd_data[stn,"stid"],"/data/input/"))
	setwd(paste0(home,"Europe/",grnd_data[stn,"stid"],"/data/input/"))
	write.csv(ghi_all,"train.csv",row.names=F)
	setwd(home)
}


