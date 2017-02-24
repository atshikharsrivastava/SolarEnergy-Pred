#### Creates fake station locations using the Latitude and Longitude of chosen location

rm(list=ls())

user=(Sys.info()[6])
Desktop=paste("C:/Users/",user,"/Desktop/",sep="")
setwd(Desktop)
dir.create(paste0(home,"MEMS/"))
dir.create(paste0(home,"THESIS/"))
dir.create(paste0(home,"RESEARCH/"))
home=paste(Desktop,"MEMS/THESIS/RESEARCH/",sep="")
setwd(home)


grnd_data = read.csv("ground_data_correctelev_eu.csv",as.is=T)

for (stn in 1:nrow(grnd_data)){
	lat = grnd_data[stn,"nlat"]
	lon = grnd_data[stn,"elon"]
	lat_edge1 = ceiling(lat)
	lat_edge2 = floor(lat)
	lon_edge1 = ceiling(lon)  ###abs to take care of negatives
	lon_edge2 = floor(lon)
	dir.create(paste0(home,"Europe/"))
	dir.create(paste0(home,"Europe/",grnd_data[stn,"stid"]))
	setwd(paste0(home,"Europe/",grnd_data[stn,"stid"]))
	dir.create("data")
	dir.create("nsrdb_irr")  ###Folder where satellite irradiation will be downloaded in a later code
	dir.create("data/input")
	dir.create("data/output-R")
	setwd("data")
	##### Building 36 Uniformly distributed fake locations
	fake_lats = seq(min(lat_edge1,lat_edge2),max(lat_edge1,lat_edge2),0.2)  ####change 0.2 for increasing number of fake stations CHANGE TO 10*10 later
	fake_lats[1] = fake_lats[1] + 0.01
	fake_lats[length(fake_lats)] = fake_lats[length(fake_lats)] - 0.01

	fake_lons = seq(min(lon_edge1,lon_edge2),max(lon_edge1,lon_edge2),0.2)
	fake_lons[1] = fake_lons[1] + 0.01
	fake_lons[length(fake_lons)] = fake_lons[length(fake_lons)] - 0.01
	
 	##### Building 13 randomly distributed fake locations
	fake_locations=expand.grid(nlat=fake_lats,elon=fake_lons)
 fake_locations = rbind( fake_locations , data.frame(nlat= sample((floor(fake_lats[1])*100+1):(ceiling(fake_lats[1])*100-1),13,replace=F)/100, elon = sample((floor(fake_lons[1])*100+1):(ceiling(fake_lons[1])*100-1),13,replace=F)/100))
	fake_locations$stid = paste0(substr(grnd_data[stn,"stid"],1,3),"Fake",row.names(fake_locations))
	fake_locations = fake_locations[,c(3,1,2)]
	final_stn_file = rbind(grnd_data[stn,c("stid","nlat","elon")],fake_locations)
	write.csv(final_stn_file,"station_info_noElev.csv",row.names=F)
	setwd(home)
}

