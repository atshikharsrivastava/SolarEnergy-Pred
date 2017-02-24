

rm(list=ls())

user=(Sys.info()[6])
Desktop=paste("C:/Users/",user,"/Desktop/",sep="")
setwd(Desktop)

home_base=paste(Desktop,"MEMS/THESIS/RESEARCH",sep="")
setwd(home_base)

##############################################################
## load data
##############################################################

country = "Europe"#"USA"


library("ncdf4")
library("data.table")


grnd_data = read.csv("ground_data_correctelev_eu.csv",as.is=T)

for (stn in 1:nrow(grnd_data)){

	home = paste0(home_base,"/",country,"/",grnd_data[stn,"stid"])
	setwd(home)

	source(paste0(home_base,"/helper.R"))
	tic()
	cat("Loading csv data... ")
	data.station <- read.csv(fn.in.file("station_info.csv"))
	colnames(data.station)[2]<-paste0("nlat")
	colnames(data.station)[3]<-paste0("elon")
	colnames(data.station)[4]<-paste0("elev")
	data.station$stid <- factor(data.station$stid, sort(unique(data.station$stid)))

	data.station$elon <-  data.station$elon+360     ### GEFS data has only postive values going right from prime meridian and covering the globe

	### forming a square around station

	data.station$m1_nlat <- ceiling(data.station$nlat)
	data.station$m1_elon <- ceiling(data.station$elon)

	##finding distance from the grid point
	data.station$m1_harv_dist <- fn.haversine.km(
	  data.station$nlat, data.station$elon,
	  data.station$m1_nlat, data.station$m1_elon)

	data.station$m2_nlat <- ceiling(data.station$nlat)
	data.station$m2_elon <- floor(data.station$elon)

	data.station$m2_harv_dist <- fn.haversine.km(
	  data.station$nlat, data.station$elon,
	  data.station$m2_nlat, data.station$m2_elon)


	data.station$m3_nlat <- floor(data.station$nlat)
	data.station$m3_elon <- ceiling(data.station$elon)

	data.station$m3_harv_dist <- fn.haversine.km(
	  data.station$nlat, data.station$elon,
	  data.station$m3_nlat, data.station$m3_elon)


	data.station$m4_nlat <- floor(data.station$nlat)
	data.station$m4_elon <- floor(data.station$elon)

	data.station$m4_harv_dist <- fn.haversine.km(
	  data.station$nlat, data.station$elon,
	  data.station$m4_nlat, data.station$m4_elon)


	data.tr.out <- read.csv(fn.in.file("train.csv"))			### reading the training file
	colnames(data.tr.out)[1] <- "Date"
	colnames(data.tr.out)[grep("X50",colnames(data.tr.out))] <- gsub("X50","50",colnames(data.tr.out)[grep("X50",colnames(data.tr.out))])
	# data.test.out  <- read.csv(fn.in.file("sampleSubmission.csv"))
	# data.test.out[,-1] <- NA


	data.gefs <- unique(data.frame(   #### taking the unique combination of grid centers locations
	  nlat = c(data.station$m1_nlat, data.station$m2_nlat, 
			   data.station$m3_nlat, data.station$m4_nlat),
	  elon = c(data.station$m1_elon, data.station$m2_elon, 
			   data.station$m3_elon, data.station$m4_elon),
	  elev = NA))

	googEl <- function(locs)  {
	  require(RJSONIO)
	  locstring <- paste(do.call(paste, list(locs[, 1], locs[, 2], sep=',')),
						 collapse='|')
	  u <- sprintf('http://maps.googleapis.com/maps/api/elevation/json?locations=%s&sensor=false',
				   locstring)
	  res <- fromJSON(u)
	  out <- t(sapply(res[[1]], function(x) {
		c(x[['location']]['lat'], x[['location']]['lng'], 
		  x['elevation']) 
	  }))    
	  rownames(out) <- rownames(locs)
	  return(out)
	}

	data.gefs = as.data.frame(data.gefs)
	m = data.gefs[,c(1,2)]
	
	m$elon = ifelse(m$elon>180, m$elon-360, m$elon)
	
	m <- as.matrix(m)

	elev_data = as.data.frame(googEl(m))
	nlat = as.integer(round(as.numeric(elev_data$lat),0))
	elon = as.integer(round(as.numeric(elev_data$lng),0))
	elev = as.integer(round(as.numeric(elev_data$elevation),0))
	
	
	data.gefs <-data.frame(nlat=nlat,elon=elon+360, elev=elev)

	data.gefs <- data.table(data.gefs, key=c("nlat", "elon"))


	###elevation of each grid point and difference in height of station and each grid point
	data.station$m1_elev <- data.gefs[J(data.station$m1_nlat,
										data.station$m1_elon)]$elev
	data.station$m1_elev_dist <- data.station$elev - data.station$m1_elev

	data.station$m2_elev <- data.gefs[J(data.station$m2_nlat,
										data.station$m2_elon)]$elev
	data.station$m2_elev_dist <- data.station$elev - data.station$m2_elev

	data.station$m3_elev <- data.gefs[J(data.station$m3_nlat,
										data.station$m3_elon)]$elev
	data.station$m3_elev_dist <- data.station$elev - data.station$m3_elev

	data.station$m4_elev <- data.gefs[J(data.station$m4_nlat,
										data.station$m4_elon)]$elev
	data.station$m4_elev_dist <- data.station$elev - data.station$m4_elev

	### adding long,lat coordinates of each surrounding grid point
	data.station$meso_loc <- factor(paste(
	  paste(data.station$m1_nlat, data.station$m1_elon, sep=","),
	  paste(data.station$m2_nlat, data.station$m2_elon, sep=","),
	  paste(data.station$m3_nlat, data.station$m3_elon, sep=","),
	  paste(data.station$m4_nlat, data.station$m4_elon, sep=","),
	  sep = ";"))

	 ### creating dataset with stid and corresponding spatial locations of near by grids in unrolled form (m =1,2,3,4 in one columns for each stid)
	data.station.meso <- NULL
	for (m in 1:4) {
	  cols.preffix <- paste0("m", m, "_") 
	  cols.target <- c("nlat",  "elon", "elev", "harv_dist", "elev_dist")
	  cols.station <- paste0(cols.preffix, cols.target)
	  data.station.meso.cur <- data.frame(
		stid = data.station$stid,
		m = factor(m, levels=1:4))
	  data.station.meso.cur[,paste0("m_", cols.target)] <- data.station[,cols.station]
	  
	  # for (col.rm in cols.station) {
		# data.station[[col.rm]] <- NULL
	  # }
	  data.station.meso <- rbind(data.station.meso, data.station.meso.cur)
	}
	data.station.meso <- data.table(data.station.meso, key = c("stid", "m"))
	data.station <- data.table(data.station, key = c("stid"))


	###season > month and year, data.date>station name + date
	data.season <- data.table(
	  date = c(data.tr.out$Date))
	data.season$month <- format(
	  as.POSIXct(strptime(data.season$date, "%Y%m%d")), format = "%m")
	data.season$month <- factor(data.season$month)
	data.season$year <- as.integer(format(
	  as.POSIXct(strptime(data.season$date, "%Y%m%d")), format = "%m"))
	data.season$year <- data.season$year - min(data.season$year)
	setkeyv(data.season, "date")

	data.tr.dates <- data.frame(
	  date = rep(data.tr.out$Date, each=nrow(data.station)),
	  stid = rep(data.station$stid, nrow(data.tr.out))
	  )


	data.dates <- rbind(data.tr.dates)
	data.dates <- data.table(data.dates[, c("stid", "date")], key="stid")

	fn.save.data("data.station")
	fn.save.data("data.station.meso")
	fn.save.data("data.tr.out")
	fn.save.data("data.gefs")
	fn.save.data("data.season")
	fn.save.data("data.dates")

	toc()


	##############################################################
	## creating distance data amongst the stations
	##############################################################
	# source("helper.R")

	fn.load.data("data.station")

	tic()

	data.station.dist <- data.table(data.station[,"stid", with=F])
	for (stid.neigh in as.character(data.station$stid)) {
		data.station.dist[[stid.neigh]] <- Inf
	}
	for (r.cur in 1:nrow(data.station)) {
	  stid.cur <- as.character(data.station$stid[r.cur])
	  for (r.neigh in 1:nrow(data.station)) {
		stid.neigh <- as.character(data.station$stid[r.neigh])
		if (stid.cur != stid.neigh) {
		  data.station.dist[[stid.neigh]][r.cur] <- 
			fn.haversine.km(data.station$nlat[r.cur], data.station$elon[r.cur],
							data.station$nlat[r.neigh], data.station$elon[r.neigh])
		}
	  }
	}

	data.station.dist.same.loc <- data.table(data.station.dist)
	data.station.dist.diff.loc <- data.table(data.station.dist)

	same.cnt <- 0
	diff.cnt <- 0
	for (r.cur in 1:nrow(data.station)) {
	  stid.cur <- as.character(data.station$stid[r.cur])
	  loc.cur <- as.character(data.station$meso_loc[r.cur])
	  for (r.neigh in 1:nrow(data.station)) {
		stid.neigh <- as.character(data.station$stid[r.neigh])
		loc.neigh <- as.character(data.station$meso_loc[r.neigh])
		if (stid.cur != stid.neigh) {
		  if (loc.cur == loc.neigh) {
			same.cnt <- same.cnt + 1
			data.station.dist.diff.loc[[stid.neigh]][r.cur] <- Inf
		  } else {
			diff.cnt <- diff.cnt + 1
			data.station.dist.same.loc[[stid.neigh]][r.cur] <- Inf
		  }
		}
	  }
	}
	cat("Same loc", same.cnt, "diff loc", diff.cnt, "\n")

	setkeyv(data.station.dist, key(data.station))
	setkeyv(data.station.dist.same.loc, key(data.station))
	setkeyv(data.station.dist.diff.loc, key(data.station))

	fn.save.data("data.station.dist")
	fn.save.data("data.station.dist.same.loc")
	fn.save.data("data.station.dist.diff.loc")

	toc()

	##############################################################
	## loading forecast data
	##############################################################
	fn.load.data("data.station")
	fn.load.data("data.tr.out")
	fn.load.data("data.gefs")

	train.dir <- paste0(home_base,"/",country,"/GEFS_Data")  ### Download and move all GEFS files to this folder inside "Europe" or "USA"
	train.suf <- "_latlon_mean_20050101_20151231.nc"

	forecast.files <- list.files(train.dir)
	forecast.files <- gsub(train.suf, "", forecast.files)

	cat("Loading forecast data... ")

	fn.register.wk()


	forecasts.names <- foreach(fname=forecast.files,.combine=c) %dopar% { ##### creates a file day-wise,grid-wise,ensemble-wise,characterstic measures of every 5 hour (name is stored in the end)
	  
	  library("ncdf4")
	  library("data.table")
	  
	  # fn.init.worker(paste("forecast_",fname,sep=""))
	  
	  nc.tr.cur = nc_open(paste0(train.dir, "/", fname, train.suf), 
						  write=F, readunlim=T,  verbose=F)
	  # nc.test.cur = nc_open(paste0(test.dir, "/", fname, test.suf), 
						  # write=F, readunlim=T,  verbose=F)

	  gefs.name <- tolower(names(nc.tr.cur$var)[3])
	  gefs.longs <- ncvar_get(nc.tr.cur, "lon") 		# 16
	  gefs.lats <- ncvar_get(nc.tr.cur, "lat")			 # 9
	  gefs.fhours <- ncvar_get(nc.tr.cur, "fhour") 		# 5
	  # gefs.ids <- ncvar_get(nc.test.cur, "ens")		# 11
	  # nc.cur$var$Total_precipitation$varsize = 16    9    5   11 5113
	  data.nc <- NULL   ##creating null dataset

	  for (r in 1:nrow(data.gefs)){     ###for every potential grid center surrounding any meso center
		gefs.lat <- data.gefs$nlat[r]
		gefs.long <- data.gefs$elon[r]
		idx.lat <- which(gefs.lats == gefs.lat)
		idx.lon <- which(gefs.longs == gefs.long)
		
		  ### for every Ensemble ID - gefs.id
		  
		  data.nc.cur <- data.frame(   ### creating day wise dataset for one site and one ensemble
			  date = data.tr.out$Date,
			  nlat = gefs.lat,
			  elon = gefs.long)
		  
		  gefs.name.cur <- gsub("[^a-zA-Z0-9]", "_",   #### removing special characters and naming characterstic variable with hour
								paste0("fc_", gefs.name,"_",gefs.fhours))
		  
		  nc.tr.vec <- ncvar_get(nc.tr.cur)[idx.lon, idx.lat,,]   #### taking hour-wise(rows) value of characterstic everyday (columns)
		  
		  mc.mat <- matrix(nc.tr.vec,   ### transposing above
						   nrow=nrow(data.nc.cur),
						   ncol=length(gefs.fhours),
						   byrow=T)
		  
		  
		  data.nc.cur[,gefs.name.cur] <- mc.mat  ### adding to the daywise dataset
		  data.nc <- rbind(data.nc, data.nc.cur)
		
	  }
	  
	  data.nc <- data.table(data.nc, key = c("date", "nlat", "elon"))
	  cur.name <- paste0("data.forecast.", sub("\\_", ".", fname))
	  assign(cur.name, data.nc)
	  fn.save.data(cur.name)		### creates a file with different prediction of characterstic variable by each ensemble for each grid member
	  print(summary(data.nc))
	  
	  nc_close(nc.tr.cur)
	  # nc_close(nc.test.cur)

	  fn.clean.worker()
	  
	  cur.name
	}
	fn.kill.wk()
	fn.save.data("forecasts.names")

	####joining all characterstics
	forecasts.names = c("data.forecast.apcp.sfc","data.forecast.dlwrf.sfc","data.forecast.dswrf.sfc","data.forecast.pres.msl","data.forecast.pwat.eatm","data.forecast.spfh.2m","data.forecast.tcdc.eatm","data.forecast.tcolc.eatm","data.forecast.tmax.2m","data.forecast.tmin.2m","data.forecast.tmp.2m","data.forecast.ulwrf.sfc","data.forecast.ulwrf.tatm","data.forecast.uswrf.sfc")

	library("data.table")
	data.forecast.all <- NULL
	for (fc.name in forecasts.names) {
	  fn.load.data(fc.name)
	  data.forecast.cur <- get(fc.name)
	  if (is.null(data.forecast.all)) {
		data.forecast.all <- data.forecast.cur
	  } else {
		cols.key <- key(data.forecast.cur)
		if (!all(data.forecast.all[,cols.key, with=F] == 
				   data.forecast.cur[,cols.key, with=F])) {
		  stop("Error matching dimensions")
		}
		cols.cur <- setdiff(colnames(data.forecast.cur), 
							key(data.forecast.cur))
		data.forecast.all[,cols.cur] <- data.forecast.cur[,cols.cur,with=F]
	  }
	}
	data.forecast.train=data.forecast.all
	fn.save.data("data.forecast.train")

	setwd(home_base)


}





##############################################################
## build training data --> Shift data to server
##############################################################
