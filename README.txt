<<For Github viewers: Read this file in raw format>>

<The following instructions are for the codes which would reproduce results performed in my Masters Thesis Experiment of Testing LSTM for Solar Radiation prediction>

### Windows was used for - 
Initialization: 
-- Create a home folder C:/Users/~SYSTEM_NAME~/Desktop/MEMS/THESIS/RESEARCH/  ##change ~SYSTEM_NAME~
-- The two CSVs named "ground_data_correctelev_eu" and "ground_data_stations" contains data for chosen locations of Europe and the US respectively.
-- Put these files in the home folder
-- The provided codes are tuned for Europe based modeling. For running them on US datasets, a simple find(1.Europe,2.ground_data_correctelev_eu) and replace(1.USA,2.ground_data_stations) would suffice. The parameter grid_search files don't need this as "Penn State" location from US has been hardcoded in them.
-- In all files the drive locations contains my system name. Please find  "shikhar" and replace to your PC's name.


Download GEFS data from https://esrl.noaa.gov/psd/forecasts/reforecast2/download.html
For one of the atmospheric variables as have been mentioned in the paper
	- Select this variable under "Select Desired Variables and Associated Levels" with tab value "Single Level 1x1 degree"
	- Select period of 2005 to 2015
	- Select 6, 9, 12, 15, 18, 21 and 24 in "Desired Forecast Hour(s)" option.
	- Select Ensemble Mean option from "Select Ensemble Information" option.
	- Choose Continental US or Europe from Geographical Location.
	- Select netCDF4 and provide your email
	- You will receive an email with download link when your query is processed. Download the file and rename the suffix of file to 	  	  
			"_latlon_mean_20050101_20151231.nc".

	- Place each of these files in C:/Users/~SYSTEM_NAME~/Desktop/MEMS/THESIS/RESEARCH/~(Europe or USA)~/GEFS_Data/nsrdb_irr/
	- Repeat this for all the atmospheric variables for both US and Europe one by one.

A helper file for R should be placed in folder C:/Users/~SYSTEM_NAME~/Desktop/MEMS/THESIS/RESEARCH/ . The file named "helper.R", helps in extracting data from netCDF4 files downloaded from above.


Steps to follow for building fake regions and running the experiment. The codes are in sequential order numbering.
1. First code will create a CSV list of the 1x1 degree region around chosen locations with fake locations of which 36 are uniformly distributed and 13 are randomly. Thus, including the original location, the CSV would contain 50 location's latitude and longitudes.
2. 2_get_elev.py needs to be run in python to get elevation data from Google's API
3. Downloading Satellite based daily irradiation data 

	For US: 
	Using file download_noaa_irradiation.py in python, one can download historical datasets for all the 50 locations in every fake region above created
	A login account and key needs to be generated from https://developer.nrel.gov/signup/ for API based download. Hourly resolution level radiation data for the period of 2005-2014 should be selected in options for each station (fake and real). Download these files in : 						

C:/Users/~SYSTEM_NAME~/Desktop/MEMS/THESIS/RESEARCH/USA/~location_name~/nsrdb_irr/

	For Europe:
	An account on http://www.soda-pro.com/web-services/radiation/cams-radiation-service needs to be created to download each location's radiation data. The service only allows 15 downloads a day per account. For one fake region built we would need to download 50 times. Hourly resolution level radiation data for the period of 2005-2014 should be selected in options for each station (fake and real). Download these files in : 								

C:/Users/~SYSTEM_NAME~/Desktop/MEMS/THESIS/RESEARCH/Europe/~location_name~/nsrdb_irr/


4. When all satellite based data every station of each region is downloaded run 3_create_dv_file_eu.R . This build a file with daily level target values of each station(fake or real) for 2005-2014
5. When these files are created for all the locations (chosen in Europe) run 4_my_build.R . It will generate R data files in  C:/Users/~SYSTEM_NAME~/Desktop/MEMS/THESIS/RESEARCH/~location~/data/output-R folder.

##Transfer to Linux Server
6. Transfer these files to system with more number of cores (I was provided access to 48 cores).
7. The subsequent file 4_my_build_atserver.R builds folders and pre-processes data (handling of null values).
8. Run 5_time_seriesBuild.R for creating input dataset for LSTMs (design mentioned in the paper)
9. Grid search is performed for LSTM, FFNN and GBR using LSTM_gridsearch.py, ANN_gridsearch.py and GBM_gridsearch.r.  Please note that 2-fold stratified (on target variable) was performed and the row indexes are printed using printing_bagged_indexes.py function. GBM models can only be built after this file is compiled.This is a time taking process which can be skipped as final files have best parametric combinations hard-coded in them.
10. 6_LSTM_finalModels.py, 7_GBM_build_final.R and 8_ANN_build_final.py takes the best configuration found from LSTM, GBR and FFNN based grid search on one location of the US and builds the models.

### Moving outputs back to local windows systems
11. Move predictions and score files to the system. 9_Results.R applies the selection criteria and provides best MAE and RMSE scores for each location
12. 10_persistence_scores.R gives MAE and RMSE of persistence model, for each location


### The initial results of testing on AMS data
-- For AMS competition results the 4_my_build.R, 4_my_build_atserver.R and 5_time_seriesBuild.R would be required to obtain initial dataset. LSTM_gridsearch.py would have to be then applied for finding optimal tuning. Finally, 6_LSTM_finalModels.py would need to be run using these optimal parametric combinations. The codes will require to adjust train and test dates accordingly as mentioned in the paper. For per-station modeling, the models would need to be built for each solar station using for-loops. 
-- For reproducibility, gridsearch step can be ignored and following parametric combinations can be used in given model skeleton to obtain same results.
LSTM -> {1st layer = 110 neurons
		2nd layer = 100 neurons 
		dropout  = 0.2. 
		Inner Activations = ReLu
		Batchsize = 128 
		adam optimizer learning rate = 0.01}
GBR ->	{shrinkage = 0.05
		trees = 3000
		minobsinnode = 10
		distribution = "laplace"
		int_depth = 10}