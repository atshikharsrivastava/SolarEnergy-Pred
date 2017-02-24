### US based radiations are downloaded from web api service. A login id and api_key is required. One would need to generate them to use the code below. the website address is https://developer.nrel.gov/signup/

import pandas as pd
import numpy as np
import sys, os


station_names = ['bondville','desert_rock','fort_peck','goodwin_creek','penn_state']

for station_orig in station_names:
 station = pd.read_csv('C:\Users\Shikhar\Desktop\MEMS\THESIS\RESEARCH\USA\\'+station_orig+'\data\station_info.csv', engine='python')
 station_dict = station.set_index('stid').T.to_dict('list')
 all_stid = station["stid"].tolist()
 select_years = range(2005,2015,1)
 for stid in all_stid:
  print(" For stid: "+str(stid))
  for year_select in select_years:
   lat, lon, year = round(station_dict[stid][0],5), round(station_dict[stid][1],5), year_select
   print("year:"+str(year_select))
   # You must request an NSRDB api key from the link above
  
   api_key = #paste your key here as string
   
   # Set the attributes to extract (e.g., dhi, ghi, etc.), separated by commas.
   attributes = 'ghi,dhi,dni'
   # Choose year of data
   year = str(year_select)
   # Set leap year to true or false. True will return leap day data if present, false will not.
   if year_select%4 == 0:
    leap_year = 'true'
   else:
    leap_year = 'false'
   # Set time interval in minutes, i.e., '30' is half hour intervals. Valid intervals are 30 & 60.
   interval = '60'
   # Specify Coordinated Universal Time (UTC), 'true' will use UTC, 'false' will use the local time zone of the data.
   # NOTE: In order to use the NSRDB data in SAM, you must specify UTC as 'false'. SAM requires the data to be in the
   # local time zone.
   utc = 'true'
   # Your full name, use '+' instead of spaces.
   your_name = 'Shikhar+Srivastava'
   # Your reason for using the NSRDB.
   reason_for_use = 'energy-research'
   # Your affiliation
   your_affiliation = 'Humboldt-University-Berlin'
   # Your email address
   your_email = #put your email id here
   # Please join our mailing list so we can keep you up-to-date on new developments.
   mailing_list = 'false'
   
   # Declare url string
   url = 'http://developer.nrel.gov/api/solar/nsrdb_0512_download.csv?wkt=POINT({lon}%20{lat})&names={year}&leap_day={leap}&interval={interval}&utc={utc}&full_name={name}&email={email}&affiliation={affiliation}&mailing_list={mailing_list}&reason={reason}&api_key={api}&attributes={attr}'.format(year=year, lat=lat, lon=lon, leap=leap_year, interval=interval, utc=utc, name=your_name, email=your_email, mailing_list=mailing_list, affiliation=your_affiliation, reason=reason_for_use, api=api_key, attr=attributes)
   # Return just the first 2 lines to get metadata:
   # info = pd.read_csv(url, nrows=1)
   # See metadata for specified properties, e.g., timezone and elevation
   # timezone, elevation = info['Local Time Zone'], info['Elevation']
   
   df = pd.read_csv('http://developer.nrel.gov/api/solar/nsrdb_0512_download.csv?wkt=POINT({lon}%20{lat})&names={year}&leap_day={leap}&interval={interval}&utc={utc}&full_name={name}&email={email}&affiliation={affiliation}&mailing_list={mailing_list}&reason={reason}&api_key={api}&attributes={attr}'.format(year=year, lat=lat, lon=lon, leap=leap_year, interval=interval, utc=utc, name=your_name, email=your_email, mailing_list=mailing_list, affiliation=your_affiliation, reason=reason_for_use, api=api_key, attr=attributes), skiprows=2)
   
   # Set the time index in the pandas dataframe:
   
   
   # take a look
   print 'shape:',df.shape
   df.head()
   
   print df.columns.values
   os.chdir('C:\Users\Shikhar\Desktop\MEMS\THESIS\RESEARCH\USA\\'+station_orig+'\\nsrdb_irr')
   file_name = str(stid)+"_"+str(year)+".csv"
   df.to_csv(file_name, sep=',',index=False)



