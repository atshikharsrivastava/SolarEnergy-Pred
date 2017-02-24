### ELEVATION DATA DOWNLOAD FOR EACH FAKE LOCATIONS

import geocoder
import pandas as pd
import numpy as np
import sys, os

os.chdir("C:\Users\Shikhar\Desktop\MEMS\THESIS\RESEARCH")
station = pd.read_csv('C:\Users\Shikhar\Desktop\MEMS\THESIS\RESEARCH\ground_data_stations_eu.csv', engine='python')
station_parent_dict = station.set_index('stid').T.to_dict('list')
all_stid = station["stid"].tolist()

for stid in all_stid:
# stid='penn_state'
 lat, lon = round(station_parent_dict[stid][1],5), round(station_parent_dict[stid][2],5)
 a = '['+str(lat)+","+str(lon)+']'
 g = geocoder.elevation(a)
 b = g.meters
 if b==None:
  b=0
 df_tosave = pd.DataFrame({'country' : station_parent_dict[stid][0],'stid' : [str(stid)], 'nlat' : [round(station_parent_dict[stid][1],2)],'elon' : [round(station_parent_dict[stid][2],2)],'elev' : [round(b,0)]})
 cols=['country','stid', 'nlat','elon' ,'elev']
 df_tosave=df_tosave[cols]
 filename1 = "ground_data_correctelev_eu.csv"
 if stid == all_stid[0] :
  df_tosave.to_csv(filename1, sep=',',index=False)
 else:
  with open(filename1, 'a') as f:
   (df_tosave).to_csv(f, sep=',',index=False,header=False)
 
 os.chdir("C:\Users\Shikhar\Desktop\MEMS\THESIS\RESEARCH" + "\\Europe\\"+stid + "\\data\\")
 fake_station = pd.read_csv('station_info_noElev.csv',engine = 'python') ### for each fake region created, the elevations are downloaded
 station_dict = fake_station.set_index('stid').T.to_dict('list')
 all_stid_local = fake_station["stid"].tolist()
 for stid_local in all_stid_local:
  lat, lon = round(station_dict[stid_local][0],5), round(station_dict[stid_local][1],5)
  c = '['+str(lat)+","+str(lon)+']'
  g = geocoder.elevation(c)
  d = g.meters
  if d==None:
   d=0
  df_tosave = pd.DataFrame({'stid' : [str(stid_local)], 'nlat' : [round(station_dict[stid_local][0],2)],'elon' : [round(station_dict[stid_local][1],2)],'elev' : [round(d,0)]})
  cols=['stid', 'nlat','elon' ,'elev']
  df_tosave=df_tosave[cols]
  filename2 = "station_info.csv"
  if stid_local == all_stid_local[0] :
   df_tosave.to_csv(filename2, sep=',',index=False)
  else:
   with open(filename2, 'a') as f:
    (df_tosave).to_csv(f, sep=',',index=False,header=False) ## file will be saved as station_info.csv
  
 os.chdir("C:\Users\Shikhar\Desktop\MEMS\THESIS\RESEARCH")

# lat, lon=47.05,12.96
