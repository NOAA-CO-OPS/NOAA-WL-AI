#
# FILENAME:  get_data.py
# CREATED:   2019/06/21 Armin Pruessner
#
#
# PURPOSE:  Python 2.7 code to retrieve data from Sybase database for a given station
#           and time period. Retrieves
#
#           1. Raw A1 acoustic water level and sigma data on MSL and writes to CSV
#           2. Raw B1 backup water level and sigma data on MSL and writes to CSV
#           3. Verified water level and sigma data on MSL and writes to CSV
#           4. Tide prediction data on MSL and writes to CSV
#           5. Merges data (outer join) of all 4 data sets and writes to CSV
#           6. Retrieves audit report and writes to text file
#            
#           In order to use, the user MUST update the DB_SERVER, DB_USER and DB_PASSWORD fields. 
#           Optionally, the user can update the station ID and begin/end dates for retrieval. 
#
# DEPENDENCIES:
#           Pandas library and ConfigParser
#
#           Sybase: Suupport only for Python 2.6, 2.7, 3.0 and 3.1
#
#           1. Access to Sybase Python dll (on Windows, for example, at
#              C:\SAP\OCS-16_0\python\python27_64\dll)
#
#           2. 64 bit version of Python.  The code will not run with a 32 bit
#              version of Python. 
#
# SAMPLE CALL:
#           
#           C:\Python27\ArcGISx6410.6\python.exe get_data.py
#
# DOCUMENTATION:
#
#           http://infocenter.sybase.com/help/index.jsp?topic=/com.sybase.infocenter.dc01692.1570/doc/html/car1308847098623.html
#
# NOTE:     THIS CODE IS NOT PRODUCTION READY. PLEASE ENSURE YOU ADD
#           APPROPRIATE EXCEPTION HANDLING AND LOGGING
#
# REVISION HISTORY:
#
# 2019/08/06 Armin Pruessner
# - fixed bug not applying gain and offset to B1 WL data.
#
# 2019/07/19 Armin Pruessner
# - updated to use command line Java WL prediction library instead of retrieving predictions from database 
# - retrieve data for backup WL from both DCP 1 and DCP 2 in case we switched.
#
#

import argparse
import ConfigParser
import datetime
import os
import pandas as pd
import subprocess
import sys


# Set location of Sybase Python library
sys.path.append('C:\SAP\OCS-16_0\python\python27_64\dll')
import sybpydb


###################################################################################
### Possible parameters to update
CONFIG_FILE = "config.cfg"

# Parameters below are defaults
STATION_ID = None 
BEGIN_DATE="2001/01/01"
#BEGIN_DATE="2017/01/01"
#END_DATE ="2001/12/31"
END_DATE ="2017/12/31"

NULL_VAL = -99999.999
###################################################################################



### Utility and other functions ###################################################

""" 
Read a configuration file section (denoted as [section]) and
store in a hash {}. Returns empty hash if no parameters found
Args:
  config_file (str): configuration file to read
  section (str): name of the section to read

Returns:
  List with configuration parameters from requested section

Raises:
  None
"""
def read_config_section(config_file, section):
    params = {}
    try:
        config = ConfigParser.ConfigParser()
        with open(config_file) as f:
            config.readfp(f)
            options = config.options(section)
            for option in options:            
                try:
                    params[option] = config.get(section, option)
                    if params[option] == -1:
                        print("Could not read option: %s" % option)                    
                except:
                    print("Exception reading option %s!" % option)
                    params[option] = None
    except ConfigParser.NoSectionError as nse:
        print("No section %s found reading %s: %s", section, config_file, nse)
    except IOError, ioe:
        print("Config file not found: %s: %s", config_file, ioe)

    return params
        
        
'''
Get raw A1 acoustic water level data and write to CSV file

Args:
    station_id: station ID to retrieve data for 
    output_csv_file:  CSV filename to write data to
'''  
def get_raw_acoustic_wl_data(station_id, output_csv_file):
    # File handle, DB connection handle and cursor 
    f = None 
    conn = None
    cur = None
       
    sensor_type = "A1"        
        
    try:
        # Create a connection.
        conn = sybpydb.connect(user=DB_USER, password=DB_PASSWORD, servername=DB_SERVER);

        # Create a cursor object.    
        cur = conn.cursor()
                   
        # Determine what DB table to retrieve data from           
        data_table = "OceanData.dbo.WL_ACOUSTIC"                               
        print("Retrieving data from DB table " + data_table)        
        
        sql =  'select d.STATION_ID, \
                wa.DATE_TIME, \
                wasd.MSL, \
                wa.WL_VALUE, \
                wa.WL_VALUE-wasd.MSL as "WL_VALUE_MSL", \
                wa.WL_SIGMA \
            from OceanMD.dbo.DEPLOYMENT d, \
                    ' + data_table + ' wa, \
                    OceanData.dbo.WL_ACCEPTED_STATION_DATUM wasd \
            where \
                d.STATION_ID = "' + station_id + '" and \
                d.DCP = "1" and \
                wa.DATE_TIME >= "' + BEGIN_DATE + ' 00:00" and \
                wa.DATE_TIME <= "' + END_DATE + ' 23:59" and \
                wa.DEPLOYMENT_ID = d.DEPLOYMENT_ID and \
                wasd.STATION_ID = "' + station_id + '" and \
                wasd.ACCEPTED_DATE_TIME is  not null and \
                wasd.VERIFIED_DATE_TIME is not null and \
                wasd.EPOCH = "1983-2001" and \
                wa.DATE_TIME between wasd.ACCEPTED_DATE_TIME and isnull(wasd.SUPERSEDED_DATE_TIME,getdate()) \
            order by wa.DATE_TIME DESC'
                           
        cur.execute(sql)            
        
        rows = cur.fetchall()
        
        f = open(output_csv_file, "w")
        
        f.write("STATION_ID,DATE_TIME," + sensor_type + "_MSL," + sensor_type + "_WL_VALUE," + sensor_type + "_WL_VALUE_MSL," + sensor_type + "_WL_SIGMA\n")
        for row in rows:                  
            
            this_station_id = str(row[0])
            this_date_time = row[1].strftime("%Y-%m-%d %H:%M")
            this_msl = row[2]
            this_raw_wl = row[3]
            this_raw_wl_msl = row[4]
            this_raw_sigma = row[5]
            if this_raw_wl is None:
               this_raw_wl = NULL_VAL
            if this_raw_wl_msl is None:
               this_raw_wl_msl = NULL_VAL
            if this_raw_sigma is None:
               this_raw_sigma = NULL_VAL
                      
            f.write("%s,%s,%6.3f,%6.3f,%6.3f,%6.3f\n" % (this_station_id, this_date_time, this_msl, this_raw_wl, this_raw_wl_msl, this_raw_sigma) )  
            
    except sybpydb.Error as e:     
            print("Error retrieving data from database : {}".format(e[0]));           
            print("DB error code: {}".format(e[1]));
       
    finally:
        if cur:         
            #Close the cursor object
            cur.close() 

        if conn:   
            #Close the connection
            conn.close()
          
        if f is not None:
            # Close file handle
            f.close()    

'''
Get raw B1 backup water level data and write to CSV file

Args:
    station_id: station ID to retrieve data for 
    output_csv_file:  CSV filename to write data to
'''  
def get_raw_backup_wl_data(station_id, output_csv_file):
    # File handle, DB connection handle and cursor 
    f = None 
    conn = None
    cur = None
       
    sensor_type = "B1"        
        
    try:
        # Create a connection.
        conn = sybpydb.connect(user=DB_USER, password=DB_PASSWORD, servername=DB_SERVER);

        # Create a cursor object.    
        cur = conn.cursor()
                   
        # Determine what DB table to retrieve data from           
        data_table = "OceanData.dbo.WL_BACKUP"                               
        print("Retrieving data from DB table " + data_table)        
                
        sql =  'select d.STATION_ID, \
                wa.DATE_TIME, \
                wasd.MSL, \
                (wa.WL_VALUE*convert(float,sp1.PARAMETER_VALUE)  + convert(float,sp.PARAMETER_VALUE)) as "WL_VALUE" , \
                (wa.WL_VALUE*convert(float,sp1.PARAMETER_VALUE)  + convert(float,sp.PARAMETER_VALUE))-wasd.MSL as "WL_VALUE_MSL", \
                wa.WL_SIGMA, \
                sp.PARAMETER_VALUE as "ACC_BACKUP_OFFSET", \
                sp1.PARAMETER_VALUE as "ACC_BACKUP_GAIN" \
            from ' + data_table + ' wa \
                    join OceanMD.dbo.DEPLOYMENT d on wa.DEPLOYMENT_ID = d.DEPLOYMENT_ID \
                    join OceanData.dbo.WL_ACCEPTED_STATION_DATUM wasd on  \
                        d.STATION_ID=wasd.STATION_ID and wasd.ACCEPTED_DATE_TIME is not null and  \
                        wasd.VERIFIED_DATE_TIME is not null and wasd.EPOCH = "1983-2001" \
                    join OceanMD.dbo.SENSOR_PARAMETER sp on sp.STATION_ID=d.STATION_ID \
                        and sp.DCP=d.DCP and sp.SENSOR_ID=d.SENSOR_ID \
                    join OceanMD.dbo.PARAMETER p on p.PARAMETER_ID=sp.PARAMETER_ID \
                    join OceanMD.dbo.SENSOR_PARAMETER sp1 on sp1.STATION_ID=d.STATION_ID \
                        and sp1.DCP=d.DCP and sp1.SENSOR_ID=d.SENSOR_ID \
                    join OceanMD.dbo.PARAMETER p1 on p1.PARAMETER_ID=sp1.PARAMETER_ID\
            where \
                d.STATION_ID = "' + station_id + '" and \
                (d.DCP = "1" or d.DCP = "2") and \
                wa.DATE_TIME between "' + BEGIN_DATE + ' 00:00" and "' + END_DATE + ' 23:59" and \
                wa.DATE_TIME between d.DEPLOY_DATE_TIME and isnull(d.REMOVE_DATE_TIME, getdate()) and \
                wa.DATE_TIME between sp.BEGIN_DATE_TIME and isnull(sp.END_DATE_TIME, getdate()) and \
                wa.DATE_TIME between sp1.BEGIN_DATE_TIME and isnull(sp1.END_DATE_TIME, getdate()) and \
                p.PARAMETER_NAME = "ACC_BACKUP_OFFSET" and p1.PARAMETER_NAME="ACC_BACKUP_GAIN" and \
                wa.DATE_TIME between wasd.ACCEPTED_DATE_TIME and isnull(wasd.SUPERSEDED_DATE_TIME,getdate()) \
            order by wa.DATE_TIME DESC'
        cur.execute(sql)            
        
        rows = cur.fetchall()
        
        f = open(output_csv_file, "w")
        
        f.write("STATION_ID,DATE_TIME," + sensor_type + "_MSL," + sensor_type + "_WL_VALUE," + sensor_type + "_WL_VALUE_MSL," + sensor_type + "_WL_SIGMA,GAIN,OFFSET\n")
        for row in rows:                  
            
            this_station_id = str(row[0])
            this_date_time = row[1].strftime("%Y-%m-%d %H:%M")
            this_msl = row[2]
            this_raw_wl = row[3]
            this_raw_wl_msl = row[4]
            this_raw_sigma = row[5]
            this_gain = row[6]
            this_offset = row[7]
            if this_raw_wl is None:
               this_raw_wl = NULL_VAL
            if this_raw_wl_msl is None:
               this_raw_wl_msl = NULL_VAL
            if this_raw_sigma is None:
               this_raw_sigma = NULL_VAL
            if this_gain is None:
               this_gain = NULL_VAL
            else: 
               this_gain = float(this_gain)             
            if this_offset is None:
               this_offset = NULL_VAL  
            else:
               this_offset = float(this_offset)            
            f.write("%s,%s,%6.3f,%6.3f,%6.3f,%6.3f,%6.3f,%6.3f\n" % (this_station_id, this_date_time, this_msl, \
                this_raw_wl, this_raw_wl_msl, this_raw_sigma, this_gain, this_offset) )  
            
    except sybpydb.Error as e:     
            print("Error retrieving data from database : {}".format(e[0]));           
            print("DB error code: {}".format(e[1]));
       
    finally:
        if cur:         
            #Close the cursor object
            cur.close() 

        if conn:   
            #Close the connection
            conn.close()
          
        if f is not None:
            # Close file handle
            f.close()    
            
            
'''
Get verified water level data and write to CSV file

Args:
    station_id: station ID to retrieve data for 
    output_csv_file:  CSV filename to write data to
    
'''  
def get_verified_wl_data(station_id, output_csv_file):
    # File handle, DB connection handle and cursor 
    f = None 
    conn = None
    cur = None

    try:
        # Create a connection.
        conn = sybpydb.connect(user=DB_USER, password=DB_PASSWORD, servername=DB_SERVER);

        # Create a cursor object.    
        cur = conn.cursor()
                            
        sql =  'select wa.STATION_ID, \
                wa.DATE_TIME, \
                wasd.MSL, \
                wa.WL_VALUE, \
                wa.WL_VALUE-wasd.MSL as "WL_VALUE_MSL", \
                wa.WL_SIGMA \
            from OceanData.dbo.WL_ACCEPTED_6MIN wa, \
                    OceanData.dbo.WL_ACCEPTED_STATION_DATUM wasd \
            where \
                wa.STATION_ID = "' + station_id + '" and \
                wa.DATE_TIME >= "' + BEGIN_DATE + ' 00:00" and \
                wa.DATE_TIME <= "' + END_DATE + ' 23:59" and \
                wasd.STATION_ID = "' + station_id + '" and \
                wasd.ACCEPTED_DATE_TIME is  not null and \
                wasd.VERIFIED_DATE_TIME is not null and \
                wasd.EPOCH = "1983-2001" and \
                wa.DATE_TIME between wasd.ACCEPTED_DATE_TIME and isnull(wasd.SUPERSEDED_DATE_TIME,getdate()) \
            order by wa.DATE_TIME DESC'
                    
        cur.execute(sql)            
        
        rows = cur.fetchall()
        
        f = open(output_csv_file, "w")
        
        f.write("STATION_ID,DATE_TIME,VER_MSL,VER_WL_VALUE,VER_WL_VALUE_MSL,VER_WL_SIGMA\n")
        for row in rows:                  
            
            this_station_id = str(row[0])
            this_date_time = row[1].strftime("%Y-%m-%d %H:%M")
            this_msl = row[2]
            this_raw_wl = row[3]
            this_raw_wl_msl = row[4]
            this_raw_sigma = row[5]
            if this_raw_wl is None:
               this_raw_wl = NULL_VAL
            if this_raw_wl_msl is None:
               this_raw_wl_msl = NULL_VAL
            if this_raw_sigma is None:
               this_raw_sigma = NULL_VAL
                      
            f.write("%s,%s,%6.3f,%6.3f,%6.3f,%6.3f\n" % (this_station_id, this_date_time, this_msl, this_raw_wl, this_raw_wl_msl, this_raw_sigma) )  
            
    except sybpydb.Error as e:     
            print("Error retrieving data from database : {}".format(e[0]));           
            print("DB error code: {}".format(e[1]));
       
    finally:
        if cur:         
            #Close the cursor object
            cur.close() 

        if conn:   
            #Close the connection
            conn.close()
          
        if f is not None:
            # Close file handle
            f.close()  
            

'''
Get tide prediction data and write to CSV file

Args:
    station_id: station ID to retrieve data for 
    output_csv_file:  CSV filename to write data to
   
'''  
def get_predicted_wl_data(station_id, output_csv_file):
    # File handle, DB connection handle and cursor 
    f = None 
    conn = None
    cur = None

    try:
        # Create a connection.
        conn = sybpydb.connect(user=DB_USER, password=DB_PASSWORD, servername=DB_SERVER);

        # Create a cursor object.    
        cur = conn.cursor()
                            
        sql =  'select wa.STATION_ID, \
                wa.DATE_TIME, \
                wa.WL_VALUE-wasd.MSL as "WL_VALUE_MSL" \
            from OceanData.dbo.WL_PREDICTIONS wa, \
                    OceanData.dbo.WL_ACCEPTED_STATION_DATUM wasd \
            where \
                wa.STATION_ID = "' + station_id + '" and \
                wa.DATE_TIME >= "' + BEGIN_DATE + ' 00:00" and \
                wa.DATE_TIME <= "' + END_DATE + ' 23:59" and \
                wasd.STATION_ID = "' + station_id + '" and \
                wasd.ACCEPTED_DATE_TIME is  not null and \
                wasd.VERIFIED_DATE_TIME is not null and \
                wasd.EPOCH = "1983-2001" and \
                wa.DATE_TIME between wasd.ACCEPTED_DATE_TIME and isnull(wasd.SUPERSEDED_DATE_TIME,getdate()) \
            order by wa.DATE_TIME DESC'
                   
        cur.execute(sql)            
        
        rows = cur.fetchall()
        
        f = open(output_csv_file, "w")
        
        f.write("STATION_ID,DATE_TIME,PRED_WL_VALUE_MSL\n")
        for row in rows:                  
            
            this_station_id = str(row[0])
            this_date_time = row[1].strftime("%Y-%m-%d %H:%M")           
            this_raw_wl_msl = row[2]                    
            if this_raw_wl_msl is None:
               this_raw_wl_msl = NULL_VAL
                       
            f.write("%s,%s,%6.3f\n" % (this_station_id, this_date_time, this_raw_wl_msl) )  
            
    except sybpydb.Error as e:     
            print("Error retrieving data from database : {}".format(e[0]));           
            print("DB error code: {}".format(e[1]));
       
    finally:
        if cur:         
            #Close the cursor object
            cur.close() 

        if conn:   
            #Close the connection
            conn.close()
          
        if f is not None:
            # Close file handle
            f.close()  

            

'''
Get tide prediction data from JAR app and write to CSV file

Args:
    station_id: station ID to retrieve data for 
    output_csv_file:  CSV filename to write data to
   
'''  
def get_predicted_wl_data_cmdjar(station_id, output_csv_file):
    # File handle
    f = None 

    try:
        # Create command line call
        begin = datetime.datetime.strptime(BEGIN_DATE + " 00:00", '%Y/%m/%d %H:%M').strftime("%m/%d/%Y %H:%M") 
        end = datetime.datetime.strptime(END_DATE + " 23:59", '%Y/%m/%d %H:%M').strftime("%m/%d/%Y %H:%M")
        CMD = "java -jar WLPredCommandLineApp.jar -mgl " + station_id +  " " + begin + " " + end
        print CMD

        results= subprocess.Popen(CMD,stdout=subprocess.PIPE).communicate()[0]                     
        
        f = open(output_csv_file, "w")
       
        f.write("STATION_ID,DATE_TIME,PRED_WL_VALUE_MSL\n")
        for row in results.splitlines():                  
            
            row = row.replace("  ",",")            
            s = row.split(",")
           
            this_station_id = station_id
            this_date_time = datetime.datetime.strptime(s[0].strip(), "%m/%d/%Y %H:%M").strftime('%Y-%m-%d %H:%M')            
            this_raw_wl_msl = float(s[1])
            
            if this_raw_wl_msl is None:
               this_raw_wl_msl = NULL_VAL
                       
            f.write("%s,%s,%6.3f\n" % (station_id, this_date_time,this_raw_wl_msl) )  
            
    except Exception as e:     
            print("Error retrieving data from wlpred JAR : {}".format(e[0]));                      
       
    finally:      
          
        if f is not None:
            # Close file handle
            f.close()              
            
'''
Get audit report for specified station and date/time range and write to file

Args:
    station_id: station ID to retrieve data for 
    output_file: output text file to write audit report to
'''  
def get_audit_report(station_id, output_file):
    # File handle, DB connection handle and cursor 
    f = None 
    conn = None
    cur = None

    try:
        # Create a connection.
        conn = sybpydb.connect(user=DB_USER, password=DB_PASSWORD, servername=DB_SERVER);

        # Create a cursor object.    
        cur = conn.cursor()
        cur.execute("SET CHAINED OFF")       
        cur.callproc("OceanData.dbo.AUDIT_REPORT", (station_id, 'X', 'WL', 'W1', BEGIN_DATE , END_DATE))
         
        
        rows = cur.fetchall()
        
        f = open(output_file, "w")
               
        for row in rows:    
            s = ''.join(str(e) for e in row)      
            f.write(s + "\n")                         
            
    except sybpydb.Error as e:     
            print("Error retrieving data from database : {}".format(e[0]));           
            print("DB error code: {}".format(e[1]));
       
    finally:
        if cur:         
            #Close the cursor object
            cur.close() 

        if conn:   
            #Close the connection
            conn.close()
          
        if f is not None:
            # Close file handle
            f.close() 

            
'''
Use pandas data frame to merge raw data, predictions, and verified water level data and write to CSV file

Args:
   output_csv_file: output CSV file 

''' 
def merge_raw_verified_data(output_csv_file, raw_a1_wl_file, raw_b1_wl_file, verified_wl_file, pred_wl_file):
    # Read raw A1 acoustic water level
    raw_A1_wl_data = pd.read_csv(raw_a1_wl_file, delimiter=',')
    raw_A1_wl_df = pd.DataFrame(raw_A1_wl_data, columns=['STATION_ID','DATE_TIME','A1_MSL','A1_WL_VALUE','A1_WL_VALUE_MSL','A1_WL_SIGMA'])
    raw_A1_wl_df['DATE_TIME'] = pd.to_datetime(raw_A1_wl_df.DATE_TIME) 
    
    # Read raw B1 backup water level
    raw_B1_wl_data = pd.read_csv(raw_b1_wl_file, delimiter=',')
    raw_B1_wl_df = pd.DataFrame(raw_B1_wl_data, columns=['STATION_ID','DATE_TIME','B1_MSL','B1_WL_VALUE','B1_WL_VALUE_MSL','B1_WL_SIGMA', 'GAIN', 'OFFSET'])
    raw_B1_wl_df['DATE_TIME'] = pd.to_datetime(raw_B1_wl_df.DATE_TIME)
        
    # Read verified water level
    ver_wl_data = pd.read_csv(verified_wl_file)
    ver_wl_df = pd.DataFrame(ver_wl_data, columns=['STATION_ID','DATE_TIME','VER_MSL','VER_WL_VALUE','VER_WL_VALUE_MSL','VER_WL_SIGMA'])
    ver_wl_df['DATE_TIME'] = pd.to_datetime(ver_wl_df.DATE_TIME)   
    
    # Read tide predictions
    pred_wl_data = pd.read_csv(pred_wl_file)
    pred_wl_df = pd.DataFrame(pred_wl_data, columns=['STATION_ID','DATE_TIME','PRED_WL_VALUE_MSL'])
    pred_wl_df['DATE_TIME'] = pd.to_datetime(pred_wl_df.DATE_TIME)   
    
     
    # Merge A1 and B1 first, then with verified, and finally with predictions   
    df_merge_col_A1_B1 = pd.merge(raw_A1_wl_df, raw_B1_wl_df, on=['STATION_ID','DATE_TIME'], how='outer') 
    df_merge_col_raw_ver = pd.merge(df_merge_col_A1_B1, ver_wl_df, on=['STATION_ID','DATE_TIME'], how='outer') 
    df_merge_col = pd.merge(df_merge_col_raw_ver, pred_wl_df, on=['STATION_ID','DATE_TIME'], how='outer') 
    
    # Sort and fill nulls with default value
    df_merge_col = df_merge_col.sort_values(['STATION_ID','DATE_TIME'], ascending=True)       
    df_merge_col = df_merge_col.fillna(NULL_VAL)
    
    # Write to file
    df_merge_col.to_csv(path_or_buf=output_csv_file, index=False)

  
    
### Begin main  ################################################################################### 

# Obtain database connection info from config file
database_params = read_config_section(CONFIG_FILE, "database");
DB_SERVER = database_params['db_server']
DB_USER = database_params['db_user']
DB_PASSWORD = database_params['db_password']


# Parse (optional and required) command line arguments   
parser = argparse.ArgumentParser(
        prog='python get_data.py', 
        usage='%(prog)s --station=8443970 [options]',        
        description="Data retrieval program"
    );
parser.add_argument("-s", "--station", required=False, help="Station to retrieve data for - (required)")
parser._optionals.title = "Arguments"

# Check if arguments passed. If not print help page
if len(sys.argv) == 1:   
    parser.parse_args(['--help']);        
args = parser.parse_args();
if args.station:
    if len(args.station) == 7 and args.station.isdigit():
        STATION_ID = args.station          
    else:
        print("Invalid station specificed (%s)\n...Goodbye!" % args.station)
        sys.exit(-1)
        

# Define output file names
RAW_A1_WL_FILE = "data/" + STATION_ID + "_A1_raw_wl.csv"
RAW_B1_WL_FILE = "data/" + STATION_ID + "_B1_raw_wl.csv"
VERIFIED_WL_FILE = "data/" + STATION_ID + "_ver_wl.csv" 
PRED_WL_FILE = "data/" + STATION_ID + "_pred_wl.csv" 
MERGED_FILE = "data/" + STATION_ID + "_raw_ver_merged_wl.csv"
AUDIT_FILE = "data/" + STATION_ID + "_audit_wl.dat"
        
     
print("\nRetrieving data for {} for time period {} to {}".format(STATION_ID, BEGIN_DATE, END_DATE)) 
print("Using database {}".format(DB_SERVER))

# Check if data subdir exists and create if needed
if not (os.path.isdir("data")):
    print("Creating data/ subdirectory")
    os.mkdir("data")


       
# Retrieve data    
print("Retrieving raw A1 acoustic water level data. Writing to {}...".format(RAW_A1_WL_FILE))    
get_raw_acoustic_wl_data(STATION_ID, RAW_A1_WL_FILE)         
print("...Done")   

print("Retrieving raw B1 backup water level data. Writing to {}...".format(RAW_B1_WL_FILE))    
get_raw_backup_wl_data(STATION_ID, RAW_B1_WL_FILE)         
print("...Done")   

print("Retrieving verified water level data. Writing to {}...".format(VERIFIED_WL_FILE))    
get_verified_wl_data(STATION_ID, VERIFIED_WL_FILE)
print("...Done")   

print("Retrieving tide prediction data. Writing to {}...".format(PRED_WL_FILE))    
#get_predicted_wl_data(STATION_ID, PRED_WL_FILE)
get_predicted_wl_data_cmdjar(STATION_ID, PRED_WL_FILE)
print("...Done")   

print("Merging raw A1, B1, predicted and verified data. Writing to {}...".format(MERGED_FILE))
merge_raw_verified_data(MERGED_FILE, RAW_A1_WL_FILE, RAW_B1_WL_FILE, VERIFIED_WL_FILE, PRED_WL_FILE)
print("...Done")  

print("Retrieving audit report. Writing to {}...".format(AUDIT_FILE))
get_audit_report(STATION_ID, AUDIT_FILE)
print("...Done")

print("Removing individual data files.")
for f in [RAW_A1_WL_FILE, RAW_B1_WL_FILE, VERIFIED_WL_FILE, PRED_WL_FILE]:
    print("Removing %s..." % f)
    os.remove(f)
    print("...Done")

print("Completed data retrieval. Goodbye.");
 