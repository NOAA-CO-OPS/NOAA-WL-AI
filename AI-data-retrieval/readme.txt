Prerequisites
---------------
This application requires the "Pandas" Python libraries as well as 
the ConfigParser library

The Sybase Python connector only works with Python 2.6, 2.7, 3.0, and 3.1.
If you have ArcGIS installed, you can usually use the built-in Python 2.7.
The code has been written for 2.7 and uses the Sybase library at

C:\Python27\ArcGISx6410.7



Installation / Setup:
---------------------
1. Copy the entire AI-data-retrieval folder to your local machine
2. Update the file config.cfg with the appropriate database server,
   username and password

Note: predictions are created using the Java wlpred command line library
   and console application. A Java runtime (JRE) is required.


How to Run:
-----------

# Change directories to the  main directory
cd AI-data-retrieval 

# Use Python 2.7, for example from ArcGIS installation at
# C:\Python27\ArcGISx6410.7\python.exe 
 
python get_data.py --station 8443970      
python get_data.py --station 8418150      
python get_data.py --station 8534720 
python get_data.py --station 8536110  
python get_data.py --station 8557380 



