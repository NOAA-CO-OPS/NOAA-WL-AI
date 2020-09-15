#!python37

## By Elim Thompson (09/05/2020)
##
## This script uses data_cleaner class to perform cleaning procedure for Greg's
## WL-AI project. Prior executing this script, one must do the following steps.
##
## 1. Download the latest WL-AI Station List from ..
##    https://docs.google.com/spreadsheets/d/1tLoaNPWNnHneWOZlpS38S7ldlkSs0wiCq_E6_39u_Qg/edit?usp=sharing
##    The file must be stored as CSV file. This will be parsed into this script
##    via station_info_csv flag.
## 2. Copy Armin's raw file from CO-OPS Common ..
##    N:\CO-OPS_Common\CODE\AI-data-retrieval\data
##    to your local desktop. Unzip all files to a location. This location will
##    be parsed to this script via raw_path flag.
##
## Only when the above steps are completed, you can then execute this script:
## > python clean_data.py --raw_path <raw file location> 
##                        --proc_path <where you want to store cleaned data>
##                        --station_info_csv <location of station info csv>
##                        --log_level <debug/info/warn/error>
##                        (--do_midstep_files)
###############################################################################

###############################################
## Import libraries
###############################################
import logging, argparse, os
import data_cleaner

###############################################
## Define constants
###############################################
# Default log level
log_level = 'info'

# Default location of Armin's raw data
raw_path = 'C:/Users/elim.thompson/Documents/wl_ai/data/raw/'

# Default location of processed, cleaned data
proc_path = 'C:/Users/elim.thompson/Documents/wl_ai/data/processed/'

# Default location of WL-AI Station list csv file
station_info_csv = 'C:/Users/elim.thompson/Documents/wl_ai/data/WLAIStationList.csv'

# Ask cleaner to create mid-step files & plots
do_midstep_files = False

###############################################
## Define functions
###############################################
def get_parser ():

    ''' A function to handle user inputs via command line. For raw_path and
        station_info_csv file, they must exist. Otherwise, FileNotFoundError is
        raised. For proc_path, if the folder does not exist, this function will
        create it. For log level, only info, debug, warn, and error are accepted.

        return params
        -------------
        raw_path (str): Path to Armins unzipped raw files
        proc_path (str): Path to store processed, cleaned files
        station_info_csv (str): Location of station info sheet
        log_level (str): either info, debug, warn, or error
        create_midstep_files (bool): If true, create all mid-step files / plots
    '''

    ## Define parser to get arguments
    parser = argparse.ArgumentParser (description='')
    parser.add_argument('-r', '--raw_path', default=raw_path, type=str,
                        help='Path to Armins unzipped raw files')
    parser.add_argument('-p', '--proc_path', default=proc_path, type=str,
                        help='Path to store processed, cleaned files')
    parser.add_argument('-s', '--station_info_csv', default=station_info_csv,
                        type=str, help='Location of station info sheet')
    parser.add_argument('-l', '--log_level', default=log_level, type=str,
                        help='Log level: info, debug, warn, error')
    parser.add_argument('-m', '--do_midstep_files', default=do_midstep_files,
                        action='store_true',
                        help='If turned on, create all mid-step files.')
    args = parser.parse_args()

    ## 1. Check if raw path exists. If not, raise exception.
    if not os.path.exists (args.raw_path):
        message = 'Raw folder, {0}, does not exist!'.format (args.raw_path)
        raise FileNotFoundError (message)

    ## 2. Check if station info sheet exists
    if not os.path.exists (args.station_info_csv):
        message = 'Station info sheet, {0}, does not exist!'.format (args.station_info_csv)
        raise FileNotFoundError (message)

    ## 3. Check if proc path exists. If not, create it now.
    if not os.path.exists (args.proc_path):
        os.mkdir (args.proc_path)

    ## 4. Check if log level is one of info / debug / warn / error
    if not args.log_level.lower() in ['debug', 'info', 'warn', 'error']:
        message = 'Log level must be either debug, info, warn, or error.'
        raise IOError (message)

    return args.raw_path, args.proc_path, args.station_info_csv, \
           args.log_level.upper(), args.do_midstep_files

def print_summary_stats (train, valid, test):

    ''' A function to print nspike percentages on console. For training, the
        percentage is w.r.t. 200000. For validation and testing, they are w.r.t.
        the total number of records in the sets.

        input params
        ------------
        train (pandas.DataFrame): dataframe with training set stats
        valid (pandas.DataFrame): dataframe with validation set stats
        test  (pandas.DataFrame): dataframe with testing set stats
    '''

    ## Print header
    print ('+-{0}-+-{0}-+-{0}-+-{0}-+'.format ('-'*7))
    print ('| {0} | {1} | {2} | {3} |'.format ('station', ' train ', ' valid ', ' test  '))
    print ('+-{0}-+-{0}-+-{0}-+-{0}-+'.format ('-'*7))

    ## Loop through each station and calculate percentage
    lineFmt = '| {0} | {1:.5f} | {2:.5f} | {3:.5f} |'
    for station_id in train.station_id.sort_values():
        # calculate percentage in training set
        nspikes = train[train.station_id==station_id].n_spikes.values[0]
        train_pct = nspikes / 200000.
        # calculate percentage in validation set
        nspikes = valid[valid.station_id==station_id].n_spikes.values[0]
        ntotal  = valid[valid.station_id==station_id].n_total.values[0]
        valid_pct = nspikes / ntotal
        # calculate percentage in training set
        nspikes = test[test.station_id==station_id].n_spikes.values[0]
        ntotal  = test[test.station_id==station_id].n_total.values[0]
        test_pct = 0 if ntotal == 0 else nspikes / ntotal
        # print on console
        print (lineFmt.format (station_id, train_pct, valid_pct, test_pct))
        print ('+-{0}-+-{0}-+-{0}-+-{0}-+'.format ('-'*7))

###############################################
## Script begins here!
###############################################
if __name__ == '__main__':

    ## Get user arguments
    raw_path, proc_path, station_info_csv, log_level, do_midstep_files = get_parser ()

    ## Set log level
    level = getattr (logging, log_level)
    logging.basicConfig (level=level)

    ## Initialize a new data cleaner 
    cleaner = data_cleaner.data_cleaner()

    ## Set up the cleaner using input arguments
    cleaner.raw_path = raw_path
    cleaner.proc_path = proc_path
    cleaner.station_info_csv = station_info_csv
    cleaner.create_midstep_files = do_midstep_files

    ## Load station info
    cleaner.load_station_info()
    
    ## Clean all stations
    #  1. Default way: include nan VER_WL_VALUE_MSL in counting spikes
    cleaner.clean_stations (exclude_nan_verified=False, station_ids=[8443970])
    #  2. EXCLUDE nan VER_WL_VALUE_MSL in counting spikes
    #cleaner.clean_stations (exclude_nan_verified=True)

    ## Clean subset of stations. This will clean both the requested stations
    ## and their neighbor stations.
    #station_ids = [8418150, 8454049]
    #cleaner.clean_stations (exclude_nan_verified=False, station_ids=station_ids)

    ## Save stats data (if not already) and print out a summary
    cleaner.save_stats_data()
    print_summary_stats (cleaner.train_stats, cleaner.validation_stats,
                         cleaner.test_stats)
