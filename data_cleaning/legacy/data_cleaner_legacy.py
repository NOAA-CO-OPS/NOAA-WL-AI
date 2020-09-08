#!C:/Users/elim.thompson/AppData/Local/Programs/Python/Python37/python

## By Elim Thompson (04/26/2020)
##
## This script is developed based on Greg Dusek's clean.ipynb. Additional requirements
## are included based on his documentation at
##
## https://docs.google.com/document/d/1BfyIQE9GXPCRbBSkyurd3UeGqpGkAr1UYkMZzh5LBNk/edit?usp=sharing
##
## Data pipeline before AI modeling is divided into 2 parts.
##  1. Raw data acquisition (Armin)
##      * Retrieval of the observational values (columns 1-21) and basic data cleanup steps 1-3.
##      * Write to CSV as "raw" data
##  2. Date Clean-up
##      * Calculate residuals, and apply "business" rules as outlined in data cleanup steps 4-8
##      * Write to new CSV as "processed" data 
##
## Processed pickled file is used for ML/AI training, whereas csv files can be used for data
## visualization in MatLab if needed.
######################################################################################################

###############################################
## Import libraries
###############################################
import numpy, pandas, datetime, os
import matplotlib.pyplot as plt
from glob import glob
import _pickle as pickle

###############################################
## Define constants
###############################################
dataPath      = 'C:/Users/elim.thompson/Documents/wl_ai/data/'
stationFile   = dataPath + 'WLAIStationList.csv'
rawCSVPath    = dataPath + 'raw/'
processedPath = dataPath + 'processed/'

# Final columns to be included 
out_columns = ['STATION_ID', 'DATE_TIME', 'SENSOR_USED_PRIMARY', 'PRIMARY', 'PRIMARY_TRUE',
               'PRIMARY_SIGMA', 'PRIMARY_SIGMA_TRUE', 'PRIMARY_RESIDUAL', 'BACKUP',
               'BACKUP_TRUE', 'BACKUP_SIGMA', 'BACKUP_SIGMA_TRUE', 'BACKUP_RESIDUAL',
               'PREDICTION', 'VERIFIED', 'TARGET', 'OFFSETS_APPLIED']
neighbor_keys = ['NEIGHBOR_PRIMARY', 'NEIGHBOR_PREDICTION', 'NEIGHBOR_PRIMARY_RESIDUAL', 'NEIGHBOR_TARGET']

# Meta data frame from WLAIstationFile.csv. This must follow the ordering of columns in stationFile.
station_columns = ['STATION_NAME', 'STATION_ID', 'GT_RANGE', 'PROBLEM', 'EXPLANATION', 'WL_MIN', 'WL_MAX',
                   'REGION', 'DATES_DOWNLOADED',
                   'DATES_FOR_TRAINING', 'DATES_FOR_VALIDATION', 'DATES_FOR_TESTING', 'PRIMARY_SENSOR_TYPE',
                   'PRIMARY_SENSOR_DCP', 'PRIMARY_SENSOR_DATES', 'OTHER_PRIMARY_SENSOR_TYPE', 'OTHER_PRIMARY_SENSOR_DCP',
                   'OTHER_PRIMARY_SENSOR_DATES', 'NEIGHBOR_STATION_NAME', 'NEIGHBOR_STATION_ID', 'NOTES', 'EPOCH']

# Extract all three dataset types if their time periods are available
dataset_types = ['train', 'validation', 'test']

# Information related to sensors
sensor_names = ['PRIMARY_SENSOR_TYPE', 'OTHER_PRIMARY_SENSOR_TYPE']
valid_sensor_types = ['A1', 'B1', 'Y1', 'NT', 'N1', 'T1']

# Dates for training / testing / validation
train_end_date = "2016-12-31"
valid_start_date, valid_end_date = "2017-01-01", "2018-12-31"
test_start_date = "2019-01-01"

# TARGET threshold in meters between PRIMARY and VERIFIED
target_thresh = 0.02

# Default caps for PRIMARY and BACKUP if no WL_MIN and WL_MAX available in station csv.
WL_default_caps = 5 

# Turn on / off console display. Will be changed to argparse.
verbose = True

###############################################
## Define functions
###############################################
# Short function to pull all station IDs from available raw CSV files.
get_stations = lambda fileList: numpy.array ([os.path.basename (afile).split ('_')[0] for afile in fileList])
get_periods = lambda aString: numpy.array ([pandas.to_datetime (adate.strip ()) for adate in aString.split ('to')])
get_primaries = lambda df: df[df.SENSOR_USED_PRIMARY + '_WL_VALUE_MSL']
get_primary_sigmas = lambda df: df[df.SENSOR_USED_PRIMARY + '_WL_SIGMA']

def build_error (msg, key, verbose=False):
    
    ''' Build_error returns the error message (if verbose) to be shown on console when information
        is incomplete from station list or offset file.

        input param
        -----------
        msg (str): error message to be displayed
        key (str): key that will be skipped
    '''
    if not verbose: return
    print ('#    {0}'.format (msg))
    print ('#    Skipping {0} might lead to issue later.'.format (key))

# +------------------------------------------------------------
# | Function for massaging station list, offset, backup files
# +------------------------------------------------------------
def redefine_begin_end_date (dates, all_begin, all_end):
    
    ''' Redefine_begin_end_date redefines the dates of all stations if needed. For any dates
        before the all_begin dates (i.e. the start of downloaded time-series), the dates are
        re-defined to be the start of time-series. For any dates after the all_end dates (i.e.
        the end of downloaded time-series), the dates are re-defined to be the end of time-
        series.

        input params
        ------------
        dates (numpy.array): each element is either a begin or end date of a station for
                             either train, test, or validation set.
        all_begin (numpy.array): each element is the download begin date of a station.
        all_end (numpy.array): each element is the download end date of a station.

        return param
        ------------
        dates (numpy.array): redefined begin / end date for a set of a station for
                             either train, test, or validation set. 
    '''

    # Check if this date is before end date
    dates_before_allend = pandas.to_datetime (dates) <=  pandas.to_datetime (all_end)
    dates[~dates_before_allend] = all_end[~dates_before_allend]

    # Check if this date is after begin date
    dates_after_allbegin = pandas.to_datetime (dates) >=  pandas.to_datetime (all_begin)
    dates[~dates_after_allbegin] = all_begin[~dates_after_allbegin]
    return dates

def read_station_list (stations):
    
    ''' Read_station_list reads a station list file downloaded from WL-AI Google drive:

        https://docs.google.com/spreadsheets/d/1tLoaNPWNnHneWOZlpS38S7ldlkSs0wiCq_E6_39u_Qg/edit?usp=sharing

        It contains information such as primary sensors, offset values, etc indicated by Lindsay. The
        first 4 rows are skipped. First 2 are color code for station status. Third row is empty, and
        forth row is the column names, which are redefined manually as mf_columns for convenience with
        shorter names without spacings.

        If no periods are indicated for training, validation, and testing in the file,
            * Train period = data set download begin date to train_end_date i.e. 2016-12-31
            * Validation period = valid_start_date i.e. 2017-01-01 to valid_end_date i.e. 2018-12-31
            * Test period = test_start_date i.e. 2019-01-01 to data set download end date
        If the time series for this station is too short, it may not have a testing set, and its
        validation period may be shorter than 2 years.        

        input param
        -----------
        stations (numpy.array): Array containing unique, sorted station IDs available for cleaning.

        return param
        ------------
        station_frame (pandas.DataFrame): a dataframe contains station settings.
    '''

    # Only load the stations that are relevant
    station_frame = pandas.read_csv (stationFile, skiprows=4, names=station_columns)
    station_frame = station_frame[numpy.in1d (station_frame.STATION_ID.astype (str), stations)]

    # Replace training / validation / testing dates based on download dates if missing 
    n_stations = len (station_frame)
    all_begin = station_frame.DATES_DOWNLOADED.apply (lambda x: x.split (' ')[0])
    all_end   = station_frame.DATES_DOWNLOADED.apply (lambda x: x.split (' ')[2])
    # Handle the period of each dataset type (i.e. train / test / validation)
    for dtype in dataset_types:
        # Define default begin / end dates of this type
        begin = all_begin.values if dtype=='train' else [test_start_date] * n_stations if dtype=='test'  else [valid_start_date] * n_stations
        end   = all_end.values   if dtype=='test'  else [train_end_date]  * n_stations if dtype=='train' else [valid_end_date]   * n_stations
        # Make sure begin & end dates are within downloaded range
        begin = redefine_begin_end_date (numpy.array (begin), all_begin, all_end)
        end = redefine_begin_end_date (numpy.array (end), all_begin, all_end)
        # Redefine data set date period. If begin date is the same as end date, not enough data i.e. NaN
        period = [numpy.NaN if begin[index]==end[index] else begin[index] + ' to ' + end[index] for index in range (len (end))]
        column_name = 'DATES_FOR_TRAINING' if dtype=='train' else 'DATES_FOR_TESTING' if dtype=='test' else 'DATES_FOR_VALIDATION' 
        station_frame[column_name] = period

    return station_frame

def read_offset_file (station):
    
    ''' 

        input param
        -----------
        stations (numpy.array): Array containing unique, sorted station IDs available for cleaning.

        return param
        ------------
        auditframe (pandas.DataFrame): Offset data approved by Lindsay; only available station IDs are included.
    '''

    offsetFile = glob (rawCSVPath + station + '_offsets.csv')[0]
    offset_frame = pandas.read_csv (offsetFile)
    offset_frame.columns = [col.strip() for col in offset_frame.columns]
    # Drop rows with [NULL] - Only exist in SENSOR_ID and DCP_NUM
    offset_frame = offset_frame.replace ('[NULL]', numpy.nan)
    # Convert all begin date times to Timestamp. Backup offset / gain end dates are unreliable.
    offset_frame['BEGIN_DATE_TIME'] = pandas.to_datetime (offset_frame.BEGIN_DATE_TIME.values)
    offset_frame['END_DATE_TIME'] = pandas.to_datetime (offset_frame.END_DATE_TIME.values)
    return offset_frame

def read_backup_files (station):
    
    ''' 

        input param
        -----------
        stations (numpy.array): Array containing unique, sorted station IDs available for cleaning.

        return param
        ------------
        auditframe (pandas.DataFrame): Offset data approved by Lindsay; only available station IDs are included.
    '''

    offsetFile = glob (rawCSVPath + station + '_B1_gain_offsets.csv')[0]
    offset_frame = pandas.read_csv (offsetFile)
    offset_frame.columns = [col.strip() for col in offset_frame.columns]
    # Drop rows with [NULL] - Only exist in SENSOR_ID and DCP_NUM
    offset_frame = offset_frame.replace ('[NULL]', numpy.nan)
    # Convert PRAMETER_NAME into gains / offsets array
    offset_frame['IS_GAIN'] = offset_frame.PARAMETER_NAME == 'ACC_BACKUP_GAIN'
    offset_frame['IS_OFFSET'] = offset_frame.PARAMETER_NAME == 'ACC_BACKUP_OFFSET'
    # Convert all begin / end date times to Timestamp.  Backup offset / gain end dates are unreliable.
    offset_frame['BEGIN_DATE_TIME'] = pandas.to_datetime (offset_frame.BEGIN_DATE_TIME.values)
    offset_frame = offset_frame.sort_values (by='BEGIN_DATE_TIME').reset_index()
    # Drop unnecessary columns
    offset_frame = offset_frame.drop (axis=1, columns=['END_DATE_TIME', 'PARAMETER_NAME', 'index'])
    # Transform offset_frame into unique time per row with both gain and offset info
    offset, gain = 0, 1 # just to start with    
    offsets = {'BEGIN_DATE_TIME':[], 'B1_DCP':[], 'OFFSET':[], 'GAIN':[]}
    for date_time in sorted (offset_frame.BEGIN_DATE_TIME.unique()):
        # Extract this time 
        subframe = offset_frame[offset_frame.BEGIN_DATE_TIME == date_time]
        if len (subframe) > 2:
            print ('#    This {0} from backup G/O file has duplicated date time.'.format (date_time))
        offsets['BEGIN_DATE_TIME'].append (pandas.to_datetime (date_time))
        offsets['B1_DCP'].append (subframe.B1_DCP.values[0]) # should be the same between 2 rows
        # Determine the offset value
        if len (subframe[subframe.IS_OFFSET]) > 0:
            this_offset = subframe[subframe.IS_OFFSET].ACC_PARAM_VAL.values[0]
            if numpy.isfinite (this_offset): offset = this_offset
        offsets['OFFSET'].append (offset)
        # Determine the gain value
        if len (subframe[subframe.IS_GAIN]) > 0:
            this_gain = subframe[subframe.IS_GAIN].ACC_PARAM_VAL.values[0]
            if numpy.isfinite (this_gain): gain = this_gain
        offsets['GAIN'].append (gain)
    offset_frame = pandas.DataFrame (offsets)

    return offset_frame

# +------------------------------------------------------------
# | Function for 0. Dividing data set into 3
# +------------------------------------------------------------
def read_dataset_periods (station_frame, verbose=False):
    
    ''' Read_dataset_periods reads in the date periods for training / testing / validation sets.
        The periods are expected to be in a format of 'YYYY-MM-DD to YYYY-MM-DD'. The begin date
        starts at 00:00, and the end date ends at 23:59. If no periods are indicated in the file,
            * Train period = data set download begin date to train_end_date i.e. 2016-12-31
            * Validation period = valid_start_date i.e. 2017-01-01 to valid_end_date i.e. 2018-12-31
            * Test period = test_start_date i.e. 2019-01-01 to data set download end date
        If the time series for this station is too short, it may not have a testing set, and its
        validation period may be shorter than 2 years.

        input params
        ------------
        station_frame (pandas.DataFrame): info from station List file about this station
        verbose (bool): If true, print progress on console.

        return param
        ------------
        periods (dictionary): {dataset_type: [begin, end]} indicating the dataset type and the
                              corresponding period dates.
    '''

    periods = {}

    for dtype in dataset_types:
        ## Read the period from the station list. These periods are redefined by now.
        key = 'DATES_FOR_TRAINING' if dtype=='train' else \
              'DATES_FOR_TESTING'  if dtype=='test'  else 'DATES_FOR_VALIDATION'
        period = station_frame[key].values[0]
        ## Try to actually read period for this dataset
        try: 
            period = get_periods (period)
            # Add 23 hours and 59 minutes to end date
            period[1] += pandas.offsets.Hour (23) + pandas.offsets.Minute (59)
        except:
            # If cannot interpret from string, this period is an empty list.
            msg = 'Cannot read period of {0} for {1} set !!!'.format (station_frame.STATION_ID.values[0], dtype)
            build_error (msg, 'key', verbose=verbose)
            period = []
        ## Add to periods dictionary
        periods[dtype] = period

    return periods

def apply_periods (dataframe, periods, verbose=False):
    
    ''' Apply_periods removes any rows that are not within the time periods for training / testing / 
        validation sets. The three time periods are parsed as a dictionary. This function loops through
        each dataset type, extracts the relevant rows from the whole dataframe, and appends it to a
        new dataframe. If input periods is empty, we assume nothing to be done i.e. full dataframe
        is returned.

        input param
        -----------
        dataframe (pandas.DataFrame): time-series dataframe at a specific station, before extracting
                                      relevant date ranges for the requested dataset type.
        periods (dictionary): {dataset_type: [begin, end]} indicating the dataset type and the
                              corresponding period dates.

        return param
        ------------
        dataframe (pandas.DataFrame): time-series dataframe at a specific station, after extracting
                                      relevant date ranges for the requested dataset type.
    '''

    ## If no periods available, do nothing to time-series dataframe
    if len (periods) == 0: return dataframe
    if verbose: print ('#    A total of {0} rows from this station.'.format (len (dataframe)))    

    newframe, setType = None, []
    ## Loop through each available offset periods
    for dtype, period in periods.items():
        # If no period for this set, skip.
        if len (period) == 0: continue
        # Extract the slice based on period
        thisframe = dataframe.loc[period[0]:period[1]]
        # Assign boolean indicating data set type
        setType += [dtype] * len (thisframe)
        # If this is the first one, just assign it to newframe
        if newframe is None:
            newframe = thisframe
        else:
            # For the following periods, append it to the new frame
            newframe = newframe.append (thisframe)
        if verbose: print ('#       * {0} for {1} between {2} and {3}.'.format (len (thisframe), dtype, period[0], period[1]))
    
    ## Add new column indicating dataset type
    newframe['setType'] = setType
    return newframe

# +------------------------------------------------------------
# | Function for 1. Define SENSOR_USED_PRIMARY
# +------------------------------------------------------------
def define_sensor_used (dataframe, station_frame, verbose=False):
    
    ''' Define_sensor_used defines the sensor used for each time record. The primary sensor of this
        station from the station list is the default primary sensor. This function then reads in the
        OTHER_PRIMARY_SENSOR_TYPE column and its DATES to modify the sensor used for time records
        within that DATES range. In the future, if more than 2 primary sensor types are needed at
        a station, just append the new column name to sensor_names array and add the TYPE and DATES
        columns in the station list. 

        input params
        ------------
        dataframe (pandas.DataFrame): time-series dataframe at a specific station before primary
                                    sensor type column is defined
        station_frame (pandas.DataFrame): info from station List file about this station
        verbose (bool): If true, print progress on console

        return param
        ------------
        dataframe (pandas.DataFrame): time-series dataframe at a specific station after primary
                                    sensor type column is defined    
    '''

    # Assume all time steps have the first primary sensor
    first_sensor_type = station_frame.PRIMARY_SENSOR_TYPE.values[0]
    if verbose: print ('#    The default primary type is {0}.'.format (first_sensor_type))
    dataframe['SENSOR_USED_PRIMARY'] = [first_sensor_type] * len (dataframe)

    # Loop through other available sensors
    for stype in sensor_names[1:]:
        # Extract the next sensor type
        next_sensor_type =  station_frame[stype].values[0]
        # If no sensor type, continue to the next sensor type
        if not next_sensor_type in valid_sensor_types: continue
        # If no date available for this sensor type, continue as well
        next_sensor_date = station_frame[stype.replace ('TYPE', 'DATES')].values[0]
        # Try to read the string as a date 
        try:
            period = get_periods (next_sensor_date)
        except:
            msg = 'Cannot read primary sensor used dates in station list'
            build_error (msg, next_sensor_type, verbose=verbose)
        # Modify the primary used for the time period specified
        dataframe.loc[period[0]:period[1], 'SENSOR_USED_PRIMARY'] = next_sensor_type
        # Print info to console if asked
        if verbose:
            nChanged = len (dataframe[dataframe.SENSOR_USED_PRIMARY == next_sensor_type])
            print ('#    {0} records are changed to secondary sensor type of {1}.'.format (nChanged, next_sensor_type))

    return dataframe

# +------------------------------------------------------------
# | Function for 2. Handle offsets
# +------------------------------------------------------------
def read_offsets (offset_frame, verbose=False):
       
    ''' Read_offsets checks and interprets offsets information from offset file. Each station has
        its offset file. This function makes sure the begin time is before the end time. In case of
        duplicated rows with the same begin and end times, only the first row is actually read. If
        everything is correct, a dictionary {(begin, end):[sensor ID, offset]} is returned. As of
        6/16, the sensor ID is not actually used and can be None.

        input param
        -----------
        offset_frame (pandas.DataFrame): offset info from Armin's offset file for this station
        verbose (bool): If true, print progress on console.

        return param
        ------------
        offsets (dictionary): {(begin, end):[sensor ID, offset]} with the offset periods and values.
    '''

    ## If no offsets available, return empty dictionary.
    if len (offset_frame) == 0: return {}

    ## Check the offset_frame for duplicated rows and periods of offsets.
    offsets = {}
    for row in offset_frame.itertuples():
        # Extract the begin & end dates of the offset value
        begin, end = row.BEGIN_DATE_TIME, row.END_DATE_TIME
        # Make sure the end date is after begin date
        if begin > end :
            msg = 'Found an offset period with its end date earlier than its begin date'
            build_error (msg, 'OFFSET', verbose=verbose)
            continue
        # Make sure duplicated periods are not included.
        key = (begin, end)
        if key in offsets.keys():
            if verbose: print ('#       * Duplicated offset found: {0} - {1}'.format (begin, end))
            continue
        # Actually add this offset info to the dictionary. SENSOR ID will not actually be used,
        # and its value can be None. But offset value must be available.
        offsets[key] = [row.SENSOR_ID, row.OFFSET]
    
    if verbose: print ('#       * {0} sets of unique offsets are found'.format (len (offsets)))
    return offsets

def apply_offsets (dataframe, offsets, verbose=False):
    
    ''' Apply_offsets takes in a time-series dataframe and a offsets dictionary. This function loops
        through each set of offsets and apply the corresponding offset values to a sub-set of data-
        frame defined by the corresponding offset period if the verified WL sensor ID from the raw
        file matches the primary sensor ID defined previously. The sensor ID from the offset file
        is not actually used here.

        input params
        ------------
        dataframe (pandas.DataFrame): time-series dataframe at a specific station, before offsets
        offsets (dictionary): {(begin, end):[sensor ID, value]} with the offset periods and their values.

        return param
        ------------
        dataframe (pandas.DataFrame): time-series dataframe at a specific station, after offsets
    '''

    ## Initialize a boolean column indicating if offsets are applied
    dataframe['OFFSETS_APPLIED'] = False

    ## If no offsets available, do nothing to time-series dataframe
    if len (offsets) == 0: return dataframe

    ## Loop through each available offset periods
    for period, [sensor, value] in offsets.items():
        # Extract the slice based on period and modify PRIMARY column. Only apply offsets if the verified
        # sensor ID from the raw statoin file is the same as the previously defined primary sensor ID. 
        is_sensor = dataframe.loc[period[0]:period[1], 'VER_WL_SENSOR_ID'] == dataframe.loc[period[0]:period[1], 'SENSOR_USED_PRIMARY']
        if verbose: 
            print ('#       * +---------------------------------------------------------')
            print ('#       * | {0} - {1}'.format (period[0], period[1]))
            print ('#       * |   {0} records are found with matching sensor ID'.format (len (is_sensor[is_sensor])))
        # If no matching sensor, continue to the next set of offset.
        if len (is_sensor[is_sensor]) == 0: continue
        # Apply offset for specific rows
        dataframe.loc[period[0]:period[1], 'PRIMARY'][is_sensor] += value
        dataframe.loc[period[0]:period[1], 'OFFSETS_APPLIED'][is_sensor] = True
        if verbose: print ('#       * |   Offset value of {0:.5f} is added to those records'.format (value))

    if verbose:
        print ('#       * +---------------------------------------------------------')
        nApply = len (dataframe[dataframe.OFFSETS_APPLIED])
        print ('#       * Offsets are applied to a total of {0} records'.format (nApply))
    return dataframe

# +------------------------------------------------------------
# | Function for 8-10. Extra massaging
# +------------------------------------------------------------
def replace_nan (dataframe, verbose=False):
    
    ''' Replace_nan checks for invalid entries in the following columns:
            * PRIMARY, PRIMARY_SIGMA, PRIMARY_RESIDUAL
            * BACKUP, BACKUP_SIGMA, BACKUP_RESIDUAL
        Invalid values are replaced by 0.0. For PRIMARY, PRIMARY_SIGMA, BACKUP, and BACKUP_SIGMA,
        a _TRUE column is added to indicate whether its original values are invalid.

        input params
        ------------
        dataframe (pandas.DataFrame): time-series dataframe at a specific station before NaN
                                      values are replaced by 0.0.
        verbose (bool): If true, print progress on console

        return param
        ------------
        dataframe (pandas.DataFrame): time-series dataframe at a specific station after NaN
                                      values are replaced by 0.0.
    '''

    # Loop over PRIMARY, PRIMARY_SIGMA, PRIMARY_RESIDUAL, BACKUP, BACKUP_SIGMA, and BACKUP_RESIDUAL
    for main in ['PRIMARY', 'BACKUP']:
        for suffix in ['', '_SIGMA', '_RESIDUAL']:
            # Reconstruct the actual key
            key = main + suffix
            # Which row has a valid value for the key? 1 = is valid; 0 = is invalid
            isValid = (~dataframe[key].isna ()).astype (int)
            # Add in _TRUE column for PRIMARY, PRIMARY_SIGMA, BACKUP, and BACKUP_SIGMA
            if not 'RESIDUAL' in key: dataframe[key + '_TRUE'] = isValid
            # Print on console if any invalid values are found and replaced
            nInvalid = len (isValid[~isValid.astype (bool)])
            if verbose and nInvalid > 0:
                print ('#       * {0} {1} values are reset to 0.0'.format (nInvalid, key))
            # Replace the invalid value by 0.0
            dataframe.loc[~isValid.astype (bool), key] = 0.0

    return dataframe

def cap_values (dataframe, station_frame, verbose=False):
    
    ''' Cap_values replaces any PRIMARY and BACKUP water level that are beyond an accepted range of
        water levels. The water level thresholds (min, max) are defined in the station list file. 
        Their SIGMAs are must also be within 0 and 1. Any values that are beyond the accepted range
        are replaced by either the max or min values.

        input params
        ------------
        dataframe (pandas.DataFrame): time-series dataframe at a specific station before water
                                      level and sigma values are cap-ed.
        station_frame (pandas.DataFrame): info from station List file about this station
        verbose (bool): If true, print progress on console

        return param
        ------------
        dataframe (pandas.DataFrame): time-series dataframe at a specific station after water
                                      level and sigma values are cap-ed.
    '''

    ## 1. Cap PRIMARY & BACKUP water level
    #  Caps are defined in station list. If not, use default value.
    wlmin = station_frame.WL_MIN.values[0]
    if not numpy.isfinite (wlmin): wlmin = -1 * WL_default_caps
    wlmax = station_frame.WL_MAX.values[0] 
    if not numpy.isfinite (wlmax): wlmax = WL_default_caps
    #  Apply caps
    for key in ['PRIMARY', 'BACKUP']:
        if verbose:
            nBeyondMax = len (dataframe[dataframe[key] > wlmax])
            nBeyondMin = len (dataframe[dataframe[key] < wlmin])
            if nBeyondMax > 0: print ('#       * {0} records have {1} above max value of {2}'.format (nBeyondMax, key, wlmax))
            if nBeyondMin > 0: print ('#       * {0} records have {1} below min value of {2}'.format (nBeyondMin, key, wlmin))
        dataframe.loc[dataframe[key] > wlmax, key] = wlmax
        dataframe.loc[dataframe[key] < wlmin, key] = wlmin

    ## 2. Cap PRIMARY_SIGMA & BACKUP_SIGMA between 0 and 1
    for key in ['PRIMARY_SIGMA', 'BACKUP_SIGMA']:
        if verbose:
            nBeyondMax = len (dataframe[dataframe[key] > 1])
            nBeyondMin = len (dataframe[dataframe[key] < 0])
            if nBeyondMax > 0: print ('#       * {0} records have {1} above max value of 1'.format (nBeyondMax, key))
            if nBeyondMin > 0: print ('#       * {0} records have {1} below min value of 0'.format (nBeyondMin, key))
        dataframe.loc[dataframe[key] > 1, key] = 1
        dataframe.loc[dataframe[key] < 0, key] = 0

    return dataframe

def scale_values (dataframe, station_frame, verbose=False):
    
    ''' Scale_values scales PRIMARY, BACKUP, and PREDICTION by GT range stated in the station list. 

        input params
        ------------
        dataframe (pandas.DataFrame): time-series dataframe at a specific station before values
                                      are scaled by GT range.
        station_frame (pandas.DataFrame): info from station List file about this station
        verbose (bool): If true, print progress on console

        return param
        ------------
        dataframe (pandas.DataFrame): time-series dataframe at a specific station after values
                                      are scaled by GT range.
    '''

    # Read GT range from station csv
    gtrange = station_frame.GT_RANGE.values[0]
    if verbose: print ('#       * GT range is {0}'.format (gtrange))
    # Scale each column
    for key in ['PRIMARY', 'BACKUP', 'PREDICTION']:
        dataframe[key] = dataframe[key] / gtrange

    return dataframe

# +------------------------------------------------------------
# | Function for cleaning a station
# +------------------------------------------------------------
def massage (dataframe, station_frame, offset_frame, verbose=False):
    
    ''' Massage function performs data-cleaning for a specific station. The procedure is documented
        in Greg's Google doc:

        https://docs.google.com/document/d/1BfyIQE9GXPCRbBSkyurd3UeGqpGkAr1UYkMZzh5LBNk/edit?usp=sharing

        Each step is also commeted below. The input dataframe must be a time-series dataframe i.e.
        the index are datetimes. Besides dropping redundant rows, this function also adds information
        including sensor type, primary definition, offset application, and target definition, based on
        the station list and offsets for this specific station. 

        input params
        ------------
        dataframe (pandas.DataFrame): time-series dataframe at a specific station, before cleaning.
        station_frame (pandas.DataFrame): info from station List file about this station
        offset_frame (pandas.DataFrame): offset info from Armin's offset file for this station
        verbose (bool): If true, print progress on console.

        return param
        ------------
        dataframe (pandas.DataFrame): time-series dataframe at a specific station, after cleaning.
    '''

    # 0. Divide data into train, validate, test sets based on dates
    if verbose: print ('#  0. Divide data into 3 sets')
    periods = read_dataset_periods (station_frame, verbose=verbose)
    dataframe = apply_periods (dataframe, periods, verbose=verbose)
    if verbose: print ('#')

    # 5. Add SENSOR_USED_PRIMARY column from station list
    if verbose: print ('#  1. Define SENSOR_USED_PRIMARY column')
    dataframe = define_sensor_used (dataframe, station_frame, verbose=verbose)
    if verbose: print ('#')

    # 6. Add PRIMARY column based on SENSOR_USED_PRIMARY
    if verbose: print ('#  2. Define PRIMARY water level based on SENSOR_USED_PRIMARY')
    dataframe['PRIMARY'] = dataframe.apply (get_primaries, axis=1)
    if verbose: print ('#')

    # 7. Apply offset values
    if verbose: print ('#  3. Apply offset to PRIMARY')
    offsets = read_offsets (offset_frame, verbose=verbose)
    dataframe = apply_offsets (dataframe, offsets, verbose=verbose)
    if verbose: print ('#')

    # 8. Add PRIMARY_SIGMA i.e. A1_WL_SIGMA & PRIMARY_RESIDUAL i.e. PRIMARY - PRED
    if verbose: print ('#  4. Define PRIMARY_SIGMA & PRIMARY_RESIDUAL')
    dataframe['PRIMARY_SIGMA'] = dataframe.apply (get_primary_sigmas, axis=1)
    dataframe['PRIMARY_RESIDUAL'] = dataframe.PRIMARY - dataframe.PRED_WL_VALUE_MSL
    if verbose: print ('#')

    # 9. Add BACKUP i.e. B1_WL_VALUE_MSL, BACKUP_SIGMA i.e. B1_WL_VALUE_MSL, BACKUP_RESIDUAL i.e. B1_WL_VALUE_MSL - PRED
    if verbose: print ('#  5. Define BACKUP, BACKUP_SIGMA & BACKUP_RESIDUAL')
    dataframe['BACKUP'] = dataframe.B1_WL_VALUE_MSL
    dataframe['BACKUP_SIGMA'] = dataframe.B1_WL_SIGMA
    dataframe['BACKUP_RESIDUAL'] = dataframe.B1_WL_VALUE_MSL - dataframe.PRED_WL_VALUE_MSL
    if verbose: print ('#')

    # Add PREDICTION i.e. PRED_WL_VALUE_MSL & VERIFIED i.e. VER_WL_VALUE_MSL
    if verbose: print ('#  6. Define VERIFIED & PREDICTION')
    dataframe['VERIFIED'] = dataframe.VER_WL_VALUE_MSL
    dataframe['PREDICTION'] = dataframe.PRED_WL_VALUE_MSL
    if verbose: print ('#')

    # 10. Add TARGET based on target threshold between PRIMARY and VER_WL_VALUE_MSL
    if verbose: print ('#  7. Define TARGET with threshold value of {0} meters'.format (target_thresh))
    dataframe['TARGET'] = ((dataframe.PRIMARY - dataframe.VER_WL_VALUE_MSL).abs() <= target_thresh).astype (int)
    if verbose:
        nSpikes = len (dataframe[dataframe.TARGET==0])
        print ('#       * {0} records are identified as target spikes'.format (nSpikes))
        print ('#')

    # 11. Replace all NaNs or missing values with 0 and create dummy boolean column for PRIMARY, BACKUP, and their SIGMA
    if verbose: print ('#  8. Replace NaNs with 0 and dummy column for PRIMARY, BACKUP and their SIGMA, RESIDUAL')    
    dataframe = replace_nan (dataframe, verbose=verbose)
    if verbose: print ('#')

    # 12. Cap PRIMARY & BACKUP between WL_MIN & WL_MAX and their SIGMAs between 0 and 1
    if verbose: print ('#  9. Cap PRIMARY & BACKUP and their SIGMAs')
    dataframe = cap_values (dataframe, station_frame, verbose=verbose)
    if verbose: print ('#')

    # 13. Scale PRIMARY, BACKUP, and PREDICTION by GT range
    if verbose: print ('#  10. Scale PRIMARY, BACKUP, and PREDICTION by GT range.')    
    dataframe = scale_values (dataframe, station_frame, verbose=verbose)
    if verbose: print ('#')

    # Keep columns requested in specific order
    return dataframe[out_columns + ['setType']]

def massage_backup_data (dataframe, backup_go_frame, verbose=False):

    # Drop the raw backup water level data - this will be replaced later
    #dataframe = dataframe.drop (axis=1, columns=['B1_WL_VALUE_MSL'])

    # Loop through each gain / offset rows and apply offset / gain to the backup_frame.
    for index, row in enumerate (backup_go_frame.itertuples()):
        end_date = pandas.to_datetime ('2100-12-31') if index==len(backup_go_frame)-1 else \
                   backup_go_frame.iloc[index+1].BEGIN_DATE_TIME- pandas.Timedelta (minutes=1)
        offset_period = (row.BEGIN_DATE_TIME, end_date)
        # Check 
        same_DCP = dataframe.loc[offset_period[0]:offset_period[1]].B1_DCP == row.B1_DCP
    
        ## update with g/o and MSL for those that have the same DCP
        subframe = dataframe.loc[offset_period[0]:offset_period[1]][same_DCP]
        if len (subframe) > 0:
            backup_values = subframe.B1_WL_VALUE * row.GAIN + row.OFFSET - subframe.B1_MSL
            dataframe.loc[offset_period[0]:offset_period[1], 'B1_WL_VALUE'][same_DCP] = backup_values.values
        ## Apply MSL for those that don't have the same DCP 
        ## Change to NaN instead of applying MSL 
        subframe = dataframe.loc[offset_period[0]:offset_period[1]][~same_DCP]
        if len (subframe) > 0:
            #backup_values = subframe.B1_WL_VALUE - subframe.B1_MSL
            dataframe.loc[offset_period[0]:offset_period[1], 'B1_WL_VALUE'][~same_DCP] = numpy.NaN # backup_values.values
    
    # Now that backup_frame is modified, merge the B1_WL_VALUE_MSL back to main dataframe
    dataframe['B1_WL_VALUE_MSL'] = dataframe.B1_WL_VALUE
    dataframe = dataframe.drop (axis=1, columns=['B1_WL_VALUE', 'B1_MSL'])
    return dataframe

def clean_a_station (station, station_frame, offset_frame, backup_go_frame, verbose=False):
    
    ''' Clean_a_station reads in the raw data generated by Armin (directly from database) and 
        converts it to a time-series dataframe, massage it based on Greg's Google doc. The
        cleaned dataframe is then returned.

        input params
        ------------
        station (str): Station ID to be cleaned
        station_frame (pandas.DataFrame): info from station List file about this station
        offset_frame (pandas.DataFrame): offset info from Armin's offset file for this station
        verbose (bool): If true, print progress on console.

        return param
        ------------
        dataframe (pandas.DataFrame): a cleaned dataframe for this station
    '''

    # Read the raw csv file
    aFile = glob (rawCSVPath + station + '_raw_ver_merged_wl.csv')[0]
    dataframe = pandas.read_csv (aFile)
    if verbose:
        print ('#  Raw file {0} is successfully read.'.format (os.path.basename (aFile)))
        print ('#')
    # Handle duplicated time stamps
    repeated_times = dataframe.DATE_TIME.value_counts()[dataframe.DATE_TIME.value_counts () >1].index
    if len (repeated_times) > 0:
        if verbose: print ('#  This station has {0} repeated times at ...'.format (len (repeated_times)))
        remove_row_indices = []
        for repeated_time in repeated_times:
            if verbose: print ('#     * {0}'.format (repeated_time))
            found_good = False
            subframe = dataframe[dataframe.DATE_TIME == repeated_time]
            for index, row in subframe.iterrows():
                if verbose: print ('#         - row {0}'.format (index))
                ## Try to look for -99999.999 values
                any999 = [row[col] == -99999.999 for col in row.index]
                if numpy.array (any999).any():
                    if verbose: print ('#            -99999.999 is found!')
                    remove_row_indices.append (index)
                    continue
                ## Try to use the first 
                if not found_good:
                    if verbose: print ('#            This row is the official data for this timestamp!')
                    found_good = True
                    continue
                if verbose: print ('#            Good data exists for timestamp - removing this one!')
                remove_row_indices.append (index)
        dataframe = dataframe.drop (axis=0, index=remove_row_indices)
        if verbose: print ('#')
        
    # Massage dataframe
    dataframe['DATE_TIME'] = pandas.to_datetime (dataframe.DATE_TIME)
    # Convert data frame to time-series
    dataframe.index = dataframe.DATE_TIME
    # Handle backup data
    dataframe = massage_backup_data (dataframe, backup_go_frame, verbose=verbose)
    return massage (dataframe, station_frame, offset_frame, verbose=verbose)

def clean_all_stations (stations, station_list, rawFiles, verbose=False):

    ''' Clean_all_stations loops through each station and preform cleaning procedure.
        All cleaned dataframes are collected as a dictionary with key = station ID.

        input params
        ------------
        stations (numpy.array): an array of all stations to be cleaned
        station_list (pandas.DataFrame): all station configurations for this project
        verbose (bool): If true, print progress on console.

        return param
        ------------
        cleaned (dict): {stationID:cleaned dataframe}
    '''

    cleaned = {}
    for station in stations:
        # Print on console for this station
        if verbose:
            print ('# ======================================================')
            print ('# Cleaning {0} ..'.format (station))
            print ('#')
        # Determine the output csv processed file
        #infile = os.path.basename (rawFiles[numpy.array ([station in afile for afile in rawFiles])][0])
        #outfile = infile.replace ('raw', 'processed').split ('.')[0]
        # Extract info of this station
        station_frame = station_list[station_list.STATION_ID.astype (str) == station]
        # Load the offset info from offset csv file
        offset_frame = read_offset_file (station)
        # Load the back up data, offset, and gain info from other csv files
        backup_go_frame = read_backup_files (station)
        # Clean !
        cleaned_station = clean_a_station (station, station_frame, offset_frame, backup_go_frame, verbose=verbose)
        if cleaned_station is not None: cleaned[station] = cleaned_station
        if verbose: print ('#')

    if verbose:
        print ('# ======================================================')
        print ('#')

    return cleaned

# +------------------------------------------------------------
# | Function for adding in neighbor station info
# +------------------------------------------------------------
def add_neighbor_to_dataframe (dataframe, neighbor_dataframe=None, verbose=False):

    ''' Add_neighbor_to_dataframe adds in the required columns related to neighboring station.
        If the input neighbor_dataframe is not available, the columns will still be added but
        with NaN values for all entries.

        input params
        ------------
        dataframe (pandas.DataFrame): time-series dataframe at the targeted station
        neighbor_dataframe (pandas.DataFrame): time-series dataframe of the neighbor station
        verbose (bool): If true, print progress on console

        return prarm
        ------------
        dataframe (pandas.DataFrame): time-series dataframe at the targeted station after
                                      adding neighbor columns.
    '''

    if verbose:
        line = 'does not exist' if neighbor_dataframe is None else 'exists'
        print ('#    * Neighbor data frame {0} in cleaned station list.'.format (line))

    for key in neighbor_keys:
        if neighbor_dataframe is None:
            dataframe[key] = numpy.NaN
            continue
        dataframe[key] = neighbor_dataframe.loc[dataframe.index, '_'.join (key.split ('_')[1:])]
        #new_columns = list (dataframe) + [key]
        #dataframe = pandas.concat ([dataframe, neighbor_dataframe['_'.join (key.split ('_')[1:])]], axis=1)
        #dataframe.columns = new_columns
        #dataframe[key] = numpy.NaN if neighbor_dataframe is None else \
        #                 neighbor_dataframe['_'.join (key.split ('_')[1:])]
        # Remove any rows that wasn't there before merging
        #dataframe = dataframe[~dataframe.setType.isna()]

    return dataframe

def add_all_neighbors (cleaned, station_list, verbose=False):
    
    ''' Add_all_neighbors loops through each station and add 4 new columns related to its
        neighbor station. The neighbor station should be already be in the cleaned dict.
        If not, these columns will have NaN values.

        input params
        ------------
        cleaned (dict): all cleaned dataframe from all stations before neighbor info
        station_list (pandas.DataFrame): all station configurations for this project
        verbose (bool): If true, print progress on console.

        return param
        ------------
        cleaned (dict): all cleaned dataframe from all stations after neighbor info is added
    '''

    for station, dataframe in cleaned.items():
        # Print on console for this station
        if verbose:
            print ('# ======================================================')
            print ('# Adding neighbor info to {0} ..'.format (station))
        # Figure out the ID of neighbor station
        station_frame = station_list[station_list.STATION_ID.astype (str) == station]
        neighborID = str (station_frame.NEIGHBOR_STATION_ID.values[0])
        if verbose: print ('#    * Neighbor ID: {0}'.format (neighborID))
        # Add in neighbor information
        neighbor_frame = cleaned[neighborID] if neighborID in cleaned.keys() else None
        dataframe = add_neighbor_to_dataframe (dataframe, neighbor_dataframe=neighbor_frame, verbose=verbose)        
        # Rearrange column ordering
        cleaned[station] = dataframe[out_columns + neighbor_keys + ['setType']]
        if verbose: print ('#')

    if verbose:
        print ('# ======================================================')
        print ('#')

    return cleaned

# +------------------------------------------------------------
# | Function for writing out csv files
# +------------------------------------------------------------
def write_processed_files (cleaned, rawFiles, verbose=False):
    
    ''' Write_processed_files splits and dumps the training, testing, and validation sets
        for all stations.

        input params
        ------------
        cleaned (dict): all cleaned dataframe from all stations after neighbor info is added
        rawFiles (numpy.array): an array of full filenames of raw data from all stations
        verbose (bool): If true, print progress on console.
    '''

    if verbose: '# Dump processed files!'
    for station, dataframe in cleaned.items():

        # don't write frames that don't have neighbor info
        #if pandas.isnull (dataframe.NEIGHBOR_PRIMARY).all(): continue

        # Determine the output csv processed file
        infile = os.path.basename (rawFiles[numpy.array ([station in afile for afile in rawFiles])][0])
        outfilebase = infile.replace ('raw', 'processed').split ('.')[0]        
        # Loop through available train, validation, and test set
        for dtype in dataframe.setType.unique ():
            # Determine the actual file name
            outfile = outfilebase + '_' + dtype
            # Extract the set & drop the setType column
            subframe = dataframe[dataframe.setType == dtype].drop (axis=1, columns=['setType'])
            # Write the dataframe out!
            subframe.to_csv (processedPath + outfile + '.csv', index=False)

    if verbose: print ('#')

###############################################
## Script begins here!
###############################################
if __name__ == '__main__':

    if verbose:
        print ('#########################################################################')
        print ('# This script cleans the raw data from Step 1 (by Armin). Cleaning steps')
        print ('# are based on Gregs Google doc. ')
        print ('#')

    ## 1. Gather all station IDs to be processed
    rawFiles = numpy.array (glob (rawCSVPath + '*_merged_wl.csv'))
    stations = sorted (numpy.unique (get_stations (rawFiles)))
    if not len (stations) == len (rawFiles):
        msg = '{0} raw CSV files are found with {1} unique station IDs.\n'.format (len (rawFiles), len (stations))
        msg += 'Please check for duplicated files per station before re-running.'
        raise IOError (msg)
    if verbose:
        print ('# {0} unique stations are found.'.format (len (stations)))

    ## 2. Load the relevant meta data from meta data csv file
    station_list = read_station_list (stations)
    stations = station_list.STATION_ID.values.astype (str)
    if verbose:
        print ('# Station List are read from {0}.'.format (os.path.basename (stationFile)))
        print ('#   {0} stations have valid train / test / validation dates for processing.'.format (len (stations)))
        print ('#')

    ## 3. Loop and clean each station
    cleaned = clean_all_stations (stations, station_list, rawFiles, verbose=verbose)

    ## 4. Find and add neighbor station info
    cleaned = add_all_neighbors (cleaned, station_list, verbose=verbose)

    ## 5. Split and write to train, validation, and test processed files
    write_processed_files (cleaned, rawFiles, verbose=verbose)

    if verbose:
        print ('# This script has completed successfully. Bye!')
        print ('#########################################################################')
        