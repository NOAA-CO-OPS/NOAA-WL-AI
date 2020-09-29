#!python37

## By Elim Thompson (09/02/2020)
##
## This script defines a station class that encapsulate all data and methods
## within a given station. Each station has a set of meta-data from WL-AI
## Station List (i.e. station info sheet):
##  https://docs.google.com/spreadsheets/d/1tLoaNPWNnHneWOZlpS38S7ldlkSs0wiCq_E6_39u_Qg/edit?usp=sharing
## 
## This class is internally used by data_cleaner class but can potentially used
## standalone if one is interested in one specific station.
## 
## Prior cleaning, the following information must be set
##  * station meta-data either from a row in the station info sheet via
##    set_station_info() or individually set them via setters.
##  * primary offsets via load_primary_offsets()
##  * backup gain and offsets via load_backup_B1_gain_offsets()
## After that, one can call clean_raw_data() to return a cleaned dataframe.
##
## 2 notes on clean_raw_data():
##  * cleaned dataframe is returned instead of stored in the class to reduce
##    memory usage. 
##  * this method does not write out a csv for cleaned dataframe because it
##    does not know about data from its neighbor station.
##
## Example snippet to use the station class:
## +-------------------------------------------------------------
## # Import station class
## import station, pandas
## 
## # Define new station instance
## station_id = 9414290 # int not string
## astation = station.station (station_id)
##
## # Tell it to create mid-step files as the cleaning process goes
## astation.create_midstep_files = True
## # Tell it where to dump any midstep files
## astation.proc_path = 'C:/TMP/'
##
## # Parse station meta-data 
## astation.gt_range = 3.131
## astation.has_bad_results = False
## astation.wl_range = (-5, 4)
## astation.train_dates = [pandas.to_datetime ('2007-01-01 00:00'),
##                         pandas.to_datetime ('2016-12-31 23:59')]
## astation.valid_dates = [pandas.to_datetime ('2017-01-01 00:00'),
##                         pandas.to_datetime ('2018-12-31 23:59')]
## astation.test_dates  = [pandas.to_datetime ('2019-01-01 00:00'),
##                         pandas.to_datetime ('2020-03-31 23:59')]
## astation.primary_type = 'A1'
## astation.other_primary_type = 'Y1'
## astation.other_primary_type  = [pandas.to_datetime ('2017-06-01 00:00'),
##                                 pandas.to_datetime ('2020-03-31 23:59')]
## astation.neighbor_id = 8418150
##
## # Define & parse the raw file location
## astation.raw_file = 'C:/to/raw/{0}_raw_ver_merged_wl.csv'.format (station_id)
## 
## # Define & parse processed file location for any midstep files
## astation.proc_path = 'C:/to/processed/'
##
## # Load primary offset and backup gain / offsets
## primary_offset_file = 'C:/to/raw/{0}_offsets.csv'.format (station_id)
## backup_gain_offset_file = 'C:/to/raw/{0}_B1_gain_offsets.csv'.format (station_id)
## astation.load_primary_offsets (primary_offset_file)
## astation.load_backup_B1_gain_offsets (backup_gain_offset_file)
## # To check the primary offsets and backup gain / offsets
## primary_offsets = astation.primary_offsets
## backup_gain_offsets = astation.backup_gain_offsets
##
## # Clean raw data & store as csv. Notes
## #  * Output all train / validation / test into 1 csv file
## #  * No neighbor info is included
## cleaned_df = astation.clean_raw_data ()
## processed = 'C:/to/processed/cleaned_data.csv'
## cleaned_df.to_csv (processed, index=False)
##
## # Collect stats
## train_stats = astation.train_stats
## valid_stats = astation.validation_stats
## test_stats  = astation.test_stats
## +-------------------------------------------------------------        
#############################################################################

###############################################
## Import libraries
###############################################
import numpy, pandas, logging, os
from scipy.interpolate import interp1d

import matplotlib
matplotlib.use ('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.rc ('text', usetex=False)
plt.rc ('font', family='sans-serif')
plt.rc ('font', serif='Computer Modern Roman')

###############################################
## Define constants
###############################################
# Raw data is divided into three dataset types based on date periods
DATASET_TYPES = ['train', 'validation', 'test']

# Only the following sensor types are accepted
VALID_SENSOR_TYPES = ['A1', 'B1', 'Y1', 'NT', 'N1', 'T1']

# TARGET threshold in meters between PRIMARY and VERIFIED
TARGET_THRESH = 0.04

# Final columns to be included in the order requested
CLEANED_COLUMNS = ['STATION_ID', 'DATE_TIME', 'SENSOR_USED_PRIMARY', 'PRIMARY',
                   'PRIMARY_TRUE', 'PRIMARY_SIGMA', 'PRIMARY_SIGMA_TRUE',
                   'PRIMARY_RESIDUAL', 'BACKUP', 'BACKUP_TRUE', 'BACKUP_SIGMA',
                   'BACKUP_SIGMA_TRUE', 'BACKUP_RESIDUAL', 'PREDICTION',
                   'VERIFIED', 'VERIFIED_RESIDUAL','TARGET', 'OFFSETS_APPLIED', 
                   'VERIFIED_SENSOR_ID']

# Keys for the cleaning summary sheet. Each set has its own summary dictionary.
CLEAN_STATS_KEYS = ['has_bad_results', 'n_raw', 'has_repeated_raw', 'n_total',
		            'n_with_primary_sensor', 'n_with_other_primary_sensor',
		            'n_primary_offsets_applied', 'n_spikes', 
                    'n_nan_verified', 'n_nan_verified_valid_primary',
                    'n_nan_verified_nan_primary',
                    'n_nan_primary', 'n_nan_primary_sigma',
                    'n_nan_backup', 'n_nan_backup_sigma', 
                    'n_capped_primary_max', 'n_capped_primary_min',
                    'n_capped_backup_max', 'n_capped_backup_min',
                    'n_capped_primary_sigma_max', 'n_capped_primary_sigma_min',
                    'n_capped_backup_sigma_max', 'n_capped_backup_sigma_min']

# Keys for statistics dictionary for the difference between primary and verified
DIFF_STATS_KEYS = ['lower', 'upper', 'min', 'max', 'mean']

# Expected columns in raw primary offset files
PRIMARY_OFFSET_KEYS = ['BEGIN_DATE_TIME', 'END_DATE_TIME', 'SENSOR_ID', 'OFFSET']

# Columns in converted backup gain / offset dataframe
RAW_BACKUP_GO_KEYS = ['BEGIN_DATE_TIME', 'B1_DCP', 'PARAMETER_NAME', 'ACC_PARAM_VAL']
CONVERTED_BACKUP_GO_KEYS = ['BEGIN_DATE_TIME', 'B1_DCP', 'OFFSET', 'GAIN']

# Number of bins in histogram of primary - verified
HIST_NBINS = 10

# Possible types of number
NUMBER_TYPES = [float, int, numpy.float, numpy.float16, numpy.float32,
                numpy.float64, numpy.int, numpy.int0, numpy.int8, numpy.int16,
                numpy.int32, numpy.int64]

# Possible types of array
ARRAY_TYPES = [list, tuple, numpy.ndarray]

# Settings for giant histogram with all differences
GIANT_HIST_NBINS = 50
GIANT_HIST_RANGE = [-0.1, 0.1]

###############################################
## Define lambda functions
###############################################
get_primaries = lambda df: df[df.SENSOR_USED_PRIMARY + '_WL_VALUE_MSL']
get_primary_sigmas = lambda df: df[df.SENSOR_USED_PRIMARY + '_WL_SIGMA']
has_repeated_raw = lambda df: len (df.DATE_TIME.value_counts()[df.DATE_TIME.value_counts()>1]) > 0

###############################################
## Define station class
###############################################
class station (object):

    ''' This class encapsulates the meta-data of a specific station and perform
        cleaning on its raw data. 
    '''

    def __init__ (self, station_id):

        ''' To initialize a new station class, a station ID (int) is required. 
        '''

        self._station_id = station_id

        ## Station meta-data from info sheet
        self._gt_range = None
        self._has_bad_results = False
        self._wl_range = None
        self._train_dates = None
        self._valid_dates = None
        self._test_dates = None
        self._primary_type = None
        self._other_primary_type = None
        self._other_primary_type_period = None
        self._neighbor_id = None

        ## Raw file & processed folder locations
        self._raw_file = None
        self._proc_path = None

        ## Offsets information
        self._primary_offset_dict = None
        self._backup_gain_offset_df = None

        ## Dump mid-step files to processed folder?
        self._create_midstep_files = False

        ## Information during cleaning process
        self._has_repeated_primary_offsets = False
        self._train_stats = {key:None for key in CLEAN_STATS_KEYS}
        self._validation_stats = {key:None for key in CLEAN_STATS_KEYS}
        self._test_stats = {key:None for key in CLEAN_STATS_KEYS}

        ## Information related to the difference between primary and verified
        self._diff_stats = None
        self._diff_hist = {'edges':None, 'all':None, 'bad_only_by_thresh':None,
                           'bad_only_by_sensor_id':None}
        self._diff_hist_settings = {'nbins':GIANT_HIST_NBINS,
                                    'range':GIANT_HIST_RANGE}

        ## Logger
        self._logger = logging.getLogger ('station {0}'.format (station_id))

    def __repr__ (self):
        
        print ('station ID: {0}'.format (self._station_id))
        print ('+------------------------------------------')
        print ('|Meta-data')
        print ('|  GT range                 : {0}'.format (self._gt_range))
        print ('|  WL range                 : {0}'.format (self._wl_range))
        print ('|  Neightbor ID             : {0}'.format (self._neighbor_id))
        print ('|  Primary type             : {0}'.format (self._primary_type))
        print ('|  Other primary type       : {0}'.format (self._other_primary_type_period))
        print ('|  Other primary type period: {0}'.format (self._other_primary_type_period))
        print ('|  Training set period      : {0}'.format (self._train_dates))
        print ('|  Validation set period    : {0}'.format (self._valid_dates))
        print ('|  Testing set period       : {0}'.format (self._test_dates))
        print ('|  Has bad results          : {0}'.format (self._has_bad_results))
        print ('+------------------------------------------')
        print ('|File / folder path')
        print ('|  Raw file      : {0}'.format (self._raw_file))
        print ('|  Processed path: {0}'.format (self._proc_path))
        print ('+------------------------------------------')
        print ('|Status of offsets')        
        print ('|  Primary offsets ready         ? {0}'.format (self._primary_offset_dict is not None))
        print ('|  Repeated primary offsets found? {0}'.format (self._has_repeated_primary_offsets))
        print ('|  Backup G/O ready              ? {0}'.format (self._backup_gain_offset_df is not None))
        print ('+------------------------------------------')

    # +------------------------------------------------------------
    # | Getters & setters
    # +------------------------------------------------------------
    @property
    def station_id (self): return self._station_id

    @property
    def primary_offsets (self): return self._primary_offset_dict

    @property
    def backup_gain_offsets (self): return self._backup_gain_offset_df

    @property
    def train_stats (self): return self._train_stats

    @property
    def validation_stats (self): return self._validation_stats

    @property
    def test_stats (self): return self._test_stats

    @property
    def diff_stats (self): return self._diff_stats

    @property
    def diff_hist (self): return self._diff_hist

    @property
    def create_midstep_files (self): return self._create_midstep_files
    @create_midstep_files.setter
    def create_midstep_files (self, aBoolean):
        if not isinstance (aBoolean, bool):
            message = 'Input, {0}, is not a boolean.'.format (aBoolean)
            self._logger.fatal (message)
            raise IOError (message)
        self._create_midstep_files = aBoolean

    @property
    def raw_file (self): return self._raw_file
    @raw_file.setter
    def raw_file (self, apath):
        ## Make sure input path exists
        self._check_file_path_existence (apath)
        self._logger.info ('Raw data folder is set to {0}.'.format (apath))
        self._raw_file = apath

    @property
    def proc_path (self): return self._proc_path
    @proc_path.setter
    def proc_path (self, apath):
        ## Make sure input path exists
        self._check_file_path_existence (apath)
        self._logger.info ('Processed data folder is set to {0}.'.format (apath))
        self._proc_path = apath

    @property
    def gt_range (self): return self._gt_range
    @gt_range.setter
    def gt_range (self, gtrange):
        ## Make sure input is a number        
        self._check_is_number(gtrange)
        self._logger.info ('GT_range is set to be {0}.'.format (gtrange))
        self._gt_range = gtrange

    @property
    def has_bad_results (self): return self._has_bad_results
    @has_bad_results.setter
    def has_bad_results (self, has_bad_results):
        ## Make sure input is a boolean
        if not isinstance (has_bad_results, bool):
            message = 'Input, {0}, is not a boolean.'.format (has_bad_results)
            self._logger.fatal (message)
            raise IOError (message)
        self._logger.info ('Has_bad_results is set to be {0}.'.format (has_bad_results))
        self._has_bad_results = has_bad_results

    @property
    def wl_range (self): return self._wl_range
    @wl_range.setter
    def wl_range (self, wl_range):
        ## Make sure input is an array of 2 elements
        self._check_is_array (wl_range, length=2)
        ## Make sure the elements are numbers
        for wl in wl_range: self._check_is_number(wl)
        self._logger.info ('WL_range is set to be {0}.'.format (wl_range))        
        self._wl_range = wl_range

    @property
    def train_dates (self): return self._train_dates
    @train_dates.setter
    def train_dates (self, train_dates):
        ## Make sure input is an array of 2 elements
        self._check_is_array (train_dates, length=2)
        ## Make sure the elements are type of Timestamp
        for adate in train_dates: self._check_is_timestamp (adate)
        self._logger.info ('Train_dates is set to be {0}.'.format (train_dates))
        self._train_dates = train_dates

    @property
    def valid_dates (self): return self._valid_dates
    @valid_dates.setter
    def valid_dates (self, valid_dates):
        ## Make sure input is an array of 2 elements
        self._check_is_array (valid_dates, length=2)
        ## Make sure the elements are type of Timestamp
        for adate in valid_dates: self._check_is_timestamp (adate)
        self._logger.info ('Valid_dates is set to be {0}.'.format (valid_dates))
        self._valid_dates = valid_dates

    @property
    def test_dates (self): return self._test_dates
    @test_dates.setter
    def test_dates (self, test_dates):
        ## Make sure input is an array of 2 elements
        self._check_is_array (test_dates, length=2)
        ## Make sure the elements are type of Timestamp
        for adate in test_dates: self._check_is_timestamp (adate)
        self._logger.info ('Test_dates is set to be {0}.'.format (test_dates))
        self._test_dates = test_dates

    @property
    def primary_type (self): return self._primary_type
    @primary_type.setter
    def primary_type (self, primary_type):
        ## Make sure input primary type is one of the accepted values
        if not primary_type in VALID_SENSOR_TYPES:
            message = 'Input, {0}, is an invalid sensor type.'.format (primary_type)
            message += 'Please provide one of the followings: ' + VALID_SENSOR_TYPES
            self._logger.fatal (message)
            raise IOError (message)            
        self._logger.info ('Primary_type is set to be {0}.'.format (primary_type))
        self._primary_type = primary_type

    @property
    def other_primary_type (self): return self._other_primary_type
    @other_primary_type.setter
    def other_primary_type (self, other_primary_type):
        ## Accept None 
        if other_primary_type is None:
            self._logger.info ('Other_primary_type is set to be None.')
            self._other_primary_type = other_primary_type
            return

        ## Make sure input primary type is one of the accepted values
        if not other_primary_type in VALID_SENSOR_TYPES:
            message = 'Input, {0}, is an invalid sensor type.'.format (other_primary_type)
            message += 'Please provide one of the followings: ' + VALID_SENSOR_TYPES
            self._logger.fatal (message)
            raise IOError (message)
        self._logger.info ('Other_primary_type is set to be {0}.'.format (other_primary_type))
        self._other_primary_type = other_primary_type

    @property
    def other_primary_type_period (self): return self._other_primary_type_period
    @other_primary_type_period.setter
    def other_primary_type_period (self, other_primary_type_period):
        ## Accept None 
        if other_primary_type_period is None:
            self._logger.info ('Other_primary_type_period is set to be None.')
            self._other_primary_type_period = other_primary_type_period
            return

        ## Make sure input is an array of 2 elements
        self._check_is_array (other_primary_type_period, length=2)
        ## Make sure the elements are type of Timestamp
        for adate in other_primary_type_period: self._check_is_timestamp (adate)
        self._logger.info ('Other_primary_type_period is set to be {0}.'.format (other_primary_type_period))
        self._other_primary_type_period = other_primary_type_period

    @property
    def neighbor_id (self): return self._neighbor_id
    @neighbor_id.setter
    def neighbor_id (self, neighbor_id): 
        self._logger.info ('Neighbor_id is set to be {0}.'.format (neighbor_id))
        self._neighbor_id = neighbor_id

    # +------------------------------------------------------------
    # | Misc functions
    # +------------------------------------------------------------
    def _dump_file (self, dataname, filebasename, dataframe):

        ''' A private function to dump a mid-step dataframe into a csv file.
            So far, this function is only used to write out the backup gain and
            offset dataframe. But this function can be used for other future
            dataframes.

            input params
            ------------
            dataname (str): Variable name of the dataframe to be written
            filebasename (str): Base name of the output file
            dataframe (pandas.DataFrame): Dataframe to be written out
        '''

        ## If user didn't ask for midstep files, exit now
        if not self._create_midstep_files: return

        ## Make sure proc path is already set
        if self._proc_path is None:
            message = 'Processed data path is None. Do not know where to write.'
            self._logger.fatal (message)
            raise IOError ('Please set the path to processed folder.')

        ## Make sure the input dataframe is valid
        if dataframe is None:
            message = 'Input {0} dataframe is None. Nothing to write.'
            self._logger.debug (message.format (dataname))
            return
        if not isinstance (dataframe, pandas.core.frame.DataFrame):
            message = 'Input {0} dataframe is not a pandas dataframe. ' + \
                      'Only pandas dataframe can be written.'
            self._logger.debug (message.format (dataname))
            return

        ## Write the dataframe to a csv file!
        filename = '{0}/{1}_{2}.csv'.format (self._proc_path, self._station_id,
                                             filebasename)
        dataframe.to_csv (filename, index=False)
        message = '{0} dataframe is written to {1}.'.format (dataname, filename)
        self._logger.info (message)

    def _check_file_path_existence (self, afilepath):

        ''' A private function to check if an input file path exists. If it
            doesn't a FileNotFoundError is raised.
            
            input param
            -----------
            afilepath (str): A folder to be checked
        '''

        if not os.path.exists (afilepath):
            message = 'Path or file, {0}, does not exist!'.format (afilepath)
            self._logger.fatal (message)
            raise FileNotFoundError (message)

    def _check_is_number (self, number):

        ''' A private function to check if an input number is a float or int
            with a finite, valid value. If it doesn't an IOError is raised.
            
            input param
            -----------
            number (anything): A value to be checked
        '''

        if not type (number) in NUMBER_TYPES:
            message = 'Input, {0}, is not a float or int.'.format (number)
            self._logger.fatal (message)
            raise IOError (message)

        if not numpy.isfinite (number):
            message = 'Input, {0}, cannot be nan or infinite.'.format (number)
            self._logger.fatal (message)
            raise IOError (message)

    def _check_is_array (self, array, length=None):

        ''' A private function to check if an input array is a numpy array, a
            list, or a tuple. If it isn't, an IOError is raised. If length is
            provided, check if the input array has the required length.
            
            input param
            -----------
            array (anything): A value to be checked
            length (int): Required length of the input array
        '''

        ## 1. Check if input array is a valid array type.
        if not type (array) in ARRAY_TYPES:
            message = 'Input, {0}, is not an array / list / tuple.'.format (array)
            self._logger.fatal (message)
            raise IOError (message)

        ## Leave if no specific (integer) length required
        if length is None: return
        if not isinstance (length, int): return

        ## 2. Check if input array has a specific length
        if not len (array) == length:
            message = 'Input, {0}, does not have a length of {1}.'
            self._logger.warn (message.format (array, length))

    def _check_is_timestamp (self, adate):

        ''' A private function to check if an input is a valid pandas timestamp
            object. If it is not, an IOError is raised.
            
            input param
            -----------
            adate (anything): A value which its type is checked
        '''

        ## Check if input adate is a valid pandas timestamp object.
        if not isinstance (adate, pandas._libs.tslibs.timestamps.Timestamp) :
            message = 'Input, {0}, is not a pandas timestamp object.'.format (adate)
            self._logger.fatal (message)
            raise IOError (message)

    def _check_df_has_column (self, dataframe, column, dataname):

        ''' A private station that checks if the input dataframe contains the
            input column name. If not an IOError is raised.

            input params
            ------------
            dataframe (pandas.DataFrame): dataframe to be checked
            column (str): column name to be checked
            dataname (str): Name of the dataframe
        '''

        if not column in dataframe:
            raise IOError ('Input df does not have column, {0}.'.format (column))

    def _set_stats (self, df, key):
        
        ''' A private function to set one of the keys in stats dictionary. The
            input dataframe is expected to have a 'setType' column specifying
            the data set type of all rows. This function than counts how many
            rows in the input dataframe belong to each of the three types. The
            counts are then stored in the stats dictionary with the input key.
            
            input param
            -----------
            df (pandas.DataFrame): dataframe where # of rows per set is counted
            key (str): the key to be stored into the stats dictionary
        '''

        ## 1. Input dataframe should have 'setType' column
        if not 'setType' in df:
            self._logger.warn ('Input dataframe does not have a setType column.')
            return

        ## 2. Make sure the key is one of the valid stats keys 
        if not key in CLEAN_STATS_KEYS:
            self._logger.warn ('Input key, {0}, is not registered.')
            return

        ## 3. Loop through each dataset type and count # rows
        for dtype in DATASET_TYPES:
            # Extract the rows with the matching dataset type
            subframe = df[df.setType == dtype]
            # Get the corresponding stats attribute
            adict = getattr (self, '_' + dtype + '_stats')
            # Store stats - number of total records in this set
            adict[key] = len (subframe)
    
    # +------------------------------------------------------------
    # | Related to difference between primary and verified
    # +------------------------------------------------------------
    def _get_statistics (self, points):
        
        ''' A private function to obtain statistics from a set of data points.
            It builds a cumulative histogram and flips (inverts) it. The mean,
            min, max, top and bottom 5% are obtained to get a rough shape of
            the histogram. 

            lower: 5%
            mean: 50%
            upper: 95%
            min: minimum value from all points
            max: maximum value from all points

            If only 1 value from the entire set of data points, all stats
            values are equal to that value.

            input params
            ------------
            points (array): a set of data points where stats are obtained

            return params
            -------------
            stats (dict): 5%, 50%, 95%, and min / max of the data points
        '''
        ## Only includes valid points
        points = points [numpy.isfinite (points)]

        ## If all points are the same, all stats are the same.
        if len (numpy.unique (points)) == 1:
            value = numpy.unique (points)[0]
            return {'lower':value, 'upper':value, 'mean':value, 'min':value, 'max':value}

        dmin, dmax = min (points), max (points)
        ## Build a histogram to get 2-tailed 90% values
        hist, edge = numpy.histogram (points, bins=1400)
        cdf = numpy.cumsum (hist) / sum (hist)
        bins = edge[:-1] + (edge[1:] - edge[:-1])/2.
        icdf = interp1d (cdf, bins)
        lowP = max (0.025, min (icdf.x))
        highP = min (0.975, max (icdf.x))
        midP = 0.5
        if highP < lowP:
            highP = (max (icdf.x) - lowP)*0.975 + lowP
            midP = (max (icdf.x) - lowP)*0.5 + lowP
        lower, mean, upper = icdf(lowP), icdf(midP), icdf(highP)
        return {'lower':float (lower), 'upper':float (upper),
                'mean':float (mean), 'min':dmin, 'max':dmax}

    def _plot_sub_diff (self, axis, diff_df, reqEdges=None):

        ''' A private function to plot a sub histgram for all dataset types
            available in input diff_df. There should only be 3 at max. Their
            histograms are stacked in the order of train, validation, and test.
            If reqEdges is provided, the x-axis follows the requested bin edges.
            Otherwise, the bin edges are determined by the min / max values 
            from the differences. 

            input params
            ------------
            axis (matplotlib.Axes): axis on which plots are made
            diff_df (pandas.DataFrame): data with 'delta' and 'setType' columns
            reqEdges (array): requested bin edges

            return params
            -------------
            edges (array): bin edges used by this histogram
        '''

        if len (diff_df) == 0: return

        ## Generate histogram by dataset type
        groups = diff_df.groupby ('setType')
        if len (groups.groups.keys()) == 0: return

        #  Histogram x-axis based on requested edges if available.
        params = {'bins':reqEdges}
        if reqEdges is None:
            hist_xrange = (diff_df.delta.min(), diff_df.delta.max())
            params = {'bins':HIST_NBINS, 'range':hist_xrange}
        hists = groups.apply (numpy.histogram, **params)

        ## Plot stacked histogram
        reference = numpy.array ([0.] * (HIST_NBINS+1)).astype (float)
        for dtype in DATASET_TYPES[::-1]:
            # Next dataset type if not available
            if not dtype in hists: continue
            # Get histogram and edges
            hist, edges = hists.get (dtype)
            # Stacking the histogram
            hist = [hist[0]] + list (hist)
            yvalues = reference + numpy.log10 (hist)
            # Replace any inf / nan to 0
            yvalues[~numpy.isfinite(yvalues)] = 0
            yvalues = yvalues.astype (float)
            # Determine the color based on dataset type
            color = '#666666' if dtype=='train' else '#489cbd' if dtype=='validation' else '#ff4f6b'
            # Plot a stack histogram
            e = edges.astype (float) if reqEdges is None else reqEdges.astype (float)
            axis.fill_between (e.astype (float), reference, yvalues, color=color,
                               step='pre', label=dtype)
            # Update reference
            reference = yvalues

        ##  Plot legend if more than 1 dataset type
        if len (groups) > 1: axis.legend (loc=1, fontsize=8)

        ##  Format x-axis
        edges = e[numpy.isfinite (e)]
        axis.set_xlim ([min (edges), max(edges)])
        axis.tick_params (axis='x', labelsize=8)
        axis.set_xlabel ('Primary - Verified [meters]', fontsize=8)    
        ##  Format y-axis
        axis.set_ylim ([numpy.floor (min (yvalues)), numpy.ceil (max (yvalues))])
        axis.tick_params (axis='y', labelsize=8)
        axis.set_ylabel ('log10 #', fontsize=8)
        ##  Plot grid lines
        for ytick in axis.yaxis.get_majorticklocs():
            axis.axhline (y=ytick, color='gray', alpha=0.3, linestyle=':', linewidth=0.2)
        for xtick in axis.xaxis.get_majorticklocs():
            axis.axvline (x=xtick, color='gray', alpha=0.3, linestyle=':', linewidth=0.2)

        ## Set title
        dtype = 'all sets' if len (groups) > 1 else list (groups.groups.keys())[0]
        axis.set_title ('From {0}'.format (dtype), fontsize=9)

        return edges

    def plot_diff_histogram (self, diff_df):

        ''' A public function to plot histograms of differences between primary
            and verified for all dataset types. All x-axis are the same, based
            on the range from the full dataset.

            Top left: stacked histogram from all sets
            Top right: histogram from training set
            Bottom left: histogram from validation set
            Bottom right: histogram from testing set

            input params
            ------------
            diff_df (pandas.DataFrame): data with 'delta' and 'setType' columns
        '''

        ## Start plotting!
        h = plt.figure (figsize=(9, 9))
        gs = gridspec.GridSpec (2, 2, wspace=0.25, hspace=0.2)

        ## Top left: All records from train, valid, and test sets
        axis = h.add_subplot (gs[0])
        main_edges = self._plot_sub_diff (axis, diff_df)

        ## Top right: From training set
        axis = h.add_subplot (gs[1])
        self._plot_sub_diff (axis, diff_df[diff_df.setType=='train'], reqEdges=main_edges)

        ## Bottom left: From validation set
        axis = h.add_subplot (gs[2])
        self._plot_sub_diff (axis, diff_df[diff_df.setType=='validation'],
                             reqEdges=main_edges)

        ## Bottom right: From testing set
        axis = h.add_subplot (gs[3])
        self._plot_sub_diff (axis, diff_df[diff_df.setType=='test'], reqEdges=main_edges)

        ## Store plot as PDF
        title = 'Histogram of Primary - Verified at {0}'.format (self.station_id)
        plt.suptitle (title, fontsize=15)
        h.savefig ('{0}/{1}_diff_histogram.pdf'.format (self._proc_path, self.station_id))
        plt.close ('all')

    def _store_giant_hist (self, diff_df):

        ''' A private function to store histograms of differences. Note that
            the histogram values are stored instead of the arrays of differences.
            This is to save memory - 57 stations x over 1M records / stations
            is a lot of data points to store!

            The goal here is that... Per station, ths histogram is stored. The
            settings (range and nbins) are set globally across all stations.
            Then, in data_cleaner.py, the histograms are all summed up to get
            the total histogram from all stations, from which the 90% interval
            is obtained.

            Greg is interested in 3 histograms:
                * with all (good and bad) data points - centered at 0
                * with bad points only where bad is defined by target threshold 
                * with bad points only where primary ID is not the same as verified
            They are all stored as self._diff_hist, together with the histogram
            edges / bins.

            input params
            ------------
            diff_df (pandas.DataFrame): data with 'delta' and 'same_as_ver_sensor_id'
        '''

        ## Build a histogram with all data points
        points = diff_df.delta
        hist, edges = numpy.histogram (points,
                                    bins=self._diff_hist_settings['nbins'],
                                    range=self._diff_hist_settings['range'])
        self._diff_hist['all'], self._diff_hist['edges'] = hist, edges

        ## Build a histogram with only bad points based on threshold
        points = diff_df.delta [diff_df.delta.abs() > TARGET_THRESH]
        self._diff_hist['bad_only_by_thresh'] = numpy.histogram (points,
                                    bins=self._diff_hist_settings['nbins'],
                                    range=self._diff_hist_settings['range'])[0]

        ## Build a histogram with only bad points based on sensor ID
        points = diff_df.delta [~diff_df.same_as_ver_sensor_id]
        self._diff_hist['bad_only_by_sensor_id'] = numpy.histogram (points,
                                    bins=self._diff_hist_settings['nbins'],
                                    range=self._diff_hist_settings['range'])[0]

    def _handle_primary_verified_differences (self, dataframe):

        ''' A private function to handle all information related to the diff
            between primary and verified:
                * define the differences
                * store the differences as self_diff
                * generate histograms of the differences per set
                * estimate 5%, 50%, 95% values

            input params
            ------------
            dataframe (pandas.DataFrame): data with PRIMARY, VERIFIED, and setType
        '''

        ## Plot difference between PRIMARY and VERIFIED histogram
        diff_df = pandas.concat ([dataframe.PRIMARY - dataframe.VERIFIED,
                                  dataframe.SENSOR_USED_PRIMARY == dataframe.VER_WL_SENSOR_ID,
                                  dataframe.setType], axis=1)
        diff_df.columns = ['delta', 'same_as_ver_sensor_id', 'setType']
        #  Plot the histogram for this station
        if self.create_midstep_files: self.plot_diff_histogram (diff_df)
        #  Store the histogram of differences
        self._store_giant_hist (diff_df)
        #  Get the percentile stats of differences
        self._diff_stats = self._get_statistics (diff_df.delta)
        message = '   5, 50, 95 percentiles: {0:.4f}, {1:.4f}, {2:.4f}'
        self._logger.info (message.format (self._diff_stats['lower'],
                           self._diff_stats['mean'], self._diff_stats['upper']))  

    # +------------------------------------------------------------
    # | Set station meta-data from info sheet
    # +------------------------------------------------------------
    def _read_dataset_periods (self, train_period_info, valid_period_info,
                               test_period_info):

        ''' A private function to interpret the period string from station info
            sheet for each dataset type. In the info sheet, the 'Dates used for
            training', 'Dates used for Validation', and 'Dates used for testing'
            are the 3 columns that define the period of training, validation, 
            and testing sets. Each of them is expected to follow a format of
            'YYYY-mm-dd to YYYY-mm-dd'. The first date is the begin date from
            00:00, and the second date is the end date until 23:59. 

            Note: The station info sheet downloaded from google spreadsheet may
            have empty values for the 3 columns. They are automatically filled
            by data_cleaner.py when cleaning all stations. To do individual
            stations, try setting these periods via individual setters.

            input params
            ------------
            train_period_info (str): Training period 'YYYY-mm-dd to YYYY-mm-dd'
            valid_period_info (str): Validation period 'YYYY-mm-dd to YYYY-mm-dd'
            test_period_info (str): Testing period 'YYYY-mm-dd to YYYY-mm-dd'

            return params
            -------------
            periods (dict): Key-value pair indicating the dataset type (key)
                            and its begin / end periods (value)
        '''

        ## Define a holder to store the periods
        periods = {}

        ## Loop through each dataset type 
        for dtype in DATASET_TYPES:
            # Define the period from inputs - 'YYYY-mm-dd to YYYY-mm-dd'
            period_str = train_period_info if dtype=='train'      else \
                         valid_period_info if dtype=='validation' else \
                         test_period_info 
            # Try to interpret the period as timestamps
            try: 
                period = numpy.array ([pandas.to_datetime (adate.strip ())
                                       for adate in period_str.split ('to')])
                # Add 23 hours and 59 minutes to end date
                period[1] += pandas.offsets.Hour(23) + pandas.offsets.Minute(59)
            except:
                # If cannot interpret from string, this period is an empty list.
                self._logger.warn ('Cannot read period for {0} set.'.format (dtype))
                period = []
            # Add to periods dictionary
            periods[dtype] = period

        return periods

    def set_station_info (self, info_row):

        ''' A public function that allows user to set all station meta-data by
            parsing a row from the station info sheet. This method has hard-
            coded column names. So, any changes in station info sheet column
            names will result in errors - this is case sensitive! Any date
            periods are expected to be 'YYYY-mm-dd to YYYY-mm-dd' or
            'YYYY-mm-dd HH:MM to YYYY-mm-dd HH:MM'. Other values are checked
            as well to ensure valid user inputs.

            Alternatively, one can set the individual meta-data via setters.

            input params
            ------------
            info_row (pandas.DataFrame): Dataframe with only 1 row
        '''

        ## Make sure input station info row is a dataframe with only 1 row
        if not isinstance (info_row, pandas.core.frame.DataFrame):
            raise IOError ('Input station info row must be a pandas df.')
        if not len (info_row) == 1:
            raise IOError ('Input station info row must have only 1 row.')

        ## 1. Has bad results? from 'Problem station?'
        self._check_df_has_column (info_row, 'Problem station?', 'station info')
        #     This station has bad results if a string of problem is recorded
        self.has_bad_results = isinstance (info_row['Problem station?'].values[0], str)

        ## 2. Neighbor station ID from 'Neighbor station number'
        self._check_df_has_column (info_row, 'Neighbor station number', 'station info')
        self.neighbor_id = int (info_row['Neighbor station number'].values[0])

        ## 3. GT range from 'GT Range'
        self._check_df_has_column (info_row, 'GT Range', 'station info')
        self.gt_range = info_row['GT Range'].values[0]

        ## 4. WL range from 'WL Min' and 'WL Max'
        self._check_df_has_column (info_row, 'WL Min', 'station info')
        self._check_df_has_column (info_row, 'WL Max', 'station info')
        self.wl_range = (info_row['WL Min'].values[0],
                         info_row['WL Max'].values[0])

        ## 5. Periods for 3 dataset type
        self._check_df_has_column (info_row, 'Dates used for training', 'station info')
        self._check_df_has_column (info_row, 'Dates used for Validation', 'station info')
        self._check_df_has_column (info_row, 'Dates used for testing', 'station info')
        periods = self._read_dataset_periods (info_row['Dates used for training'].values[0],
                                              info_row['Dates used for Validation'].values[0],
                                              info_row['Dates used for testing'].values[0])
        self.train_dates = periods[DATASET_TYPES[0]]
        self.valid_dates = periods[DATASET_TYPES[1]]
        self.test_dates  = periods[DATASET_TYPES[2]]

        ## 6. Primary type from 'Primary sensor Type'
        self._check_df_has_column (info_row, 'Primary sensor Type', 'station info')
        self.primary_type = info_row['Primary sensor Type'].values[0].strip()

        ## 7. Other primary type from 'Other primary sensor used?'
        self._check_df_has_column (info_row, 'Other primary sensor used?', 'station info')
        other_primary_type = info_row['Other primary sensor used?'].values[0]
        #  If no string value in this cell, nothing else need to be set.
        if not isinstance (other_primary_type, str): return 
        #  Otherwise, set other primary type and its period
        self.other_primary_type = other_primary_type.strip()

        ## 8. The time period of the other primary type
        self._check_df_has_column (info_row, 'Other primary sensor dates', 'station info')
        period = info_row['Other primary sensor dates'].values[0]
        try:
            period = numpy.array ([pandas.to_datetime (adate.strip ())
                                   for adate in period.split ('to')])
        except:
            message = 'Input, {0}, cannot be read as period when setting ' + \
                      ' other primary sensor type period.'
            self._logger.fatal (message.format (period))
            raise IOError ('Use \'yyyy-mm-dd HH:MM to yyyy-mm-dd HH:MM\' format.')
        self.other_primary_type_period = period

    # +------------------------------------------------------------
    # | Load primary offset & backup gain/offset data
    # +------------------------------------------------------------
    def _transform_primary_offsets (self, offset_df):
        
        ''' A private function to check and interpret primary offsets from
            offset file. Each station has its offset file. This function makes
            sure the begin time is before the end time. In case of duplicated
            rows with the same begin and end times, only the first row is
            actually read. If everything is correct, a dictionary
            {(begin, end):[sensor ID, offset]} is returned. As of 2020/9/7, the
            sensor ID from offset file is not actually used but still stored.

            input params
            ------------
            offset_frame (pandas.DataFrame): offset info from raw offset file

            return param
            ------------
            offsets (dictionary): {(begin, end):[sensor ID, offset]} 
        '''

        ## If no offsets available, return empty dictionary.
        if len (offset_df) == 0: return {}

        ## Check the offset_frame for duplicated rows and periods of offsets.
        offsets = {}
        for row in offset_df.itertuples():
            # Extract the begin & end dates of the offset value
            begin, end = row.BEGIN_DATE_TIME, row.END_DATE_TIME
            # Make sure the end date is after begin date
            if begin > end :
                message = 'An offset period has its end date, {0}, ' + \
                          'earlier than its begin date, {1}.'
                self._logger.fatal (message.format (end, begin))
                raise IOError ('Make sure offset end dates are after the begin dates.')
            # When there are duplicated offset periods but different offset 
            # values, use the first set. For the other sets, show a warning!
            key = (begin, end)
            if key in offsets.keys():
                self._has_repeated_primary_offsets = True
                message = 'Duplicated primary offset found: {0} - {1}'
                self._logger.warn (message.format (begin, end))
                continue
            # Finally add this offset info to the dictionary. 
            offsets[(begin, end)] = [row.SENSOR_ID, row.OFFSET]

        return offsets

    def load_primary_offsets (self, primary_offset_file):

        ''' A public function that reads the input offset file and store the
            primary offset information into the class. The input file is 
            expected to be a csv file with 4 columns PRIMARY_OFFSET_KEYS. The
            offset dataframe is then checked and converted into a dictionary.

            input params
            ------------
            primary_offset_file (str): Location of raw primary offset file
        '''

        ## Make sure the path exists
        self._check_file_path_existence (primary_offset_file)

        ## Try to read file as dataframe
        try:
            offset_df = pandas.read_csv (primary_offset_file)
        except Exception:
            message = 'Failed to read file as dataframe: {0}'
            self._logger.fatal (message.format (primary_offset_file))
            raise IOError (message.format (primary_offset_file))

        ## Make sure dataframe has the required columns
        offset_df.columns = [col.strip() for col in offset_df.columns]  
        for col in PRIMARY_OFFSET_KEYS:
            if not col in offset_df:
                message = 'Cannot find column, {0}, from primary offset file.'
                raise IOError (message.format (col))

        ## Drop rows with [NULL] - Only exist in SENSOR_ID and DCP_NUM
        offset_df = offset_df.replace ('[NULL]', numpy.nan)

        ## Convert all begin date times to Timestamp. 
        offset_df['BEGIN_DATE_TIME'] = pandas.to_datetime (offset_df['BEGIN_DATE_TIME'].values)
        offset_df['END_DATE_TIME']   = pandas.to_datetime (offset_df['END_DATE_TIME'].values)

        ## Remove duplicated rows that are identical
        offset_df = offset_df.drop_duplicates()

        ## Convert it into a dictionary {(begin, end): [sensor ID, offset]}
        self._primary_offset_dict = self._transform_primary_offsets (offset_df)

    def _transform_backup_gain_offsets (self, backup_go_df):

        ''' A private function to check and interpret backup gain / offsets from
            raw file. Each station has its B1 G/O file. This function converts
            the raw dataframe into a more functional dataframe. In raw dataframe,
            gains and offsets are recorded separately with the same begin date
            time. The end date time is un-reliable and is not used. 

            Raw:
                STATION_ID,B1_DCP,PARAMETER_NAME   , ACC_PARAM_VAL,BEGIN_DATE_TIME ,END_DATE_TIME
                9462450   ,2     ,ACC_BACKUP_GAIN  , 1.384        ,2009-08-18 00:00,2010-08-04 00:00
                9462450   ,2     ,ACC_BACKUP_GAIN  , 1.369        ,2009-08-05 00:00,2009-08-17 23:59
                9462450   ,2     ,ACC_BACKUP_OFFSET, 0.517        ,2009-08-18 00:00,2010-08-04 00:00
                9462450   ,2     ,ACC_BACKUP_OFFSET, 0.407        ,2009-08-05 00:00,2009-08-17 23:59            

            This raw frame is converted into the following. Each row is a
            unique begin time with both offset and gain values to be applied.

            Converted:
                BEGIN_DATE_TIME,B1_DCP,OFFSET,GAIN
                2008-05-21     ,2     ,0.46  ,1.365
                2008-08-14     ,2     ,0.435 ,1.362

            input params
            ------------
            backup_go_df (pandas.DataFrame): backup gain / offset dataframe

            return param
            ------------
            converted_df (pandas.DataFrame): converted backup g/o dataframe
        '''

        ## Define default values and data holder
        backup_offset, backup_gain = 0, 1 # Start with no gain / offsets
        gains_and_offsets = {key:[] for key in CONVERTED_BACKUP_GO_KEYS}

        ## Loop through each begin time in input dataframe
        for date_time in sorted (backup_go_df['BEGIN_DATE_TIME'].unique()):

            # Extract rows with this begin time 
            subframe = backup_go_df[backup_go_df['BEGIN_DATE_TIME'] == date_time]
            if len (subframe) > 2:
                message = 'Multiple entries at date-time, {0}, are found ' + \
                          'for backup gain & offset.\nTaking the first ' + \
                          'entry as the official gain & offset.'
                self._logger.warn (message.format (date_time))

            # Determine the offset value
            if len (subframe[subframe.IS_OFFSET]) > 0:
                this_offset = subframe[subframe.IS_OFFSET].ACC_PARAM_VAL.values[0]
                # If the offset value for this date time is invlid, use previous offset value.
                if not numpy.isfinite (this_offset):
                    message = 'Invalid backup offset value is found at {0}. ' + \
                              'Using previous offset value {1}.'
                    self._logger.warn (message.format (date_time, backup_offset))
                # When valid offset is found, update backup_offset.
                if numpy.isfinite (this_offset): backup_offset = this_offset 

            # Determine the gain value
            if len (subframe[subframe.IS_GAIN]) > 0:
                this_gain = subframe[subframe.IS_GAIN].ACC_PARAM_VAL.values[0]
                # If the gain value for this date time is invlid, use previous gain value.
                if not numpy.isfinite (this_gain):
                    message = 'Invalid backup gain value is found at {0}. ' + \
                              'Using previous gain value {1}.'
                    self._logger.warn (message.format (date_time, backup_gain))
                # When valid gain is found, update backup_gain.
                if numpy.isfinite (this_gain): backup_gain = this_gain
            
            # Append this gain / offset to the dictionary
            gains_and_offsets['BEGIN_DATE_TIME'].append (pandas.to_datetime (date_time))
            gains_and_offsets['B1_DCP'].append (subframe.B1_DCP.values[0])
            gains_and_offsets['OFFSET'].append (backup_offset)
            gains_and_offsets['GAIN'].append (backup_gain)

        ## Convert the dictionary into a dataframe
        return pandas.DataFrame (gains_and_offsets) 

    def load_backup_B1_gain_offsets (self, backup_go_file):

        ''' A public function to read an input backup gain and offset file,
            convert it into a more ready dataframe, and then store the data-
            frame into the class. If set to dump intermediate files, this
            function will also dump the converted dataframe into a csv file.

            input params
            ------------
            backup_go_file (str): Location of raw backup gain/offset file
        '''

        ## Make sure the path exists
        self._check_file_path_existence (backup_go_file)

        ## Try to read file as dataframe
        try:
            backup_go_df = pandas.read_csv (backup_go_file)
        except Exception:
            message = 'Failed to read file as dataframe: {0}'
            self._logger.fatal (message.format (backup_go_file))
            raise IOError (message.format (backup_go_file))

        ## Make sure dataframe has the required columns
        backup_go_df.columns = [col.strip() for col in backup_go_df.columns]
        for col in RAW_BACKUP_GO_KEYS:
            if not col in backup_go_df:
                message = 'Cannot find column, {0}, from backup g/o file.'
                raise IOError (message.format (col))
        
        ## Drop rows with [NULL] - Same date-time has NULL values
        backup_go_df = backup_go_df.replace ('[NULL]', numpy.nan)

        ## Convert PRAMETER_NAME into gains / offsets array
        backup_go_df['IS_GAIN']   = backup_go_df['PARAMETER_NAME'] == 'ACC_BACKUP_GAIN'
        backup_go_df['IS_OFFSET'] = backup_go_df['PARAMETER_NAME'] == 'ACC_BACKUP_OFFSET'

        ## Convert all begin date times to Timestamp.
        ## End dates for backup offset / gain are unreliable.
        backup_go_df['BEGIN_DATE_TIME'] = pandas.to_datetime (backup_go_df.BEGIN_DATE_TIME.values)
        backup_go_df = backup_go_df.sort_values (by='BEGIN_DATE_TIME').reset_index()

        ## Drop duplicated rows that are identical
        backup_go_df = backup_go_df.drop_duplicates()

        ## Convert this frame into unique time per row with both gain and offset info
        self._backup_gain_offset_df = self._transform_backup_gain_offsets (backup_go_df)

        ## Dump a midstep file if asked
        self._dump_file ('Backup gain & offset', 'backup_gain_offsets',
                         self._backup_gain_offset_df)

    # +------------------------------------------------------------
    # | Load raw data
    # +------------------------------------------------------------
    def _check_start_end_dates (self, dataframe):

        ''' A private function that chops the raw dataframe based on training
            and testing periods. The input raw data should starts on the train
            begin date-time and ends on the test end date-time. 

            If the dataframe starts before train begin time, rows before train
            begin time are dropped. If the dataframe starts after train begin
            time, training begin time is adjusted with a warning.

            If the dataframe ends before test end time, test end time is adjusted
            with a warning. If the dataframe ends after test end time, rows
            after test end time are dropped.

            input params
            ------------
            dataframe (pandas.DataFrame): raw data

            output params
            -------------
            dataframe (pandas.DataFrame): data after adjustment 
        '''

        ## Make sure periods for training, validation, and testing are set
        if self._train_dates is None or self._valid_dates is None or self._test_dates is None:
            raise IOError ('Please set train/valid/test periods before loading data.')

        ## 1. Check if dataframe period start date is before training start date
        if dataframe.index[0] < self._train_dates[0]:
            message = 'Raw start date, {0}, is before training set start date, {1}. Resetting raw data ..'
            self._logger.warn (message.format (dataframe.index[0], self._train_dates[0]))
            dataframe = dataframe[dataframe.index >= self._train_dates[0]]

        ## 2. Check if dataframe period start date is after training start date
        if dataframe.index[0] > self._train_dates[0]:
            message = 'Raw start date, {0}, is after training set start date, {1}. Resetting training start date ..'
            self._logger.warn (message.format (dataframe.index[0], self._train_dates[0]))
            self._train_dates[0] = dataframe.index[0]

        last_dates = self._test_dates  if not len (self._test_dates) == 0 else \
                     self._valid_dates if not len (self._valid_dates) == 0 else \
                     self._train_dates

        ## 3. Check if dataframe period end date is before testing end date
        if dataframe.index[-1] < last_dates[-1]:
            message = 'Raw end date, {0}, is before testing set end date, {1}. Resetting testing end date ..'
            self._logger.warn (message.format (dataframe.index[-1], last_dates[-1]))
            last_dates[-1] = dataframe.index[-1]

        ## 4. Check if dataframe period end date is after testing end date
        if dataframe.index[-1] > last_dates[-1]:
            message = 'Raw end date, {0}, is before testing set end date, {1}. Resetting raw data ..'
            self._logger.warn (message.format (dataframe.index[-1], last_dates[-1]))
            dataframe = dataframe[dataframe.index <= last_dates[-1]]

        return dataframe

    def _divide_raw_into_3_sets (self, dataframe):
    
        ''' A private function to divide raw data into 3 sets based on the
            train / validation / test period. This function creates a new column
            setType ('train', 'validation', 'test') indicating the dataset type.

            input params
            ------------
            dataframe (pandas.DataFrame): data without setType column

            output params
            -------------
            dataframe (pandas.DataFrame): data with setType column
        '''

        ## Make sure periods for training, validation, and testing are set
        if self._train_dates is None or self._valid_dates is None or self._test_dates is None:
            raise IOError ('Please set train/valid/test periods before loading data.')

        ## Create a list holder to store the dataset type per row
        setType = []

        ## Loop through each dataset type
        periods = [self._train_dates, self._valid_dates, self._test_dates]
        for dtype, period in zip (DATASET_TYPES, periods):
            # If no period for this set, skip.
            if len (period) == 0: continue
            # Extract the dataframe slices within this period
            thisframe = dataframe.loc[period[0]:period[1]]
            # Assign dataset type name
            setType += [dtype] * len (thisframe)
            # Log it
            message = '{0} rows between {1} and {2} in {3}.'
            self._logger.info (message.format (len (thisframe), period[0], period[1], dtype))
        
        ## Add new column indicating dataset type
        dataframe['setType'] = setType
        return dataframe

    def _handle_duplicated_timestamps_in_raw_file (self, dataframe):

        ''' A private function to handle duplicated timestamps in raw file. In
            our database, some stations have repeated timestamps that have diff
            observed values. This should be cleaned up in the database, but it
            is easier to handle it at the beginning of the cleaning process.

            The rule of thumb is that
                * Do not use the row if there is any -99999.999 value
                * Use the first good row as the official data 

            input params
            ------------
            dataframe (pandas.DataFrame): data with duplicated timestamps

            output params
            -------------
            dataframe (pandas.DataFrame): data without duplicated timestamps
        '''

        ## Store the flag whether each set has duplicated time stamps
        self._train_stats['has_repeated_raw'] = has_repeated_raw (dataframe[dataframe.setType == 'train'])
        self._validation_stats['has_repeated_raw'] = has_repeated_raw (dataframe[dataframe.setType == 'validation'])
        self._test_stats['has_repeated_raw']  = has_repeated_raw (dataframe[dataframe.setType == 'test'])    

        ## Extract a list of repeated date-times 
        repeated_times = dataframe.DATE_TIME.value_counts()[dataframe.DATE_TIME.value_counts ()>1].index    
        ## If no repeated date-times, nothing needs to be done.
        if len (repeated_times) == 0: return dataframe

        ## Log the repeated times found
        self._logger.debug ('This station has {0} repeated times at ...'.format (len (repeated_times)))

        ## Loop through each repeated times and decide which one is the official
        ## one and collect the indices of those that need to be removed
        remove_row_indices = []
        for repeated_time in repeated_times:
            self._logger.debug ('     * {0}'.format (repeated_time))
            found_good = False
            subframe = dataframe[dataframe.DATE_TIME == repeated_time]
            # For a given repeated time, loop through all duplicated rows
            for index, row in subframe.iterrows():
                self._logger.debug ('        - row {0}'.format (index))
                # If this row has any -99999.999 value in any columns,
                # this row must be removed.
                any999 = [row[col] == -99999.999 for col in row.index]
                if numpy.array (any999).any():
                    self._logger.debug ('          -99999.999 is found!')
                    remove_row_indices.append (index)
                    continue
                ## If all values are valid, use the first found row as official.
                if not found_good:
                    self._logger.debug ('          This row is the official data for this timestamp!')
                    found_good = True
                    continue
                ## For any 2nd, 3rd, 4th, etc good rows, they are removed
                self._logger.debug ('          Good data exists for timestamp - removing this one!')
                remove_row_indices.append (index)

        ## Return a dataframe with repeated timestamps row dropped
        return dataframe.drop (axis=0, index=remove_row_indices)

    def _redefine_backup_data_in_raw_file (self, dataframe):

        ''' A private function to re-define backup B1_WL_VALUE_MSL in raw data.
            The backup data are re-calibrated using the gain and offset info
            from B1_gain_offset csv file. In general, 
                B1_WL_VALUE_MSL = B1_WL_VALUE * gain + offset - MSL

            For each set of gain and offset values, there is a begin time.
            Records between the begin time of one set and right before the
            begin time of the next set have the backup value re-calibrated by
            that set of gain and offset values if and only if the record has
            the same DCP as that for the gain / offset.

            If the record does not have a valid gain / offset values (with the
            same DCP), the record does not have a valid backup value, and so a
            nan is given.

            input params
            ------------
            dataframe (pandas.DataFrame): data with wrong B1_WL_VALUE_MSL

            output params
            -------------
            dataframe (pandas.DataFrame): data with re-calibrated B1_WL_VALUE_MSL.
        '''

        ## Make sure backup gain/offsets are defined properly
        if self._backup_gain_offset_df is None:
            raise IOError ('Please provide backup B1 gain & offset data before loading raw data.')

        ## Loop through each gain / offset rows and re-define backup value.
        ## B1_WL_VALUE is the raw-est backup data from the database. 
        for index, row in enumerate (self._backup_gain_offset_df.itertuples()):
            # End dates from B1_gain_offsets file are not trusted. Hence, it is
            # not included in self._backup_gain_offset_df. Instead, the end date
            # of a given g/o set is defined to be 1 minute before the next g/o
            # set. If this g/o set is the last one, then its end date is set to
            # be 2100-12-31 i.e. forever from now.
            is_last = index == len (self._backup_gain_offset_df) - 1
            end_date = pandas.to_datetime ('2100-12-31') if is_last else \
                       self._backup_gain_offset_df.iloc[index+1].BEGIN_DATE_TIME - pandas.Timedelta (minutes=1)
            offset_period = (row.BEGIN_DATE_TIME, end_date)
            # Backup gain & offsets are only applied to row records with the same DCP
            same_DCP = dataframe.loc[offset_period[0]:offset_period[1]].B1_DCP == row.B1_DCP
        
            # 1. For records within the offset period and with the same DCP
            #    Their backup B1 is re-calculated with gain and offset
            subframe = dataframe.loc[offset_period[0]:offset_period[1]][same_DCP]
            if len (subframe) > 0:
                # New backup = raw B1 x gain + offset - MSL
                backup_values = subframe.B1_WL_VALUE * row.GAIN + row.OFFSET - subframe.B1_MSL
                dataframe.loc[offset_period[0]:offset_period[1], 'B1_WL_VALUE'][same_DCP] = backup_values.values

            # 2. For records within the offset period but with different DCP. Their backup
            #    B1 is set to nan. B1 data was likely bad given no valid g/o are available.
            subframe = dataframe.loc[offset_period[0]:offset_period[1]][~same_DCP]
            if len (subframe) > 0:
                dataframe.loc[offset_period[0]:offset_period[1], 'B1_WL_VALUE'][~same_DCP] = numpy.NaN
        
        ## Re-define backup B1 value by the new one
        dataframe['B1_WL_VALUE_MSL'] = dataframe.B1_WL_VALUE
        ## Remove other backup columns
        dataframe = dataframe.drop (axis=1, columns=['B1_WL_VALUE', 'B1_MSL'])
        return dataframe

    def _load_raw_data (self):

        ''' A private function to load raw data with a few pre-cleaning steps.
                1. raw data is turned into a time-series dataframe
                2. adjust either dataframe or train/test times if needed
                3. divide the dataframe into 3 sets: train, valid, test
                4. handle duplicated timestamps in dataframe
                5. redefine backup B1_WL_VALUE_MSL

            return params
            -------------
            dataframe (pandas.DataFrame): raw data after pre-cleaning steps
        '''

        ## Make sure raw file is defined
        if self._raw_file is None:
            raise IOError ('Please provide raw file location first.')

        ## Read the csv file
        dataframe = pandas.read_csv (self._raw_file, low_memory=False)
        self._logger.info ('Raw file {0} is successfully read.'.format (os.path.basename (self._raw_file)))
        n_raw = len (dataframe)
        self._logger.info ('{0} records are found.'.format (n_raw))

        ## Turn dataframe into a time-series dataframe
        dataframe['DATE_TIME'] = pandas.to_datetime (dataframe.DATE_TIME)
        dataframe.index = dataframe.DATE_TIME
        dataframe = dataframe.sort_index()
        self._logger.info ('Dataframe is turned into a time-series dataframe.')

        ## Check raw data time with training start and testing end dates
        dataframe = self._check_start_end_dates (dataframe)

        ## Divide dataframe into 3 sets: train / valid / test based on timestamps
        ## This is Step 6 in WL-AI Station File Requirements
        dataframe = self._divide_raw_into_3_sets (dataframe)
        self._set_stats (dataframe, 'n_raw')
        self._logger.info ('Raw data is divided into train / valid / test dataset.')

        ## Setting stats with has-bad-results per set
        self._train_stats['has_bad_results'] = self._has_bad_results
        self._validation_stats['has_bad_results'] = self._has_bad_results
        self._test_stats['has_bad_results'] = self._has_bad_results

        ## Handle repeated rows due to duplicated date-times 
        dataframe = self._handle_duplicated_timestamps_in_raw_file (dataframe)
        self._logger.info ('{0} rows remain after removing duplicated timestamps.'.format (len (dataframe)))

        ## Massage backup data based on gain & offsets from B1 file
        ## This is Step 7 in WL-AI Station File Requirements
        dataframe = self._redefine_backup_data_in_raw_file (dataframe)
        self._logger.info ('Backup data is re-set based on B1 gain & offset.')

        self._logger.info ('{0} records in total are found.'.format (len (dataframe)))
        # Keep track of stats
        self._set_stats (dataframe, 'n_total')          
        return dataframe

    # +------------------------------------------------------------
    # | Clean data
    # +------------------------------------------------------------
    def _set_primary_sensor_type_stats (self, dataframe):

        ''' A private function to set the counts of primary sensor and other
            primary sensor per dataset type based on SENSOR_USED_PRIMARY column.

            input params
            ------------
            dataframe (pandas.DataFrame): data with SENSOR_USED_PRIMARY column
        '''

        self._set_stats (dataframe[dataframe['SENSOR_USED_PRIMARY'] == self._primary_type],
                         'n_with_primary_sensor')
        self._set_stats (dataframe[dataframe['SENSOR_USED_PRIMARY'] == self._other_primary_type],
                         'n_with_other_primary_sensor')

    def _define_sensor_used (self, dataframe):

        ''' A private function that define SENSOR_USED_PRIMARY by station info
            sheet. By default, all records have the sensor ID of primary type. 
            If other primary type is available, records within the period are
            changed to other primary type.

            input params
            ------------
            dataframe (pandas.DataFrame): dataframe without SENSOR_USED_PRIMARY

            output params
            -------------
            dataframe (pandas.DataFrame): dataframe with SENSOR_USED_PRIMARY
        '''

        ## By default, primary sensor type for all row records are the
        ## primary type 'Primary sensor Type' in the station info sheet.
        self._logger.info ('    * Default primary type is {0}.'.format (self._primary_type))
        dataframe['SENSOR_USED_PRIMARY'] = [self._primary_type] * len (dataframe)

        ## If there is no other primary types (indicated in station info sheet),
        ## nothing else needs to be done. 
        if self._other_primary_type is None:
            self._set_primary_sensor_type_stats (dataframe)
            return dataframe

        ## When there is another primary type, re-define the primary type
        ## for the records within the time period.
        period = self._other_primary_type_period
        dataframe.loc[period[0]:period[1], 'SENSOR_USED_PRIMARY'] = self._other_primary_type

        ## Just log the number of records with this change of primary sensor type
        nChanged = len (dataframe[dataframe.SENSOR_USED_PRIMARY == self._other_primary_type])
        self._logger.info ('    * {0} records are changed from {1} to {2}.'.format (nChanged, self._primary_type,
                                                                                    self._other_primary_type))

        ## Count the number of rows with primary and other primary sensors
        self._set_primary_sensor_type_stats (dataframe)
        return dataframe

    def _apply_offsets_on_primary (self, dataframe):
        
        ''' A private function that takes the dataframe and applies offsets to
            PRIMARY. This function loops through each set of offsets and apply
            the corresponding offset values to a sub-set of dataframe defined
            by the corresponding offset period if the verified WL sensor ID
            from the raw file matches the primary sensor ID defined previously.
            The sensor ID from the offset file is not actually used here.

            input params
            ------------
            dataframe (pandas.DataFrame): dataframe before primary offsets

            return param
            ------------
            dataframe (pandas.DataFrame): dataframe after primary offsets
        '''

        ## Initialize a boolean column indicating if offsets are applied
        dataframe['OFFSETS_APPLIED'] = False
        ## If no offsets available, do nothing to time-series dataframe
        if len (self._primary_offset_dict) == 0:
            self._set_stats (dataframe[dataframe['OFFSETS_APPLIED']],
                             'n_primary_offsets_applied')
            return dataframe

        ## Loop through each available offset periods
        for period, (sensor_id, offset_value) in self._primary_offset_dict.items():
            # Extract the slice based on period and modify PRIMARY column. Only
            # apply offsets if the verified sensor ID from the raw statoin file
            # is the same as the previously defined primary sensor ID. i.e. do
            # not make use of the sensor_id column in offset file.
            is_sensor = dataframe.loc[period[0]:period[1], 'VER_WL_SENSOR_ID'] == \
                        dataframe.loc[period[0]:period[1], 'SENSOR_USED_PRIMARY']
            self._logger.info ('    * +---------------------------------------------------------')
            self._logger.info ('    * | {0} - {1}'.format (period[0], period[1]))
            self._logger.info ('    * |   {0} records are found with matching sensor ID'.format (len (is_sensor[is_sensor])))
            # If no matching sensor, continue to the next set of offset.
            if len (is_sensor[is_sensor]) == 0: continue
            # Apply offset for specific rows
            dataframe.loc[period[0]:period[1], 'PRIMARY'][is_sensor] += offset_value
            dataframe.loc[period[0]:period[1], 'OFFSETS_APPLIED'][is_sensor] = True
            self._logger.info ('    * |   Offset value of {0:.5f} is added to those records'.format (offset_value))

        self._logger.info ('    * +---------------------------------------------------------')
        nApply = len (dataframe[dataframe.OFFSETS_APPLIED])
        self._logger.info ('    * Offsets are applied to a total of {0} records'.format (nApply))

        #  Count the number of rows with primary offsets applied
        self._set_stats (dataframe[dataframe['OFFSETS_APPLIED']], 'n_primary_offsets_applied')
        return dataframe

    def _replace_nan (self, dataframe):
        
        ''' A private function that checks for invalid entries in the columns:
                * PRIMARY, PRIMARY_SIGMA, PRIMARY_RESIDUAL
                * BACKUP, BACKUP_SIGMA, BACKUP_RESIDUAL
            Invalid values are replaced by 0.0. For PRIMARY, PRIMARY_SIGMA,
            BACKUP, and BACKUP_SIGMA, a _TRUE column is added to indicate
            whether its original values are invalid.

            input params
            ------------
            dataframe (pandas.DataFrame): data before TRUE columns are added

            return param
            ------------
            dataframe (pandas.DataFrame): data after TRUE columns are added
        '''

        ## Loop over PRIMARY, PRIMARY_SIGMA, PRIMARY_RESIDUAL, BACKUP,
        ## BACKUP_SIGMA, and BACKUP_RESIDUAL
        for main in ['PRIMARY', 'BACKUP']:
            for suffix in ['', '_SIGMA', '_RESIDUAL']:
                # Reconstruct the actual column name
                key = main + suffix
                # Which row has a valid value for the key?
                # 1 = is valid; 0 = is invalid
                isValid = (~dataframe[key].isna ()).astype (int)
                # Add in _TRUE column except RESIDUAL
                if not 'RESIDUAL' in key: dataframe[key + '_TRUE'] = isValid
                # Print on console if any invalid values are found and replaced
                nInvalid = len (isValid[~isValid.astype (bool)])
                if nInvalid > 0:
                    self._logger.info ('    * {0} {1} values are reset to 0.0'.format (nInvalid, key))
                # Replace the invalid value by 0.0
                dataframe.loc[~isValid.astype (bool), key] = 0.0

        ## Count the number of nan & missing primary, primary sigma, backup, and
        ## backup sigma. The nan residuals are reflected from primary and backup.
        self._set_stats (dataframe[dataframe['PRIMARY_TRUE'] == 0], 'n_nan_primary')
        self._set_stats (dataframe[dataframe['PRIMARY_SIGMA_TRUE'] == 0], 'n_nan_primary_sigma')
        self._set_stats (dataframe[dataframe['BACKUP_TRUE'] == 0], 'n_nan_backup')
        self._set_stats (dataframe[dataframe['BACKUP_SIGMA_TRUE'] == 0], 'n_nan_backup_sigma')
        return dataframe

    def _cap_values (self, dataframe):
        
        ''' A private function to replace any PRIMARY and BACKUP water level
            that are beyond an accepted range of water levels. The water level
            thresholds (min, max) are defined in the station list file. Their
            SIGMAs are must also be within 0 and 1. Any values that are beyond
            the accepted range are replaced by either the max or min values.

            input params
            ------------
            dataframe (pandas.DataFrame): dataframe before capping

            return param
            ------------
            dataframe (pandas.DataFrame): dataframe after capping
        '''

        ## 1. Cap PRIMARY & BACKUP water level
        for key in ['PRIMARY', 'BACKUP']:
            # Count the number of capped value per set.
            self._set_stats (dataframe[dataframe[key] < self._wl_range[0]],
                             'n_capped_' + key.lower() + '_min')
            self._set_stats (dataframe[dataframe[key] > self._wl_range[1]],
                             'n_capped_' + key.lower() + '_max')
            # Log info out
            nBeyondMin = len (dataframe[dataframe[key] < self._wl_range[0]])
            nBeyondMax = len (dataframe[dataframe[key] > self._wl_range[1]])
            self._logger.info ('    * {0} records have {1} below min value of {2}'.format (nBeyondMin, key, self._wl_range[0]))
            self._logger.info ('    * {0} records have {1} above max value of {2}'.format (nBeyondMax, key, self._wl_range[1]))
            # Apply the capping
            dataframe.loc[dataframe[key] < self._wl_range[0], key] = self._wl_range[0]
            dataframe.loc[dataframe[key] > self._wl_range[1], key] = self._wl_range[1]

        ## 2. Cap PRIMARY_SIGMA & BACKUP_SIGMA between 0 and 1
        for key in ['PRIMARY_SIGMA', 'BACKUP_SIGMA']:
            # Count the number of capped value per set.
            self._set_stats (dataframe[dataframe[key] < 0], 'n_capped_' + key.lower() + '_min')
            self._set_stats (dataframe[dataframe[key] > 1], 'n_capped_' + key.lower() + '_max')
            # Log info out            
            nBeyondMin = len (dataframe[dataframe[key] < 0])
            nBeyondMax = len (dataframe[dataframe[key] > 1])
            self._logger.info ('    * {0} records have {1} below min value of 0'.format (nBeyondMin, key))
            self._logger.info ('    * {0} records have {1} above max value of 1'.format (nBeyondMax, key))
            # Apply the capping
            dataframe.loc[dataframe[key] < 0, key] = 0                
            dataframe.loc[dataframe[key] > 1, key] = 1

        return dataframe

    # def _scale_values (self, dataframe):
        
    #     ''' A private function that scales PRIMARY, VERIFIED, BACKUP, and
    #         PREDICTION by GT range stated in the station list. 

    #         input params
    #         ------------
    #         dataframe (pandas.DataFrame): dataframe before GT scaling

    #         return param
    #         ------------
    #         dataframe (pandas.DataFrame): dataframe after GT scaling
    #     '''

    #     # Read GT range from station csv
    #     self._logger.info ('    * GT range is {0}'.format (self._gt_range))
    #     # Scale each column
    #     for key in ['PRIMARY', 'VERIFIED', 'BACKUP', 'PREDICTION']:
    #         dataframe[key] = dataframe[key] / self._gt_range

    #     return dataframe

    def clean_raw_data (self, exclude_nan_verified=False):

        ''' A public function that clean raw data. This follows step 8-17 in
            https://docs.google.com/document/d/1BfyIQE9GXPCRbBSkyurd3UeGqpGkAr1UYkMZzh5LBNk/edit?usp=sharing

            As of 2020/09/07, Elim was investigating data issue and found that
            using VER_WL_SENSOR_ID column instead of station info sheet reduces
            the number of spikes in a dozen of stations. So, this is an
            experimental function until a decision is made to move this forward
            or not. 

            input params
            ------------
            use_VER_SENSOR_ID (bool): If true, use VER_WL_SENSOR_ID to define
                                      SENSOR_USED_PRIMARY. By default, use
                                      station sheet to define primary sensor.
            exclude_nan_VER (bool): Exclude NaN VER_WL_VALUE_MSL when counting spikes
        '''

        self._logger.info ('+-------------------------------')
        self._logger.info ('|  Start Cleaning ')
        self._logger.info ('+-------------------------------')
        ## Read raw data
        dataframe = self._load_raw_data () 

        ## Add SENSOR_USED_PRIMARY column from station list
        #  This is Step 8 in WL-AI Station File Requirements  
        self._logger.info ('1. Define SENSOR_USED_PRIMARY column')
        dataframe = self._define_sensor_used (dataframe)

        ## Define PRIMARY water level based on SENSOR_USED_PRIMARY
        #  This is Step 9 in WL-AI Station File Requirements
        self._logger.info ('2. Define PRIMARY water level based on SENSOR_USED_PRIMARY')
        dataframe['PRIMARY'] = dataframe.apply (get_primaries, axis=1)

        ## Apply offsets to PRIMARY water level
        #  This is Step 10 in WL-AI Station File Requirements
        self._logger.info ('3. Apply offset to PRIMARY')
        dataframe = self._apply_offsets_on_primary (dataframe)

        ## Add PRIMARY_SIGMA column i.e. A1_WL_SIGMA
        #  This is Step 11 in WL-AI Station File Requirements
        self._logger.info ('4. Define PRIMARY_SIGMA')
        dataframe['PRIMARY_SIGMA'] = dataframe.apply (get_primary_sigmas, axis=1)

        ## Add BACKUP & BACKUP_SIGMA 
        #  This is Step 12 in WL-AI Station File Requirements
        self._logger.info ('5. Define BACKUP & BACKUP_SIGMA')
        dataframe['BACKUP'] = dataframe.B1_WL_VALUE_MSL
        dataframe['BACKUP_SIGMA'] = dataframe.B1_WL_SIGMA
        
        ## Cap PRIMARY & BACKUP between WL_MIN & WL_MAX and their SIGMAs between 0 and 1.
        #  _cap_values() automatically counts the number of capped primary, backup, and their sigmas.
        ## This is Step 13 in WL-AI Station File Requirements
        self._logger.info ('6. Cap PRIMARY & BACKUP and their SIGMAs')
        dataframe = self._cap_values (dataframe)  

        ## Add PRIMARY_RESIDUAL i.e. PRIMARY - PRED
        #  This is Step 14 in WL-AI Station File Requirements
        self._logger.info ('7. Define PRIMARY_RESIDUAL')
        dataframe['PRIMARY_RESIDUAL'] = dataframe.PRIMARY - dataframe.PRED_WL_VALUE_MSL

        ## Add BACKUP_RESIDUAL 
        #  This is Step 15 in WL-AI Station File Requirements
        self._logger.info ('8. Define BACKUP_RESIDUAL')
        dataframe['BACKUP_RESIDUAL'] = dataframe.B1_WL_VALUE_MSL - dataframe.PRED_WL_VALUE_MSL

        # ## Add PRIMARY_SIGMA column i.e. A1_WL_SIGMA & PRIMARY_RESIDUAL i.e. PRIMARY - PRED
        # #  This is Step xx in WL-AI Station File Requirements
        # self._logger.info ('5. Define PRIMARY_SIGMA & PRIMARY_RESIDUAL')
        # dataframe['PRIMARY_SIGMA'] = dataframe.apply (get_primary_sigmas, axis=1)
        # dataframe['PRIMARY_RESIDUAL'] = dataframe.PRIMARY - dataframe.PRED_WL_VALUE_MSL

        # ## Add BACKUP, BACKUP_SIGMA, BACKUP_RESIDUAL 
        # #  This is Step xx in WL-AI Station File Requirements
        # self._logger.info ('6. Define BACKUP, BACKUP_SIGMA, & BACKUP_RESIDUAL')
        # dataframe['BACKUP'] = dataframe.B1_WL_VALUE_MSL
        # dataframe['BACKUP_SIGMA'] = dataframe.B1_WL_SIGMA
        # dataframe['BACKUP_RESIDUAL'] = dataframe.B1_WL_VALUE_MSL - dataframe.PRED_WL_VALUE_MSL

        ## Add PREDICTION & VERIFIED
        #  This is Step 16 in WL-AI Station File Requirements
        self._logger.info ('9. Define VERIFIED, VERIFIED_RESIDUAL, VERIFIED_SENSOR_ID & PREDICTION')
        dataframe['VERIFIED'] = dataframe.VER_WL_VALUE_MSL
        dataframe['VERIFIED_RESIDUAL'] = dataframe.VER_WL_VALUE_MSL - dataframe.PRED_WL_VALUE_MSL
        # dataframe['PRESCALED_VERIFIED'] = dataframe.VER_WL_VALUE_MSL
        dataframe['VERIFIED_SENSOR_ID'] = dataframe.VER_WL_SENSOR_ID
        dataframe['PREDICTION'] = dataframe.PRED_WL_VALUE_MSL

        ## Count the # records
        #    .. with invalid verified
        self._set_stats (dataframe[dataframe.VER_WL_VALUE_MSL.isna()], 'n_nan_verified')        
        #    .. with valid primary but nan verified
        is_bad = numpy.logical_and (~dataframe.PRIMARY.isna(), 
                                    dataframe.VER_WL_VALUE_MSL.isna())
        self._set_stats (dataframe[is_bad], 'n_nan_verified_valid_primary')
        #    .. with nan primary and nan verified
        is_bad = numpy.logical_and (dataframe.PRIMARY.isna(), 
                                    dataframe.VER_WL_VALUE_MSL.isna())
        self._set_stats (dataframe[is_bad], 'n_nan_verified_nan_primary')

        ## Add TARGET based on target threshold between PRIMARY and VER_WL_VALUE_MSL
        #  This is Step 17 in WL-AI Station File Requirements
        self._logger.info ('10. Define TARGET with threshold value of {0} meters'.format (TARGET_THRESH))
        dataframe['TARGET'] = ((dataframe.PRIMARY - dataframe.VER_WL_VALUE_MSL).abs() <= TARGET_THRESH).astype (int)
        #  Count the number of spikes per set and in total
        is_spikes = numpy.logical_and (dataframe.TARGET==0,  ~dataframe.PRIMARY.isna())
        self._logger.info ('   {0} records are identified as target spikes'.format (len (dataframe[is_spikes])))    
        #  Exclude those with nan VER_WL_VALUE_MSL when counting n_spikes
        if exclude_nan_verified:
            is_spikes = numpy.logical_and (is_spikes, ~dataframe.VER_WL_VALUE_MSL.isna())
            message = '   {0} records are identified as target spikes after excluding nan VER'
            self._logger.info (message.format (len (dataframe[is_spikes])))    
        self._set_stats (dataframe[is_spikes], 'n_spikes')
        
        ## Plot difference between PRIMARY and VERIFIED histogram
        self._handle_primary_verified_differences (dataframe)

        ## For PRIMARY, BACKUP, and their SIGMAs & RESIDUALs, replace all NaNs / missing
        #  entries with 0 and create dummy boolean column with _TRUE suffix in column names
        #  This is Step 18 in WL-AI Station File Requirements
        self._logger.info ('11. Replace NaNs with 0 and dummy column for PRIMARY, BACKUP and their SIGMA, RESIDUAL')    
        dataframe = self._replace_nan (dataframe)     

        # ## Scale PRIMARY, BACKUP, and PREDICTION by GT range
        # ## This is Step 19 in WL-AI Station File Requirements
        # self._logger.info ('10. Scale PRIMARY, VERIFIED, BACKUP, and PREDICTION by GT range.')    
        # dataframe = self._scale_values (dataframe)

        # Keep columns requested in specific order
        return dataframe[CLEANED_COLUMNS + ['setType']]
