#!python37

## By Elim Thompson (09/02/2020)
##
## This script defines a data_cleaner class that manages the cleaning process
## on all available stations. These stations are listed on the WL-AI Station
## List (i.e. station info sheet):
##  https://docs.google.com/spreadsheets/d/1tLoaNPWNnHneWOZlpS38S7ldlkSs0wiCq_E6_39u_Qg/edit?usp=sharing
## 
## This class is used by clean_data.py script which performs cleaning on all
## stations. Please visit that script first before reading through this one.
##
## Prior calling data_cleaner.py, the following information must be set
##  * Download the latest WL-AI Station List from the URL above. This file
##    must be stored as CSV file. The full path and filename will be parsed
##    as station_info_csv in this class.
##  * Copy Armin's raw file from CO-OPS Common ..
##      N:\CO-OPS_Common\CODE\AI-data-retrieval\data
##    to your local desktop. Unzip all files to a location. This folder
##    location should be parsed as raw_path in this class.
##    NOTE: the raw_ver_merged_wl.csv, offsets.csv, and B1_gain_offsets.csv
##          must all be located in the same folder
##
## Example snippet to use data_cleaner class
## +------------------------------------------------------------------
## # Initialize a new data cleaner 
## cleaner = data_cleaner.data_cleaner()
##
## # Set up the cleaner using input arguments
## cleaner.raw_path = 'C:/path/to/Armin/unzipped/raw/files/'
## cleaner.proc_path = 'C:/path/to/store/processed/files/'
## cleaner.station_info_csv = 'C:/location/of/station_info_sheet.csv'
## cleaner.create_midstep_files = True
##
## # Load station info
## cleaner.load_station_info()
##    
## # Clean all stations
## cleaner.clean_stations ()
##
## # Save stats data (if not already) 
## cleaner.save_stats_data()
## +------------------------------------------------------------------
##
#############################################################################

###############################################
## Import libraries
###############################################
import numpy, pandas, datetime, os, logging
from scipy.interpolate import interp1d
import _pickle as pickle
from glob import glob

import station

import matplotlib
matplotlib.use ('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import AnchoredText
plt.rc ('text', usetex=False)
plt.rc ('font', family='sans-serif')
plt.rc ('font', serif='Computer Modern Roman')

###############################################
## Define constants
###############################################
# Final columns to be included in the same exact order
CLEANED_COLUMNS = station.CLEANED_COLUMNS
# Additional columns related to neighbor info
NEIGHBOR_COLUMNS = ['NEIGHBOR_PRIMARY', 'NEIGHBOR_PREDICTION',
                    'NEIGHBOR_PRIMARY_RESIDUAL', 'NEIGHBOR_TARGET']

# File name pattern of Armin's raw files
FILE_PATTERN_RAW_CSV = '_raw_ver_merged_wl.csv'
FILE_PATTERN_PRIMARY_OFFSETS = '_offsets.csv'
FILE_PATTERN_B1_GAIN_OFFSETS = '_B1_gain_offsets.csv'

# Extract all three dataset types if their time periods are available
DATASET_TYPES = station.DATASET_TYPES

# Dates for training / testing / validation
TRAIN_END_DATE = "2016-12-31"
VALID_START_DATE, VALID_END_DATE = "2017-01-01", "2018-12-31"
TEST_START_DATE = "2019-01-01"

# Columns in stats csv files
CLEAN_STATS_KEYS = station.CLEAN_STATS_KEYS
DIFF_STATS_KEYS = station.DIFF_STATS_KEYS

# Number of randomly sampled data in Greg's AI model when training
N_RANDOM_SAMPLES = 200000

# TARGET threshold in meters between PRIMARY and VERIFIED
TARGET_THRESH = station.TARGET_THRESH

# Settings for giant histogram with all differences
GIANT_HIST_NBINS = station.GIANT_HIST_NBINS
GIANT_HIST_RANGE = station.GIANT_HIST_RANGE

###############################################
## Define data_cleaner class
###############################################
class data_cleaner (object):

    ''' This class manages cleaning processes of multiple stations '''

    def __init__ (self):

        ''' To initialize a new data_cleaner class '''

        ## File paths and locations
        self._raw_path  = None
        self._proc_path = None
        self._station_info_csv = None 

        ## Station data
        self._station_groups = None
        self._station_info = None

        ## Dump mid-step files to processed folder?
        self._create_midstep_files = False

        ## Cleaning stats from all stations
        self._train_stats_df = None
        self._validation_stats_df = None
        self._test_stats_df = None

        ## Information related to differences between primary and verified
        self._diff_hist_settings = {'nbins':GIANT_HIST_NBINS,
                                    'range':GIANT_HIST_RANGE}
        self._diff_stats_df = None
        self._diff_hist = {'edges':None, 'all':None, 'bad_only_by_thresh':None,
                           'bad_only_by_sensor_id':None}        

        ## Logger
        self._logger = logging.getLogger ('data_cleaner')
        self._logger.info ('Data cleaner instance is created.')

    # +------------------------------------------------------------
    # | Getters & setters
    # +------------------------------------------------------------
    @property
    def station_info (self): return self._station_info

    @property
    def station_groups (self): return self._station_groups

    @property
    def station_ids (self):
        return numpy.array ([sid for slist in self.station_groups for sid in slist])

    @property
    def train_stats (self): return self._train_stats_df

    @property
    def validation_stats (self): return self._validation_stats_df

    @property
    def test_stats (self): return self._test_stats_df

    @property
    def diff_stats (self): return self._diff_stats_df

    @property
    def raw_path (self): return self._raw_path
    @raw_path.setter
    def raw_path (self, apath):
        ## Make sure input path exists
        self._check_file_path_existence (apath)
        self._logger.info ('Raw data folder is set to {0}.'.format (apath))
        self._raw_path = apath

    @property
    def proc_path (self): return self._proc_path
    @proc_path.setter
    def proc_path (self, apath):
        ## Make sure input path exists
        self._check_file_path_existence (apath)
        self._logger.info ('Processed data folder is set to {0}.'.format (apath))
        self._proc_path = apath

    @property
    def station_info_csv (self): return self._station_info_csv
    @station_info_csv.setter
    def station_info_csv (self, afile):
        ## Make sure input file exists
        self._check_file_path_existence (afile)
        self._logger.info ('Station info sheet is set to {0}.'.format (afile))
        self._station_info_csv = afile

    @property
    def create_midstep_files (self): return self._create_midstep_files
    @create_midstep_files.setter
    def create_midstep_files (self, aBoolean):
        if not isinstance (aBoolean, bool):
            message = 'Cannot accept a non-boolean, {0}, for create_midstep_files.'.format (aBoolean)
            self._logger.fatal (message)
            raise IOError (message)
        self._create_midstep_files = aBoolean

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

        ## Make sure proc path is already set
        if self._proc_path is None:
            message = 'Processed data path is None. Do not know where to write.'
            self._logger.fatal (message)
            raise IOError (message + ' Please set the path to processed folder.')

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
        filename = self._proc_path + '/' + filebasename + '.csv'
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
        if not type (array) in station.ARRAY_TYPES:
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

    def _null_values_found (self, column_name, series):

        ''' A private function to check if there is any null value in a column
            of a dataframe. If there is, an IOError is thrown. 

            input params
            ------------
            column_name (str): Name of the column to be checked
            series (pandas.core.series.Series): column to be checked
        '''

        if series.isna().any():

            message = 'Column, {0}, in station info csv contains null value.'
            self._logger.fatal (message.format (column_name))

            message += '\nPlease fill in all cells in that column in {1}.'
            raise IOError (message.format (column_name, self._station_info_csv))

    # +------------------------------------------------------------
    # | Plotting functions
    # +------------------------------------------------------------
    def _extract_global_stats_per_set (self, dtype):

        ''' A private function to extract out the counts per station for a
            given dataset.

            input params
            ------------
            dtype (str): name of dataset type where data is extracted
        '''

        ## Get the stats df based on input dataset type
        stats_df = getattr (self, '_' + dtype + '_stats_df')

        ## Extract the columns 'station_id' and 'n_total' in each stats
        ## dataframe is the total number of records per set i.e. not the
        ## total number per station.
        subframe = stats_df.loc[:, ['station_id', 'n_total']]

        ## Re-format dataframe before adding more stats sets.
        subframe.index = subframe.station_id
        subframe = subframe.drop (axis=1, columns='station_id').sort_index()
        subframe.columns = [col + '_' + dtype for col in subframe.columns]

        return subframe.sort_index()

    def _extract_global_stats (self):

        ''' A private function to combine train/valid/test dataframes and
            select out the counts per station per set.
        '''

        ## Collect the stats columns to be plotted
        statsframe = None

        ## Loop through each dataset type 
        for dtype in DATASET_TYPES:
            # Collect the counts for this dataset type
            subframe = self._extract_global_stats_per_set (dtype)
            # If this dataset type is the first one
            # replace statsframe with this frame
            if statsframe is None: 
                statsframe = subframe; continue
            # If not the first dataset type, merge to the existing one.
            statsframe = pandas.merge (statsframe, subframe, right_index=True,
                                       left_index=True, how='outer')

        ## Add in the column of total # records 
        statsframe['n_total'] = statsframe.n_total_train + \
                                statsframe.n_total_validation + \
                                statsframe.n_total_test

        return statsframe

    def plot_global_stats (self):
        
        ''' A public function to plot global stats.
                * Top: # records per station
                # Bottom: % of train/valid/test per station
        '''

        ## Gather dataframe with record counts per set
        statsframe = self._extract_global_stats()

        ## Start plotting!
        h = plt.figure (figsize=(9, 5))
        gs = gridspec.GridSpec (2, 1, wspace=0.1)
        gs.update (bottom=0.15)

        ## Top plot: # total records per station
        axis = h.add_subplot (gs[0])
        xvalues = numpy.arange (len (statsframe))
        yvalues = statsframe['n_total'].values / 1000000 # counts in million
        axis.scatter (xvalues, yvalues, marker='o', color='black', s=20, alpha=0.8)
            
        ##  Format x-axis
        axis.set_xlim ([min(xvalues)-1, max(xvalues)+1])
        axis.set_xticks (xvalues)
        axis.get_xaxis ().set_ticklabels ([])
        ##  Format y-axis
        axis.set_ylim ([0.8, 1.2])
        axis.tick_params (axis='y', labelsize=8)
        axis.set_ylabel ('# total [million]', fontsize=8)       
        ##  Plot grid lines
        for ytick in axis.yaxis.get_majorticklocs():
            axis.axhline (y=ytick, color='gray', alpha=0.3, linestyle=':', linewidth=0.2)
        for xtick in axis.xaxis.get_majorticklocs():
            axis.axvline (x=xtick, color='gray', alpha=0.3, linestyle=':', linewidth=0.2)

        ## Bottom plot: # total per set / # total records
        axis = h.add_subplot (gs[1])
        xvalues = numpy.arange (len (statsframe))
        for dtype in DATASET_TYPES:
            color = 'black' if dtype=='train' else 'blue' if dtype=='validation' else 'red'
            marker = 'o' if dtype=='train' else 'x' if dtype=='validation' else '+'
            yvalues = statsframe['n_total_' + dtype].values / statsframe['n_total'].values 
            axis.scatter (xvalues, yvalues, marker=marker, color=color, s=20, alpha=0.8, label=dtype)
            
        ##  Format x-axis
        axis.set_xlim ([min(xvalues)-1, max(xvalues)+1])
        axis.set_xticks (xvalues)
        axis.set_xticklabels (statsframe.index)
        axis.tick_params (axis='x', labelsize=8, labelrotation=90)
        ##  Format y-axis
        axis.set_ylim ([0, 1])
        axis.tick_params (axis='y', labelsize=8)
        axis.set_ylabel ('# per set / # total', fontsize=8)       
        ##  Plot grid lines
        for ytick in axis.yaxis.get_majorticklocs():
            axis.axhline (y=ytick, color='gray', alpha=0.3, linestyle=':', linewidth=0.2)
        for xtick in axis.xaxis.get_majorticklocs():
            axis.axvline (x=xtick, color='gray', alpha=0.3, linestyle=':', linewidth=0.2)
        ##  Plot legend
        axis.legend (loc=0, fontsize=8)        

        ### Store plot as PDF
        plt.suptitle ('Global statistics', fontsize=15)
        h.savefig (self._proc_path + '/global_stats.pdf')
        plt.close ('all')
        return

    def _plot_subplot_nan_capped_vs_n_spikes (self, axis, stats_groups, stats_df,
                                              key, dtype, doLegend=False):

        ''' A private function to plot sub-plot for nan / capped vs spikes %
            correlation plot for a given dataset type. 

            input params
            ------------
            axis (matplotlib.Axes): the axis object to be plotted
            stats_groups (grouped DataFrame): stats_df grouped by 'has bad results'
            stats_df (pandas.DataFrame): stats dataframe of a given dataset type
            key (str): the column, nan/capped primary/backup, to be plotted
            reference (int): denominator in calculating spike percentage
            doLegend (bool): plot legend if true
        '''

        ## Plot scatter plot: x = log10 (n_spikes), y = % spikes
        for isbad in stats_groups.groups.keys():
            color = 'red' if isbad else 'green'
            marker = 'x' if isbad else 'o' 
            label = 'bad stations' if isbad else 'good stations'
            # Define x and y values to be n_spikes and n_nan_primary
            this_group = stats_groups.get_group (isbad)
            yvalues = numpy.log10 (this_group[key])
            xvalues = this_group.n_spikes_percent.values
            axis.scatter (xvalues, yvalues, marker=marker, color=color,
                          s=15, alpha=0.8, label=label)

        ##  Format x-axis
        xmin = 0
        xmax = this_group.n_spikes_percent.max() + 0.05
        xticks = numpy.linspace (xmin, xmax, 6)
        axis.set_xlim ([xmin, xmax])
        axis.set_xticks (xticks)
        axis.tick_params (axis='x', labelsize=8)
        axis.set_xlabel ('# spikes / # total {0}'.format (dtype), fontsize=10)
        ##  Format y-axis
        yvalues = numpy.log10 (stats_df[key])
        yvalues [~numpy.isfinite (yvalues)] = 0
        ymin = numpy.floor (max (0, min(yvalues)))
        ymax = numpy.ceil (max(yvalues))
        yticks = numpy.linspace (ymin, ymax, 6)        
        axis.set_ylim ([ymin, ymax])
        axis.set_yticks (yticks)
        axis.tick_params (axis='y', labelsize=8)
        ylabel = 'Log10 # of '
        ylabel += 'nan ' if 'nan' in key else 'capped '
        ylabel += 'primary' if 'primary' in key else 'backup'
        axis.set_ylabel (ylabel, fontsize=10)
        ##  Plot grid lines
        for ytick in axis.yaxis.get_majorticklocs():
            axis.axhline (y=ytick, color='gray', alpha=0.3, linestyle=':', linewidth=0.2)
        for xtick in axis.xaxis.get_majorticklocs():
            axis.axvline (x=xtick, color='gray', alpha=0.3, linestyle=':', linewidth=0.2)
        ##  Plot legend
        if doLegend: axis.legend (loc=0, fontsize=10)

    def plot_nan_capped_vs_n_spikes (self, dtype):

        ''' A public function to plot correlation between of # nan & capped
            values vs spikes % for a given dataset type in log-log scale. The
            stations are grouped by "good" vs "bad" where bad stations are
            those labelled as 'problematic' from station info sheet.

            Top left    : log10 (# nan primary) vs % spikes
            Top right   : log10 (# nan backup) vs % spikes
            Bottom left : log10 (# capped primary) vs % spikes
            Bottom right: log10 (# capped backup) vs % spikes

            input params
            ------------
            dtype (str): name of dataset type to be plotted
        '''

        ## Collect this stats df and group the stations by has_bad_results
        stats_df = getattr (self, '_' + dtype + '_stats_df')
        #  Move capped max / min into 1 capped column
        stats_df['n_capped_primary'] = stats_df.n_capped_primary_min + \
                                       stats_df.n_capped_primary_max         
        stats_df['n_capped_backup']  = stats_df.n_capped_backup_min + \
                                       stats_df.n_capped_backup_max 
        #  Scale n_spikes w.r.t. # records in this set
        reference = N_RANDOM_SAMPLES if dtype=='train' else stats_df.n_total
        stats_df['n_spikes_percent'] = stats_df.n_spikes / reference
        #  Group dataframe by bad / good stations
        stats_groups = stats_df.groupby (by = 'has_bad_results')
        
        ## Start plotting!
        h = plt.figure (figsize=(9, 9))
        gs = gridspec.GridSpec (2, 2, wspace=0.2, hspace=0.2)

        ## Top left plot: log10 (# nan primary) vs % spikes
        axis = h.add_subplot (gs[0])
        self._plot_subplot_nan_capped_vs_n_spikes (axis, stats_groups, stats_df,
                                                   'n_nan_primary', dtype,
                                                   doLegend=True)

        ## Top right plot: log10 (# nan backup) vs % spikes
        axis = h.add_subplot (gs[1])
        self._plot_subplot_nan_capped_vs_n_spikes (axis, stats_groups, stats_df,
                                                   'n_nan_backup', dtype,
                                                   doLegend=False)

        ## Bottom left plot: log10 (# capped primary) vs % spikes
        axis = h.add_subplot (gs[2])
        self._plot_subplot_nan_capped_vs_n_spikes (axis, stats_groups, stats_df,
                                                   'n_capped_primary', dtype,
                                                   doLegend=False)
        
        ## Bottom right plot: log10 (# capped backup) vs % spikes
        axis = h.add_subplot (gs[3])
        self._plot_subplot_nan_capped_vs_n_spikes (axis, stats_groups, stats_df,
                                                   'n_capped_backup', dtype,
                                                   doLegend=False)        

        ### Store plot as PDF
        title = 'Nan / capped vs spikes % correlation for {0} set'.format (dtype)
        plt.suptitle (title, fontsize=15)
        h.savefig (self._proc_path + '/nan_capped_vs_spikes_' + dtype + '.pdf')
        plt.close ('all')
        return        

    def plot_stats (self, dtype):
    
        ''' A public function to plot the statistics info of a dataset type.
                * global_stats: Total # records and % per set
                * nan_capped_vs_spikes: corr of # nan & capped values vs # spikes

            input params
            ------------
            dtype (str): name of dataset type to be plotted
        '''

        ## Make sure the input dataset type name is recognizable
        ## i.e. train, validation, and test
        if dtype not in DATASET_TYPES:
            message = 'Invalid dataset type, {0}, is provided.'.format (dtype)
            self._logger.fatal (message)
            message += '\nPlease provide one of the followings: {0}'
            raise IOError (message.format (DATASET_TYPES))

        ## Plot global stats - total number of records & % per set
        self.plot_global_stats()

        ## Plot correlations of nan and capped data points with spikes per set
        self.plot_nan_capped_vs_n_spikes (dtype)

    def plot_diff_stats (self):

        ## Start plotting!
        h = plt.figure (figsize=(9, 3))
        gs = gridspec.GridSpec (1, 1)
        gs.update (bottom=0.23)
        axis = h.add_subplot (gs[0])

        ## Plot individual statistics
        xvalues = numpy.arange (len (self.diff_stats)) + 1
        yvalues = self.diff_stats['mean'].values
        yerrors = numpy.array ([yvalues - self.diff_stats['lower'],
                                self.diff_stats['upper'] - yvalues])
        axis.errorbar (xvalues, yvalues, yerr=yerrors, marker='o', color='black',
                       markersize=6, alpha=0.7, linestyle=None, linewidth=0.0,
                       ecolor='blue', elinewidth=3, capsize=0.0, capthick=0.0)

        ## Plot horizontal line
        axis.axhline (y=TARGET_THRESH, color='red', alpha=0.7, linestyle='--', linewidth=1)
        axis.axhline (y=-1*TARGET_THRESH, color='red', alpha=0.7, linestyle='--', linewidth=1)

        ##  Format x-axis
        axis.set_xlim ([min(xvalues) - 1, max(xvalues) + 1])
        axis.set_xticks (xvalues)
        axis.set_xticklabels (self.diff_stats.station_id)
        axis.tick_params (axis='x', labelsize=8, labelrotation=90)
        ##  Format y-axis
        axis.set_ylim ([-0.1, 0.1])
        axis.tick_params (axis='y', labelsize=8)
        axis.set_ylabel ('Primary - Verified\n[meters]', fontsize=10)

        ##  Add title
        axis.set_title ('Distributions of primary - verified', fontsize=12)
        ##  Plot grid lines
        for ytick in axis.yaxis.get_majorticklocs():
            axis.axhline (y=ytick, color='gray', alpha=0.3, linestyle=':', linewidth=0.2)
        for xtick in axis.xaxis.get_majorticklocs():
            axis.axvline (x=xtick, color='gray', alpha=0.3, linestyle=':', linewidth=0.2)

        ### Store plot as PDF
        h.savefig (self.proc_path + '/diff_stats.pdf')
        plt.close ('all')
        return

    def _get_diff_statistics (self, xvalues, yvalues):
        
        ''' A private function to obtain statistics from a histogram with
            x and yvalues. It builds a cumulative histogram and flips (inverts)
            it. The mean, min, max, top and bottom 5% are obtained to get a
            rough shape of the histogram. 

            lower: 5%
            mean: 50%
            upper: 95%

            input params
            ------------
            xvalues (array): the xvalues for fitting a CDF
            yvalues (array): the yvalues for fitting a CDF

            return params
            -------------
            stats (dict): 5%, 50%, 95% from the histogram
        '''
        
        cdf = numpy.cumsum (yvalues) / sum (yvalues)
        bins = xvalues[:-1] + (xvalues[1:] - xvalues[:-1])/2.
        icdf = interp1d (cdf, bins)
        lowP = max (0.025, min (icdf.x))
        highP = min (0.975, max (icdf.x))
        midP = 0.5
        if highP < lowP:
            highP = (max (icdf.x) - lowP)*0.975 + lowP
            midP = (max (icdf.x) - lowP)*0.5 + lowP
        lower, mean, upper = icdf(lowP), icdf(midP), icdf(highP)
        return {'lower':float (lower), 'upper':float (upper), 'mean':float (mean)}

    def _plot_subplot_giant_diff (self, axis, htype='all'):

        ''' A private function to plot a sub plot for the summary histogram.
            The histogram is indicated by the htype, which must be one of the
            3 keys in self._hist_diff: 'all', 'bad_only_by_thresh', or
            'bad_only_by_sensor_id'. By default this histogram covers quite
            a wide range (+/- 20 or 30 meters) with fine binning. With the full
            range, the 90% interval is extracted. This function then plots the
            sub-section of the histogram between -0.1 and 0.1 meters.

            input params
            ------------
            axis (matplotlib.Axes): axis on which plots are made
            htype (str): key in self._hist_diff
        '''

        ## Determine the 5, 50, 95%
        yvalues = self._diff_hist[htype]
        xvalues = self._diff_hist['edges']     
        stats = self._get_diff_statistics (xvalues, yvalues)

        ## Plot the histogram
        yvalues = numpy.log10 (yvalues)
        yvalues = [yvalues[0]] + list (yvalues)
        axis.plot (xvalues, yvalues, color='gray', alpha=0.7, linestyle='-',
                   linewidth=1.5, drawstyle='steps-pre')

        ## Plot vertical shaded area between 5 and 95%
        axis.axvspan (stats['lower'], stats['upper'], color='blue', alpha=0.2)

        #  Print 90% interval
        text = '50% = {0:.3f} cm\n'.format (stats['mean']*100)
        text += '90% = [{0:.3f}, {1:.3f}] cm'.format (stats['lower']*100, stats['upper']*100)
        anchored_text = AnchoredText (text, loc=2, frameon=False)
        axis.add_artist (anchored_text)

        ##  Format x-axis
        axis.set_xlim ([-0.1 ,0.1]) ## from -0.1 to 0.1 meters
        axis.set_xticks (numpy.linspace (-0.1, 0.1, 21)) # 21 xticks
        axis.tick_params (axis='x', labelsize=5)
        axis.set_xlabel ('Primary - Verified [meters]', fontsize=10)

        ##  Format y-axis
        axis.set_ylim ([0, numpy.ceil (max(yvalues))])
        axis.tick_params (axis='y', labelsize=8)
        axis.set_ylabel ('log10 #', fontsize=10)

        ##  Add title
        title = 'All data points' if htype=='all' else htype.replace ('_', ' ')
        axis.set_title (title, fontsize=12)
        ##  Plot grid lines
        for ytick in axis.yaxis.get_majorticklocs():
            axis.axhline (y=ytick, color='gray', alpha=0.3, linestyle=':', linewidth=0.2)
        for xtick in axis.xaxis.get_majorticklocs():
            axis.axvline (x=xtick, color='gray', alpha=0.3, linestyle=':', linewidth=0.2)

    def plot_giant_diff_hist (self):

        ''' A public function to plot summary histogram of primary - verified.
            For each histogram, 90% interval is shaded and printed on the plot.

            Right: Histogram with all points
            Middle: Histogram with only bad points defined by threshold
            Left: Histogram with only bad points defined by sensor ID
        '''

        ## Start plotting!
        h = plt.figure (figsize=(17, 5.5))
        gs = gridspec.GridSpec (1, 3)

        ## Plot the histogram with all data points (log y-axis)
        axis = h.add_subplot (gs[0])
        self._plot_subplot_giant_diff (axis, htype='all')

        ## Plot the histogram with bad points  (log y-axis)
        axis = h.add_subplot (gs[1])
        self._plot_subplot_giant_diff (axis, htype='bad_only_by_thresh')

        ## Plot the histogram with bad points  (log y-axis)
        axis = h.add_subplot (gs[2])
        self._plot_subplot_giant_diff (axis, htype='bad_only_by_sensor_id')

        ### Store plot as PDF
        plt.suptitle ('Distributions of primary - verified from all stations', fontsize=12)
        h.savefig (self.proc_path + '/diff_hist_summary.pdf')
        plt.close ('all')
        return

    def plot_all_stats (self):

        ''' A public function to plot the statistics info of all dataset types.
                * global_stats: Total # records and % per set
                * nan_capped_vs_spikes: corr of # nan & capped values vs # spikes
                                        for each datatype set
        '''

        ## Plot global stats - total number of records & % per set
        self.plot_global_stats()

        ## Plot diff stats - total number of records & % per set
        self.plot_diff_stats()
        self.plot_giant_diff_hist()

        ## Plot correlations of nan and capped data points with spikes per set
        for dtype in DATASET_TYPES:
            self.plot_nan_capped_vs_n_spikes (dtype)

    # +------------------------------------------------------------
    # | Handle station list and data from station info csv
    # +------------------------------------------------------------
    def _redefine_begin_end_date (self, dates, all_begin, all_end):

        ''' A private function to redefines the dates of all stations if needed.
            For any dates before the all_begin dates (i.e. the start of downloaded
            time-series), the dates are re-defined to be the start of time-series.
            For any dates after the all_end dates (i.e. the end of downloaded
            time-series), the dates are re-defined to be the end of time-series.

            input params
            ------------
            dates (numpy.array): each element is either a begin or end date of a
                                station for either train, test, or validation set
            all_begin (numpy.array): each element is the download begin date
            all_end (numpy.array): each element is the download end date

            return params
            -------------
            dates (numpy.array): redefined begin / end date for a set of a stat-
                                 ion for either train, test, or validation set
        '''

        ## Check if this date is before end date
        dates_before_allend = pandas.to_datetime (dates) <= pandas.to_datetime (all_end)
        dates[~dates_before_allend] = all_end[~dates_before_allend]

        ## Check if this date is after begin date
        dates_after_allbegin = pandas.to_datetime (dates) >= pandas.to_datetime (all_begin)
        dates[~dates_after_allbegin] = all_begin[~dates_after_allbegin]
        return dates

    def _read_station_info (self):

        ''' A private function to read station info sheet as a dataframe. In
            doing so, the periods of training / validation / testing sets are
            determined. 
                * Training  : from dataset start date to 2016-12-31
                * Validation: from 2017-01-01 to 2018-12-31
                * Testing   : from 2019-01-01 to dataset end date
            These dates are fixed. If the datasets have less data, they will
            just have shorter / no validation or testing sets.

            return params
            -------------
            station_frame (pandas.DataFrame): station meta-data from info sheet
        '''

        ## Make sure station info csv is defined
        if self._station_info_csv is None:
            message = 'Station info sheet is undefined.'
            self._logger.fatal (message)
            message += 'Please set it via `cleaner.stationInfoCSV = "/To/station/file.csv"`'
            raise FileNotFoundError (message)  

        ## Load all stations from info sheet - first 3 rows are title, color
        #  index, and blank row. Avoid renaming columns because new columns may
        #  be added in the future.
        station_frame = pandas.read_csv (self._station_info_csv, skiprows=3)
        n_stations = len (station_frame)
        
        ## Get the full time periods of station sets
        full_periods = station_frame['Dates downloaded (or to be downloaded)']
        self._null_values_found ('Dates downloaded (or to be downloaded)', full_periods)
        all_begin = full_periods.apply (lambda x: x.split (' ')[0])
        all_end   = full_periods.apply (lambda x: x.split (' ')[2])

        ## Replace training / validation / testing dates 
        for dtype in DATASET_TYPES:
            # Define default begin / end dates of this type
            begin = all_begin.values if dtype=='train' else \
                    [TEST_START_DATE]  * n_stations if dtype=='test' else \
                    [VALID_START_DATE] * n_stations
            end   = all_end.values   if dtype=='test' else \
                    [TRAIN_END_DATE] * n_stations if dtype=='train' else \
                    [VALID_END_DATE] * n_stations
            # Make sure begin & end dates are within downloaded range
            begin = self._redefine_begin_end_date (numpy.array (begin), all_begin, all_end)
            end   = self._redefine_begin_end_date (numpy.array (end)  , all_begin, all_end)
            # Redefine data set date period. If begin date is the same as end
            # date, not enough data i.e. NaN
            period = [numpy.NaN if begin[index]==end[index] else
                      begin[index] + ' to ' + end[index]
                      for index in range (len (end))]
            # Column name is the same as existing column name
            column_suffix = 'training' if dtype=='train' else \
                            'testing' if dtype=='test' else 'Validation' 
            station_frame['Dates used for ' + column_suffix] = period

        return station_frame

    def _group_stations_by_neighbor (self):

        ''' A private function to group stations by neighbor IDs. The returned
            value is a list. Each element is a sub-list of 2~3 station IDs that
            are neighbors of each other. 

            When asked to clean data, data_cleaner performs the cleaning in
            groups to reduce memory usage.

            return param
            ------------
            station_list (list): List of sub-list of neighbor stations
        '''

        ## Extract the only two columns that matter: station ID and neighbor ID
        station_df = self._station_info.loc[:, ['Station ID', 'Neighbor station number']]

        ## Initialize a holder to append station IDs
        station_list = []

        ## Loop through each row in station info csv dataframe
        for index, row in station_df.iterrows():
            # Collect the sorted list of station ID and neighbor ID
            subarray = sorted (row.values)
            # Loop through the sub ID lists in station_list holder 
            found = False
            for index, prevarray in enumerate (station_list):
                # If this new list already exists, nothing needs to be done.
                if prevarray == subarray:
                    found = True; break
                # If any one station in the new list exist, need to update the
                # other non-existing one(s) to the station_list
                if numpy.in1d (subarray, prevarray).any():
                    station_list[index] = list (numpy.unique (prevarray + subarray))
                    found = True; break
            # If stations already exist in station_list, next list!
            if found: continue
            # Otherwise, update station_list with the new list.
            station_list.append (subarray)

        return station_list

    def load_station_info (self):

        ''' A public function to load station information from station info
            sheet. This function sets 2 private variables
                * station_info: dataframe with all station meta-data
                * station_groups: station IDs grouped by neighboring info
        '''

        ## Read stations from station csv file
        self._station_info = self._read_station_info()

        ## Group station ID by neigbors 
        self._station_groups = self._group_stations_by_neighbor()

        ## Log - How many stations are in the station info sheet?
        message = 'Successfully read station info sheet - {0} stations are included.'
        self._logger.info (message.format (len (self.station_ids)))

    # +------------------------------------------------------------
    # | Load & clean stations by groups
    # +------------------------------------------------------------
    def _has_complete_set (self, station_id):
    
        ''' A private function to check if all raw files are available i.e. 
            raw data, primary offsets, and backup gain and offset files.

            input params
            ------------
            station_id (int): station ID from which a station instance is created

            return params
            -------------
            Boolean: If true, this station has all files.
        '''

        ## 1. Check if raw csv file exists
        raw_files = numpy.array (glob (self._raw_path + '/' + station_id + FILE_PATTERN_RAW_CSV))
        if len (raw_files) == 0:
            message = 'Station {0} does not have {0}{1} file.'
            self._logger.warn (message.format (station_id, FILE_PATTERN_RAW_CSV))
            return False
        if len (raw_files) > 1:
            message = 'Station {0} has multiple {0}{1} files.'
            self._logger.warn (message.format (station_id, FILE_PATTERN_RAW_CSV))        

        ## 2. Check if primary offsets file exists
        primary_offset_files = numpy.array (glob (self._raw_path + '/' + station_id + FILE_PATTERN_PRIMARY_OFFSETS))
        if len (primary_offset_files) == 0:
            message = 'Station {0} does not have {0}{1} file.'
            self._logger.warn (message.format (station_id, FILE_PATTERN_PRIMARY_OFFSETS))
            return False
        if len (primary_offset_files) > 1:
            message = 'Station {0} has multiple {0}{1} files.'
            self._logger.warn (message.format (station_id, FILE_PATTERN_PRIMARY_OFFSETS))

        ## 3. Check if backup B1 gain / offsets file exists
        backup_gain_offset_files = numpy.array (glob (self._raw_path + '/' + station_id + FILE_PATTERN_B1_GAIN_OFFSETS))
        if len (backup_gain_offset_files) == 0:
            message = 'Station {0} does not have {0}{1} file.'
            self._logger.warn (message.format (station_id, FILE_PATTERN_B1_GAIN_OFFSETS))
            return False
        if len (backup_gain_offset_files) > 1:
            message = 'Station {0} has multiple {0}{1} files.'
            self._logger.warn (message.format (station_id, FILE_PATTERN_B1_GAIN_OFFSETS))            

        return True

    def _set_up_station (self, station_id):

        ''' A private function to set up a station instance for the input
            station ID. Meta-data for this station is extracted from the
            station info sheet and parsed to the station instance. The primary
            offsets and backup gain / offsets are also loaded, and the raw
            data file is also parsed to the station instance, which is then
            returned.

            input params
            ------------
            station_id (int): station ID from which a station instance is created

            return params
            -------------
            astation (station): a station instance with all info set up 
        '''

        ## Define new station instance
        astation = station.station (station_id)

        ## Tell it to create mid-step files as the cleaning process goes
        astation.create_midstep_files = self._create_midstep_files
        astation.proc_path = self._proc_path

        ## Parse station metadata
        metadata = self._station_info[self._station_info['Station ID'] == station_id]
        astation.set_station_info (metadata)

        ## Check if this station has all raw files
        station_id_str = str (station_id)
        is_complete = self._has_complete_set (station_id_str)
        message = 'Station {0} has all raw files :)' if is_complete else \
                  'Station {0} does not have a complete set. Skipping this station from cleaning.'
        self._logger.info (message.format (station_id))
        #  If incomplete raw files, return empty station object without loading data
        if not is_complete: return astation

        ## Define the raw and offset file names required for this station
        raw_file = numpy.array (glob (self._raw_path + '/' + station_id_str + FILE_PATTERN_RAW_CSV))[0]
        primary_offset_file = numpy.array (glob (self._raw_path + '/' + station_id_str + FILE_PATTERN_PRIMARY_OFFSETS))[0]
        backup_gain_offset_file = numpy.array (glob (self._raw_path + '/' + station_id_str + FILE_PATTERN_B1_GAIN_OFFSETS))[0]

        ## Load offset data: offsets, and B1_gain_offsets
        astation.load_primary_offsets (primary_offset_file)
        astation.load_backup_B1_gain_offsets (backup_gain_offset_file)

        ## For raw file, only load it as needed to avoid intense memory usage at one
        ## point of time. For now, just let the station know the raw file location.
        astation.raw_file = raw_file

        return astation

    def _write_processed_station (self, station_id, dataframe):
        
        ''' A private function to write a dataframe into 3 files based on data-
            set type. The file name is based on station ID and the dataset type.

            input params
            ------------
            station_id (int): Station ID of this dataframe
            dataframe (pandas.DataFrame): cleaned data at input station ID
        '''

        ## Determine the output csv processed file
        outfilebase = '{0}/{1}_processed_ver_merged_wl'.format (self._proc_path, station_id)

        ## Loop through available train, validation, and test set
        for dtype in dataframe.setType.unique ():
            # Determine the actual file name
            outfile = outfilebase + '_' + dtype + '.csv'
            # Extract the set & drop the setType column
            subframe = dataframe[dataframe.setType == dtype].drop (axis=1, columns=['setType'])
            # Write the dataframe out!
            subframe.to_csv (outfile, index=False)
            self._logger.info ('{0} processed file at {1}.'.format (dtype, outfile))

    def _append_giant_histograms (self, diff_hist_per_station):

        ''' A private function to add histogram of difference from each station
            to the summary histogram plot.

            input params
            ------------
            diff_hist_per_station (dict): Has the same keys as self._diff_hist
        '''

        ## Loop through each key
        for key in self._diff_hist.keys ():
            # Get the value of histogram from this station
            value = numpy.array (diff_hist_per_station[key])
            # If the private variable does not have this value, set it!
            if self._diff_hist[key] is None:
                self._diff_hist[key] = value
                continue
            ## if edges, no need to re-set it
            if key == 'edges': continue
            ## For other histogram, add to the existing ones to sum up histogram
            self._diff_hist[key] += value

    def _clean_station_group (self, station_group, exclude_nan_verified=False):

        ''' A private function to clean 1 station group. These stations are
            neighbors. This function loops through each station in the group,
            creates a new station instance, perform the cleaning, and stores
            the training / validation / testing stats. Their dataframes are
            also temporarily stored.
            
            After all stations in the group are cleaned, each dataframe has 4
            new columns from its neighbor stations. After collecting the neighbor
            info, the dataframe is then written into 3 csv files based on data-
            set type.

            At the end, the stats are put into a dictionary of dataframes.

            input params
            ------------
            station_group (list): List of station IDs that are neighbors
            exclude_nan_verified (bool): If true, exclude nan verified from 
                                             spikes counting

            output params
            -------------
            stats_df (dict): {dtype: stats dataframe}
        '''

        ## Define holders for cleaned dataframe and stats (per set) and
        ## stats for differences from all sets
        neighbors, dataframes = [], {}
        stats = {key:{subkey:[] for subkey in ['station_id'] + CLEAN_STATS_KEYS}
                 for key in DATASET_TYPES}
        diff_stats = {key:[] for key in ['station_id'] + DIFF_STATS_KEYS}

        ## Loop through each station in the group
        for station_id in station_group:
            # Define a station instance 
            astation = self._set_up_station (station_id)
            # Collect neighbor id
            neighbors.append (astation.neighbor_id)
            # Cleaned data!
            dataframes[station_id] = astation.clean_raw_data (exclude_nan_verified=exclude_nan_verified)
            # Extract stats of primary - verified stats
            diff_stats['station_id'].append (station_id)
            for key in DIFF_STATS_KEYS:
                diff_stats[key].append (astation.diff_stats[key])
            # Extract the stats from this station
            for dtype in DATASET_TYPES:
                stats[dtype]['station_id'].append (station_id)
                stats_dict = getattr (astation, dtype + '_stats')
                for stats_key, stats_value in stats_dict.items():
                    stats[dtype][stats_key].append (stats_value)
            # Add the histograms to the giant histograms
            self._append_giant_histograms (astation.diff_hist)

        ## Handle neighbor info. The stations in the same group are related by
        ## their neighbor info. Once all of their data are cleaned, we add new
        ## columns to include neighbor info and dump out a csv per set
        for station_id, neighbor_id in zip (station_group, neighbors):
            # Get the dataframes
            this_df = dataframes[station_id]
            neighbor_df = dataframes[neighbor_id]
            # Merge the new column as 'NEIGHTBOR_xxx'
            for key in NEIGHBOR_COLUMNS:
                this_df= pandas.merge (this_df, neighbor_df['_'.join (key.split ('_')[1:])],
                                       left_index=True, right_index=True, how='left')
            # Rename the station columns and redefine the 
            this_df.columns = CLEANED_COLUMNS + ['setType'] + NEIGHBOR_COLUMNS
            # Write this station out
            self._write_processed_station (station_id, this_df)

        ## Return the stats as data frame for each set
        stats_df = {key:pandas.DataFrame (value) for key, value in stats.items()}
        return stats_df, pandas.DataFrame (diff_stats)

    def clean_stations (self, exclude_nan_verified=False, station_ids=None):

        ''' A public function to clean stations. If no station_ids provided, it
            cleans all available station listed in station info csv file. Other-
            wise, the requested stations (and their neighbor stations) are
            cleaned. 

            As of 2020/09/08, the counting of spikes may or may not include
            rows where PRIMARY is valid but VER_WL_VALUE_MSL is not. So, the
            flag exclude_nan_verified excludes those rows from spike
            counting.

            input params
            ------------
            exclude_nan_verified (bool): If true, exclude nan verified from 
                                             spikes counting
            station_ids (int or list of int): stations (and their neighbors) to
                                              be included in cleaning
        '''

        ## If station Info is not yet loaded, load it now.
        if self._station_groups is None: self.load_station_info()

        ## If there are input station_ids, identify which station group they are.
        ## If no station ids provided, clean all stations
        station_groups = self._station_groups if station_ids is None else \
                         [group for group in self._station_groups
                          if numpy.in1d (group, station_ids).any()]

        ## Make sure there are at least 1 station group. If not, it means
        ## that the input station ids are not present in station info sheet.
        if len (station_groups) == 0:
            message = 'Input station ids, {0}, are not present in station info sheet.'
            self._logger.fatal (message.format (station_ids))
            message += ' Please check your station ids with info sheet.'
            raise IOError (message.format (station_ids))

        ## Load data as groups to avoid memory demands. Stations are grouped
        ## by neighbor stations. 
        stats_df, diff_df = None, None
        for station_group in station_groups:
            # Clean this group of stations
            stats, diff = self._clean_station_group (station_group,
                                    exclude_nan_verified=exclude_nan_verified)
            # If this is the first group, just replace dataframe
            if stats_df is None and diff_df is None:
                stats_df = stats
                diff_df = diff
                continue
            # Otherwise, append individual dataframe
            diff_df = diff_df.append (diff, ignore_index=True)
            for dtype, maindf in stats_df.items():
                stats_df[dtype] = maindf.append (stats[dtype], ignore_index=True)
            
        ## Plot a giant histogram with all 


        ## Store stats_df to private variables. If they already exists, append.
        self._train_stats_df = stats_df['train'] if self._train_stats_df is None else \
                               self._train_stats_df.append (stats_df['train'], ignore_index=True)
        self._validation_stats_df = stats_df['validation'] if self._validation_stats_df is None else \
                               self._validation_stats_df.append (stats_df['validation'], ignore_index=True)
        self._test_stats_df = stats_df['test'] if self._test_stats_df is None else \
                               self._test_stats_df.append (stats_df['test'], ignore_index=True)
        self._diff_stats_df = diff_df if self._diff_stats_df is None else \
                               self._diff_stats_df.append (diff_df, ignore_index=True)

        ## Make sure there are no duplicated stations
        self._train_stats_df      = self._train_stats_df.drop_duplicates()
        self._validation_stats_df = self._validation_stats_df.drop_duplicates()
        self._test_stats_df       = self._test_stats_df.drop_duplicates()
        self._diff_stats_df       = self._diff_stats_df.drop_duplicates()

        ## If asked to create mid-step files, save and plot stats data!
        if self.create_midstep_files:
            self.save_stats_data()
            self.plot_all_stats ()

    def save_stats_data (self):

        ''' A public function to store stats csv file to proc_path.
        '''

        ## Write training stats
        self._dump_file ('train_stats', 'train_stats', self._train_stats_df)

        ## Write validation stats
        self._dump_file ('valid_stats', 'valid_stats', self._validation_stats_df)

        ## Write testing stats
        self._dump_file ('test_stats', 'test_stats', self._test_stats_df)

        ## Write diff stats
        self._dump_file ('diff_stats', 'diff_stats', self._diff_stats_df)        
