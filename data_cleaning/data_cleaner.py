#!python37

## By Elim Thompson (09/02/2020)
##
## This script defines a data_cleaner class for Greg's WL-AI project. This primary goal of this
## class is to clean raw station data extracted from the database by Armin. The cleaning procedure
## is based on Greg Dusek's clean.ipynb. Additional requirements are included in the doc:
##
## https://docs.google.com/document/d/1BfyIQE9GXPCRbBSkyurd3UeGqpGkAr1UYkMZzh5LBNk/edit?usp=sharing
##
## If asked, this class will also generate statistics and create plots to get a brief look at the
## raw data.
##
## Example snippet to use the station class:
## +-------------------------------------------------------------
## # Import station class
## import station, pandas
##
## This script defines a data_cleaner class for Greg's WL-AI project. This primary goal of this
## class is to clean raw station data extracted from the database by Armin. The cleaning procedure
## is based on Greg Dusek's clean.ipynb. Additional requirements are included in the doc:
##
## https://docs.google.com/document/d/1BfyIQE9GXPCRbBSkyurd3UeGqpGkAr1UYkMZzh5LBNk/edit?usp=sharing
##
## If asked, this class will also generate statistics and create plots to get a brief look at the
## raw data.
######################################################################################################

###############################################
## Import libraries
###############################################
import numpy, pandas, datetime, os, logging
import _pickle as pickle
from glob import glob

import station

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
# Final columns to be included 
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
CLEAN_STATS_KEYS = station.CLEAN_STATS_KEYS

###############################################
## Define functions
###############################################
# Short function to pull all station IDs from available raw CSV files.
get_periods = lambda aString: numpy.array ([pandas.to_datetime (adate.strip ()) for adate in aString.split ('to')])
get_primaries = lambda df: df[df.SENSOR_USED_PRIMARY + '_WL_VALUE_MSL']
get_primary_sigmas = lambda df: df[df.SENSOR_USED_PRIMARY + '_WL_SIGMA']

###############################################
## Define data_cleaner class
###############################################
class data_cleaner (object):

    def __init__ (self):

        self._raw_path  = None
        self._proc_path = None
        self._station_info_csv = None 

        self._station_groups = None
        self._station_info = None

        self._create_midstep_files = False

        self._train_stats_df = None
        self._validation_stats_df = None
        self._test_stats_df = None

        self._logger = logging.getLogger ('data_cleaner')
        self._logger.info ('Data cleaner instance is created.')

    # +------------------------------------------------------------
    # | Getters & setters
    # +------------------------------------------------------------
    @property
    def raw_path (self): return self._raw_path
    @raw_path.setter
    def raw_path (self, apath):
        self._check_file_path_existence (apath)
        self._logger.info ('Raw data folder is set to {0}.'.format (apath))
        self._raw_path = apath

    @property
    def proc_path (self): return self._proc_path
    @proc_path.setter
    def proc_path (self, apath):
        self._check_file_path_existence (apath)
        self._logger.info ('Processed data folder is set to {0}.'.format (apath))
        self._proc_path = apath

    @property
    def station_info_csv (self): return self._station_info_csv
    @station_info_csv.setter
    def station_info_csv (self, afile):
        self._check_file_path_existence (afile)
        self._logger.info ('Station info sheet is set to {0}.'.format (afile))
        self._station_info_csv = afile

    @property
    def create_midstep_files (self): return self._create_midstep_files
    @create_midstep_files.setter
    def create_midstep_files (self, aBoolean):
        if not type (aBoolean) == bool:
            message = 'Cannot accept a non-boolean, {0}, for create_midstep_files.'.format (aBoolean)
            self._logger.fatal (message)
            raise IOError (message)
        self._create_midstep_files = aBoolean

    @property
    def station_ids (self): return numpy.array ([sid for slist in self._station_groups for sid in slist])

    @property
    def station_info (self): return self._station_info

    @property
    def train_stats (self): return self._train_stats_df

    @property
    def validation_stats (self): return self._validation_stats_df

    @property
    def test_stats (self): return self._test_stats_df

    # +------------------------------------------------------------
    # | Misc validation functions
    # +------------------------------------------------------------
    def _check_file_path_existence (self, afilepath):
        if not os.path.exists (afilepath):
            message = 'Path or file, {0}, does not exist!'.format (afilepath)
            self._logger.fatal (message)
            raise FileNotFoundError (message)

    def _null_values_found (self, column_name, series):
        if series.isna().any():
            self._logger.fatal ('Column, {0}, in station info csv contains null value.'.format (column_name))
            raise IOError ('Column, {0}, in station info csv contains null value.\n' + 
                           'Please fill in all cells in that column in {1}.'.format (column_name, self._station_info_csv))

    def _dump_file (self, dataname, filebasename, dataframe):

        ## Make sure proc path is already set
        if self._proc_path is None:
            message = 'Processed data path is None. Do not know where to write.'
            self._logger.fatal (message)
            raise IOError (message + ' Please set the path to processed folder.')

        ## Make sure the input dataframe is valid
        if dataframe is None:
            message = 'Input {0} dataframe is None. Nothing to write.'.format (dataname)
            self._logger.debug (message)
            return
        if not isinstance (dataframe, pandas.core.frame.DataFrame):
            message = 'Input {0} dataframe is not a pandas dataframe. '.format (dataname) + \
                      'Only pandas dataframe can be written.'
            self._logger.debug (message)
            return

        ## Write the dataframe to a csv file!
        filename = self._proc_path + '/' + filebasename + '.csv'
        dataframe.to_csv (filename, index=False)
        self._logger.info ('{0} dataframe is written to {1}.'.format (dataname, filename))

    # +------------------------------------------------------------
    # | Plotting functions
    # +------------------------------------------------------------
    def _extract_global_stats_per_set (self, dtype):

        ## Get the stats df based on input dataset type
        stats_df = getattr (self, '_' + dtype + '_stats_df')

        ## Extract the columns for global stats. Note that the 'n_total' in each stats
        ## dataframe is the total number of records per set i.e. not the total number per station.
        subframe = stats_df.loc[:, ['station_id', 'n_total', 'n_with_primary_sensor', 'n_with_other_primary_sensor']]

        ## Re-format dataframe before adding more stats sets.
        subframe.index = subframe.station_id
        subframe = subframe.drop (axis=1, columns='station_id').sort_index()
        subframe.columns = [col + '_' + dtype for col in subframe.columns]

        return subframe.sort_index()

    def _extract_global_stats (self):

        ## Collect the stats columns to be plotted
        statsframe = None
        for dtype in DATASET_TYPES:
            subframe = self._extract_global_stats_per_set (dtype)
            # If this dataset type is the first one, replace statsframe with this frame
            if statsframe is None: 
                statsframe = subframe
                continue
            #  If not the first dataset type, merge to the existing one
            statsframe = pandas.merge (statsframe, subframe, right_index=True, left_index=True, how='outer')
        #  Add in the column of total # records 
        statsframe['n_total'] = statsframe.n_total_train + statsframe.n_total_validation + statsframe.n_total_test

        return statsframe

    def plot_global_stats (self):
        
        ## Gather dataframe with global stats wthat 
        statsframe = self._extract_global_stats()

        ## Start plotting!
        h = plt.figure (figsize=(9, 5))
        gs = gridspec.GridSpec (2, 1, wspace=0.1)
        gs.update (bottom=0.15)

        ## Top plot: # records per set w.r.t. # total records
        axis = h.add_subplot (gs[0])
        xvalues = numpy.arange (len (statsframe))
        for dtype in DATASET_TYPES:
            color = 'black' if dtype=='train' else 'blue' if dtype=='validation' else 'red'
            marker = 'o' if dtype=='train' else 'x' if dtype=='validation' else '+'
            yvalues = statsframe['n_total_' + dtype].values / statsframe['n_total'].values 
            axis.scatter (xvalues, yvalues, marker=marker, color=color, s=20, alpha=0.8, label=dtype)
            
        ##  Format x-axis
        axis.set_xlim ([min(xvalues)-1, max(xvalues)+1])
        axis.set_xticks (xvalues)
        axis.get_xaxis ().set_ticklabels ([])
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

        ## Bottom plot: % of sensor types w.r.t. # records per set
        axis = h.add_subplot (gs[1])
        xvalues = numpy.arange (len (statsframe))
        for dtype in DATASET_TYPES:
            xoffset = -0.25 if dtype=='train' else +0.25 if dtype=='test' else 0.0
            for btype in ['n_with_primary_sensor', 'n_with_other_primary_sensor']:
                color = '#f093a2' if btype=='n_with_primary_sensor' else '#35a8e4'
                label = 'primary sensor'  if btype=='n_with_primary_sensor' else 'other primary sensor'
                bottom = 0 if btype=='n_with_primary_sensor' else yvalues
                yvalues = statsframe[btype + '_' + dtype].values / statsframe['n_total_' + dtype]
                axis.bar (xvalues + xoffset, yvalues, 0.15, label=label, bottom=bottom, color=color)

        ##  Format x-axis
        axis.set_xlim ([min(xvalues)-1, max(xvalues)+1])
        axis.set_xticks (xvalues)
        axis.set_xticklabels (statsframe.index)
        axis.tick_params (axis='x', labelsize=8, labelrotation=90)
        ##  Format y-axis
        axis.set_ylim ([0, 1])
        axis.tick_params (axis='y', labelsize=8)
        axis.set_ylabel ('# primary sensors / # per set', fontsize=8)       
        ##  Plot grid lines
        for ytick in axis.yaxis.get_majorticklocs():
            axis.axhline (y=ytick, color='gray', alpha=0.3, linestyle=':', linewidth=0.2)
        for xtick in axis.xaxis.get_majorticklocs():
            axis.axvline (x=xtick, color='gray', alpha=0.3, linestyle=':', linewidth=0.2)

        ### Store plot as PDF
        plt.suptitle ('Global statistics', fontsize=15)
        h.savefig (self._proc_path + '/global_stats.pdf')
        plt.close ('all')
        return

    def _plot_subplot_nan_capped_vs_n_spikes (self, axis, stats_groups, stats_df, key, doLegend=False):

        #  Plot scatter plot: x = log10 (n_spikes), y = log10 (key)
        for group in stats_groups.groups.keys():
            color = 'red' if group==False else 'green'
            marker = 'o' if group==False else 'x' 
            label = 'has good results' if group==False else 'has bad results'
            # Define x and y values to be n_spikes and n_nan_primary
            this_group = stats_groups.get_group (group)
            yvalues = numpy.log10 (this_group[key])
            xvalues = numpy.log10 (this_group.n_spikes)
            axis.scatter (xvalues, yvalues, marker=marker, color=color, s=15, alpha=0.8, label=label)
        ##  Format x-axis
        xmin = numpy.floor (min(numpy.log10 (stats_df.n_spikes)))
        xmax = numpy.ceil (max(numpy.log10 (stats_df.n_spikes)))
        xticks = numpy.linspace (xmin, xmax, 6)
        axis.set_xlim ([xmin, xmax])
        axis.set_xticks (xticks)
        axis.tick_params (axis='x', labelsize=8)
        axis.set_xlabel ('Number of spikes', fontsize=10)
        ##  Format y-axis
        ymin = numpy.floor (max (0, min(numpy.log10 (stats_df[key]))))
        ymax = numpy.ceil (max(numpy.log10 (stats_df[key])))
        print (ymin, ymax)
        yticks = numpy.linspace (ymin, ymax, 6)        
        axis.set_ylim ([ymin, ymax])
        axis.set_yticks (yticks)
        axis.tick_params (axis='y', labelsize=8)
        ylabel = 'Number of '
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

        ## Collect this stats df and group the stations by has_bad_results
        stats_df = getattr (self, '_' + dtype + '_stats_df')
        ## Move capped max / min into 1 capped column
        stats_df['n_capped_primary'] = stats_df.n_capped_primary_min + stats_df.n_capped_primary_max         
        stats_df['n_capped_backup']  = stats_df.n_capped_backup_min + stats_df.n_capped_backup_max 
        ## Scale n_spikes w.r.t. # records in this set
        stats_df['n_spikes_percent'] = stats_df.n_spikes / 200000
        ## Group dataframe by bad / good stations
        stats_groups = stats_df.groupby (by = 'has_bad_results')
        
        ## Start plotting!
        h = plt.figure (figsize=(9, 9))
        gs = gridspec.GridSpec (2, 2, wspace=0.2, hspace=0.2)
        #gs.update (bottom=0.15)

        ## Top left plot: # nan primary vs # spikes
        axis = h.add_subplot (gs[0])
        self._plot_subplot_nan_capped_vs_n_spikes (axis, stats_groups, stats_df, 'n_nan_primary', doLegend=True)

        ## Top right plot: # nan backup vs # spikes
        axis = h.add_subplot (gs[1])
        self._plot_subplot_nan_capped_vs_n_spikes (axis, stats_groups, stats_df, 'n_nan_backup', doLegend=False)

        ## Bottom left plot: # capped primary vs # spikes
        axis = h.add_subplot (gs[2])
        self._plot_subplot_nan_capped_vs_n_spikes (axis, stats_groups, stats_df, 'n_capped_primary', doLegend=False)
        
        ## Bottom right plot: # capped backup vs # spikes
        axis = h.add_subplot (gs[3])
        self._plot_subplot_nan_capped_vs_n_spikes (axis, stats_groups, stats_df, 'n_capped_backup', doLegend=False)        

        ### Store plot as PDF
        plt.suptitle ('Nan / capped vs Spikes correlation', fontsize=15)
        h.savefig (self._proc_path + '/nan_capped_vs_spikes.pdf')
        plt.close ('all')
        return        

    def plot_stats (self, dtype):
    
        ## Make sure the data type is recognizable i.e. train, validation, and test
        if dtype not in DATASET_TYPES:
            message = 'Invalid dataset type, {0}, is provided.'.format (dtype)
            self._logger.fatal (message)
            raise IOError (message + '\nPlease provide one of the followings: {0}'.format (DATASET_TYPES))

        ## Plot global stats - number of records per set 
        self.plot_global_stats()

        ## Plot correlations of nan and capped data points with spikes per set
        self.plot_nan_capped_vs_n_spikes (dtype)

    # +------------------------------------------------------------
    # | Handle station list and data from station info csv
    # +------------------------------------------------------------
    def _redefine_begin_end_date (self, dates, all_begin, all_end):

        # Check if this date is before end date
        dates_before_allend = pandas.to_datetime (dates) <=  pandas.to_datetime (all_end)
        dates[~dates_before_allend] = all_end[~dates_before_allend]

        # Check if this date is after begin date
        dates_after_allbegin = pandas.to_datetime (dates) >=  pandas.to_datetime (all_begin)
        dates[~dates_after_allbegin] = all_begin[~dates_after_allbegin]
        return dates

    def _read_station_info (self):

        ## Same for station list csv file
        if self._station_info_csv is None:
            self._logger.fatal ('Station info sheet is undefined.')
            raise FileNotFoundError ('Station info CSV file is undefined. ' + 
                                     'Please set it via `cleaner.stationInfoCSV = "/To/station/file.csv"`')  

        ## Load all stations from info sheet - first 3 rows are title, color index, and blank row
        #  Avoid renaming columns because new columns may be added in the future
        station_frame = pandas.read_csv (self._station_info_csv, skiprows=3)
        n_stations = len (station_frame)
        
        ## Get the full time periods of station sets
        full_periods = station_frame['Dates downloaded (or to be downloaded)']
        self._null_values_found ('Dates downloaded (or to be downloaded)', full_periods)
        all_begin = full_periods.apply (lambda x: x.split (' ')[0])
        all_end   = full_periods.apply (lambda x: x.split (' ')[2])

        ## Replace training / validation / testing dates (if empty cells)
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
            # Redefine data set date period. If begin date is the same as end date, not enough data i.e. NaN
            period = [numpy.NaN if begin[index]==end[index] else begin[index] + ' to ' + end[index] for index in range (len (end))]
            column_suffix = 'training' if dtype=='train' else 'testing' if dtype=='test' else 'Validation' 
            station_frame['Dates used for ' + column_suffix] = period

        return station_frame

    def _group_stations_by_neighbor (self, station_df):

        station_list = []
        for index, row in station_df.iterrows():
            subarray = sorted (row.values)
            found = False
            for index, prevarray in enumerate (station_list):
                if prevarray == subarray:
                    found = True; break
                if numpy.in1d (subarray, prevarray).any():
                    station_list[index] = list (numpy.unique (prevarray + subarray))
                    found = True; break
            if found: continue
            station_list.append (subarray)

        return station_list

    def load_station_info (self):

        ## Read stations from station csv file
        self._station_info = self._read_station_info ()

        ## Group station ID by neigbors 
        self._station_groups = self._group_stations_by_neighbor (self._station_info.loc[:, ['Station ID', 'Neighbor station number']])

        ## Log - How many stations are in the station info sheet?
        self._logger.info ('Successfully read station info sheet - {0} stations are included.'.format (len (self.station_ids)))

    # +------------------------------------------------------------
    # | Load & clean stations by groups
    # +------------------------------------------------------------
    def _has_complete_set (self, station_id):
    
        ## 1. Check if raw csv file exists
        raw_files = numpy.array (glob (self._raw_path + '/' + station_id + FILE_PATTERN_RAW_CSV))
        if len (raw_files) == 0:
            self._logger.warn ('Station {0} does not have {0}{1} file.'.format (station_id, FILE_PATTERN_RAW_CSV))
            return False
        if len (raw_files) > 1:
            self._logger.warn ('Station {0} has multiple {0}{1} files.'.format (station_id, FILE_PATTERN_RAW_CSV))        

        ## 2. Check if primary offsets file exists
        primary_offset_files = numpy.array (glob (self._raw_path + '/' + station_id + FILE_PATTERN_PRIMARY_OFFSETS))
        if len (primary_offset_files) == 0:
            self._logger.warn ('Station {0} does not have {0}{1} file.'.format (station_id, FILE_PATTERN_PRIMARY_OFFSETS))
            return False
        if len (primary_offset_files) > 1:
            self._logger.warn ('Station {0} has multiple {0}{1} files.'.format (station_id, FILE_PATTERN_PRIMARY_OFFSETS))

        ## 3. Check if backup B1 gain / offsets file exists
        backup_gain_offset_files = numpy.array (glob (self._raw_path + '/' + station_id + FILE_PATTERN_B1_GAIN_OFFSETS))
        if len (backup_gain_offset_files) == 0:
            self._logger.warn ('Station {0} does not have {0}{1} file.'.format (station_id, FILE_PATTERN_B1_GAIN_OFFSETS))
            return False
        if len (backup_gain_offset_files) > 1:
            self._logger.warn ('Station {0} has multiple {0}{1} files.'.format (station_id, FILE_PATTERN_B1_GAIN_OFFSETS))            

        return True

    def _set_up_station (self, station_id):

        ## Define new station instance
        astation = station.station (station_id)
        # Tell it to create mid-step files as the cleaning process goes
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
        
        # Determine the output csv processed file
        outfilebase = '{0}/{1}_processed_ver_merged_wl'.format (self._proc_path, station_id)

        # Loop through available train, validation, and test set
        for dtype in dataframe.setType.unique ():
            # Determine the actual file name
            outfile = outfilebase + '_' + dtype + '.csv'
            # Extract the set & drop the setType column
            subframe = dataframe[dataframe.setType == dtype].drop (axis=1, columns=['setType'])
            # Write the dataframe out!
            subframe.to_csv (outfile, index=False)
            self._logger.info ('{0} processed file at {1}.'.format (dtype, outfile))

    def _clean_station_group (self, station_group, use_VER_SENSOR_ID=False, exclude_nan_VER=False):

        ## Define holders for cleaned dataframe and stats
        neighbors = []
        dataframes = {}
        stats = {key:{subkey:[] for subkey in ['station_id'] + CLEAN_STATS_KEYS}
                 for key in DATASET_TYPES}

        ## Loop through each station in the group
        for station_id in station_group:
            # Define a station instance 
            astation = self._set_up_station (station_id)
            # Collect neighbor id
            neighbors.append (astation.neighbor_id)
            # Cleaned data!
            dataframes[station_id] = astation.clean_raw_data (use_VER_SENSOR_ID=use_VER_SENSOR_ID, exclude_nan_VER=exclude_nan_VER)
            # Extract the stats from this station
            for dtype in DATASET_TYPES:
                stats[dtype]['station_id'].append (station_id)
                stats_dict = getattr (astation, dtype + '_stats')
                for stats_key, stats_value in stats_dict.items():
                    stats[dtype][stats_key].append (stats_value)

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
        return stats_df

    def clean_stations (self, station_ids=None, use_VER_SENSOR_ID=False, exclude_nan_VER=False):

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
            message = 'Input station ids, {0}, are not present in station info sheet.'.format (station_ids)
            self._logger.fatal (message)
            raise IOError (message + ' Please check your station ids with info sheet.')

        ## Load data as groups to avoid memory demands. Stations are grouped
        ## by neighbor stations. 
        stats_df = None
        for station_group in station_groups:
            # Clean this group of stations
            stats = self._clean_station_group (station_group, use_VER_SENSOR_ID=use_VER_SENSOR_ID, exclude_nan_VER=exclude_nan_VER)
            # If this is the first group, just replace stats_df
            if stats_df is None:
                stats_df = stats
                continue
            # Otherwise, append individual dataframe
            for dtype, maindf in stats_df.items():
                stats_df[dtype] = maindf.append (stats[dtype], ignore_index=True)
            
        ## Store stats_df to private variables
        self._train_stats_df = stats_df['train'] if self._train_stats_df is None else \
                               self._train_stats_df.append (stats_df['train'], ignore_index=True)
        self._validation_stats_df = stats_df['validation'] if self._validation_stats_df is None else \
                               self._validation_stats_df.append (stats_df['validation'], ignore_index=True)
        self._test_stats_df = stats_df['test'] if self._test_stats_df is None else \
                               self._test_stats_df.append (stats_df['test'], ignore_index=True)

        ## Make sure there are no duplicated stations
        self._train_stats_df = self._train_stats_df.drop_duplicates()
        self._validation_stats_df = self._validation_stats_df.drop_duplicates()
        self._test_stats_df = self._test_stats_df.drop_duplicates()

        if self.create_midstep_files:
            self._save_stats_data()
            for dtype in DATASET_TYPES:
                self.plot_stats (dtype)

    def save_stats_data (self):

        ## Write training stats
        self._dump_file ('train_stats', 'train_stats', self._train_stats_df)

        ## Write validation stats
        self._dump_file ('valid_stats', 'valid_stats', self._validation_stats_df)

        ## Write testing stats
        self._dump_file ('test_stats', 'test_stats', self._test_stats_df)
