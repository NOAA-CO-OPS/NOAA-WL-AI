#!/usr/bin/python3

###
### By Elim Thompson (09/15/2019)
###
### This python script demonstrates a possible statistical approach to identify spikes
### in an observed water level time series data. The basic concept involves  building a
### distribution function at a given data point using data a day before the data point.
### A confident interval is determined, and the data point is considered as a spike if
### its observed water level is beyond the interval. This exercise is under the guidance
### of Katerina Glebushko with the help from Lindsay Abrams.
########################################################################################

#########################################################
### import packages
#########################################################
import pandas, numpy, scipy
from copy import deepcopy
from scipy.interpolate import interp1d

import matplotlib
matplotlib.use ('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
plt.rc ('font', family='sans-serif')
plt.rc ('font', serif='Computer Modern Roman')

#########################################################
### define constants
#########################################################
##  Directory where plots are dumped.
outpath = '/home/elims/elimstuff/noaa_work/DPT_spikes/outputs/'
##  Location of input CSV file.
csvFile = '/home/elims/elimstuff/noaa_work/DPT_spikes/data/8772471_20180801_20190731.csv'
##  Time period (in days) in which data prior the current point of
##  interest are included in the statistical analysis.
period = 1
##  The same time period converted from days to minutes.
interval = int(period * 24 * 60 / 6)
##  Minimum number data points required to perform a statistical
##  analysis for a point of interest.
min_entries = 100
##  Number of bins for the histogram when performing a statistical
##  analysis at a point of interest. Its value is station-dependent.
nbins = 80
##  The confidence levels that Elim has tested. When it is 5 sigma
##  or above, the model performance does not change much.
#cdf_limits = (0.05, 0.95)
#cdf_limits = (0.0067, 9933) # 3sigma
cdf_limits = (0.00023, 0.99977) # 5sigma
#cdf_limits = (0.000004, 0.9999966) # 6sigma
#cdf_limits = (1.9e-8, 0.999999981) # 7sigma
##   The buffer allowance (in meters) for two purposes:
##    1. a lenient confident interval when defining good range.
##    2. used for assigning a true label of is or is not spike.
buffer = 0.15

#########################################################
### defined functions
#########################################################
def plot_abs_delta (outpath, subdata):

    ''' Function to plot a histogram of absolute difference between accepted and
        raw water levels. Buffer threshold is defined based on this absolute
        difference. Before deciding the value of buffer, I plot the histogram to
        get a sense of what value is reasonable. Buffer needs to be small enough
        to exclude obvious spikes but large enough to include most points that
        are not spikes.

        input param
        -----------
        outpath (str): location of histogram PDF
        subdata (dataframe): water level data frame
    '''

    ### Build a histogram of the absolute delta.
    ##  Only include data points that have finite delta values.
    isfinite = numpy.isfinite (abs(subdata.delta.values))
    ##  Customize the bin settings. 67 bins between delta of 0 to delta of 10 inclusively.
    bins = numpy.linspace (0, 10, 68)
    ##  Build the histogram. Hist is the number of entries per bin, and edges is the values
    ##  of the bin edges.
    hist, edges = numpy.histogram (abs(subdata.delta.values[isfinite]), bins=bins)
    ##  Edges is the values of the bin edges. When plotting, each point should correspond
    ##  to the centers of bin edges.
    bincenter = edges[:-1] + (edges[1:] - edges[:-1])/2

    ### Plot the distribution
    ##  Set up canvas and subplot
    h  = plt.figure (figsize=(7.5, 5.5))
    gs = gridspec.GridSpec (1, 1)
    axis = h.add_subplot (gs[0])
    ##  Plot distribution
    axis.plot (bincenter, hist, drawstyle='steps-mid', color='blue', alpha=1.0, linewidth=2.0)
    ##  Plot the grid lines
    for ytick in axis.yaxis.get_majorticklocs():
        axis.axhline (y=ytick, color='gray', alpha=0.5, linestyle=':', linewidth=0.3)
    for xtick in axis.xaxis.get_majorticklocs():
        axis.axvline (x=xtick, color='gray', alpha=0.5, linestyle=':', linewidth=0.3)
    ##  Format x axis
    axis.set_xlim (0, 10)
    axis.tick_params (axis='x', labelsize=8)
    axis.set_xlabel ('Absolute difference between raw and accepted water level [meters]',
                     fontsize=12)
    ##  Format y axis
    axis.set_yscale('log')
    axis.set_ylim (ymin=0)
    axis.tick_params (axis='y', labelsize=8)
    axis.set_ylabel ('enteries', fontsize=12)
    ##  Save plot as PDF
    plt.suptitle ('Distribution of absolute difference between raw and accepted', fontsize=15)
    h.savefig (outpath + 'abs_delta.pdf')
    plt.close ('all')
    return

def plot_time_series (outpath, subdata, spikes=None, point=None):

    ''' Function to plot time series. When spikes and point are None, a full time
        series is plotted. Given 2 years of data with more than 170k points, this
        plot can take a while to plot. When either spikes or point is not None,
        subdata is expected to be a sub-set from the 2 years of data. If point is
        not None, a vertical line is drawn at the time of the point. If spikes is
        not None, vertical lines are drawn at the time of spikes. Either case, the
        top plot shows the time series data (accepted and raw). It may include the
        location of current point of interest and previously identified spikes. The
        bottom plot shows difference between accepted and raw, visually showing the
        location of possible spikes.

        input param
        -----------
        outpath (str): location of histogram PDF
        subdata (dataframe): water level data frame
        spikes (boolean array): It has the same length as subdata. The rows where
                                spikes are True are previously identified spikes.
        point (row of dataframe): Current point of interest.
    '''

    ### Define parameters depending on whether point / spike are None
    ##  1. filename: If point is available, filename is time_series_yyyymmddHHMM.
    ##               Otherwise, filename is time_series_full.
    ext = point.Time.strftime('%Y%m%d%H%M') if point is not None else 'full'
    filename = 'time_series_' + ext
    ##  2. title: If point is available, title is Water Level Time Series (yyyy/mm/dd HH:MM).
    ##            Otherwise, title is Water Level Time Series (full).
    ext = point.Time.strftime('%Y/%m/%d %H:%M') if point is not None else 'full'
    title = 'Water Level Time Series ({0})'.format(ext)
    ##  3. factor: This factor determines the location of time ticks.
    factor  = 0.01 if point is None else 4
    xtick_factor = int(interval/factor)
    ##  4. ybuffer: The extra extension for the y axis limit.
    ybuffer = 3    if point is None else 1
    ##  5. lwidth: The line width of the time series curve
    lwidth  = 0.1  if point is None else 2.0
    ##  6. alpha: The transparency of time series curve.
    alpha   = 0.3  if point is None else 0.5
    ##  7. markersize: The size of markers in the bottom plot.
    markersize = 1.0 if point is None else 2.0

    ### Organize dataframe for time-series plot
    ##  1. Change regular data frame to time-series data frame
    subdata.index = subdata.Time
    dataframe = subdata.drop('Time', axis=1)
    ##  2. Determine x-values. In time-series plot, x-values is sequential array.
    xvalues = numpy.arange (len (subdata))
    ##  3. Determine location of x-ticks on plot and x-range.
    is_xticks = xvalues % xtick_factor==0
    xlimit = [min(xvalues), max(xvalues)]

    ### Define canvas and subplot layout.
    h  = plt.figure (figsize=(12.5, 5.5))
    gs = gridspec.GridSpec (2, 1, height_ratios=[4,1])
    gs.update (hspace=0.15, bottom=0.2)

    ### Plot time series data (top)
    axis = h.add_subplot (gs[0])
    ##  Y-range depends on raw and accepted water level in subdata.
    ylimit = [numpy.inf, -numpy.inf]
    ##  Plot both raw and accepted data
    for ytype in ['raw', 'accepted']:
        yvalues = dataframe[ytype]
        color = 'black' if ytype=='accepted' else 'red'
        axis.plot (xvalues, yvalues, color=color, alpha=alpha, linewidth=lwidth, label=ytype)
        # Update y-range
        if min (yvalues) < ylimit[0]: ylimit[0] = min(yvalues) - 0.1
        if max (yvalues) > ylimit[1]: ylimit[1] = max(yvalues) + ybuffer
    ##  If point is available, plot current point of interest.
    if point is not None:
        index = numpy.where (dataframe.index == point.Time)
        axis.axvline (x=xvalues[index], color='blue', alpha=0.3, linestyle='-', linewidth=2)
    ##  If spike is available, plot location of previously identified spikes.
    if spikes is not None:
        for xvalue in xvalues[spikes]:
            axis.axvline (x=xvalue, color='red', alpha=0.3, linestyle=':', linewidth=0.5)
    ##  Add legend to plot
    axis.legend (loc=2, prop={'size':8})
    ##  Plot grid lines
    for ytick in axis.yaxis.get_majorticklocs():
        axis.axhline (y=ytick, color='gray', alpha=0.5, linestyle=':', linewidth=0.3)
    for xtick in axis.xaxis.get_majorticklocs():
        axis.axvline (x=xtick, color='gray', alpha=0.5, linestyle=':', linewidth=0.3)
    ##  Format x-axis
    axis.set_xlim (xlimit)
    axis.set_xticks (xvalues[is_xticks])
    axis.get_xaxis ().set_ticks ([])
    ##  Format y-axis
    axis.set_ylim (ylimit)
    axis.tick_params (axis='y', labelsize=8)
    axis.set_ylabel ('Water Level [meter]', fontsize=12)

    ### Plot difference between accepted and raw (bottom)
    axis = h.add_subplot (gs[1])
    ##  Plot the difference as scatter plot
    axis.scatter (xvalues, dataframe.delta, s=markersize, c='black', alpha=0.7)
    ##  If point is available, plot the current point of interest.
    if point is not None:
        index = numpy.where (dataframe.index == point.Time)
        axis.axvline (x=xvalues[index], color='blue', alpha=0.3, linestyle='-', linewidth=2)
    ##  If spike is available, plot location of previously identified spikes.
    if spikes is not None:
        for xvalue in xvalues[spikes]:
            axis.axvline (x=xvalue, color='red', alpha=0.3, linestyle=':', linewidth=0.5)
    ##  Plot a horizontal line indicating 0 difference line for reference.
    axis.axhline (y=0.0, color='blue', alpha=1.0, linestyle='-', linewidth=0.8)
    ##  Plot grid lines
    for ytick in axis.yaxis.get_majorticklocs():
        axis.axhline (y=ytick, color='gray', alpha=0.5, linestyle=':', linewidth=0.3)
    for xtick in axis.xaxis.get_majorticklocs():
        axis.axvline (x=xtick, color='gray', alpha=0.5, linestyle=':', linewidth=0.3)
    ## Format x-axis
    axis.set_xlim (xlimit)
    axis.set_xticks (xvalues[is_xticks])
    axis.set_xticklabels (dataframe.index[is_xticks].strftime ('%Y-%m-%d\n%Hhr'))
    axis.tick_params (axis='x', labelsize=8, labelrotation=30)
    ## Format y-axis
    axis.set_ylim (numpy.floor (min(dataframe.delta)), numpy.ceil (max(dataframe.delta)))
    axis.tick_params (axis='y', labelsize=8)
    axis.set_ylabel ('accepted - raw', fontsize=10)

    ### Store plot as PDF
    plt.suptitle (title, fontsize=15)
    h.savefig (outpath + filename + '.pdf')
    plt.close ('all')
    return

def build_histogram (subdata, previous_spikes, point, plot=False):

    ''' Function to perform a statistical analysis.
         1. build a histgram of raw water levels from input subdata.
         2. cumulate the normalized histogram from left to right.
         3. perform an interpolation function that returns a raw water
            level given a probability.
         4. obtain the confident interval at pre-defined threshold.
         5. modify the confident interval if needed.
         6. plot the distribution, CDF, and interpolated function.

        input param
        -----------
        subdata (dataframe): water level data frame
        previous_spike (boolean array):
        point (row of dataframe): Current point of interest.

        return param
        ------------
        confident_interval (list): [lower, upper] of confident intervals.
    '''

    ### Only include points that are finite and are not previously identified spikes.
    isfinite = numpy.isfinite (subdata.raw.values)
    if len (previous_spikes)>0:
        isfinite = numpy.logical_and (isfinite, ~previous_spikes)
    ### Build histogram with pre-defined number of bins.
    hist, edges = numpy.histogram (subdata.raw.values[isfinite], bins=nbins)
    ### Cumulate the normalized histogram from left to right.
    ##  x-value = binned raw water level
    ##  y-value = cumulative distribution probability
    bincenter = edges[:-1] + (edges[1:] - edges[:-1])/2
    cdf = numpy.cumsum (hist/numpy.sum(hist))
    ### Interpolate the inverted cumulative function
    ##  x-value = cumulative distribution probability
    ##  y-value = binned raw water level
    icdf = interp1d (cdf, bincenter, bounds_error=False)
    ### Obtain the confident interval based on pre-defined CDF limit
    lower, upper = icdf(cdf_limits[0]), icdf(cdf_limits[1])
    ##  If any of the two are infinite, replace it by the min / max of histogram bin.
    if not numpy.isfinite(lower): lower = min (bincenter)
    if not numpy.isfinite(upper): upper = max (bincenter)
    ##  Round the confident interval to 3 digits and add buffer allowance.
    lower, upper = [numpy.round (lower, 3)-buffer, numpy.round (upper, 3)+buffer]
    ### Plot distribution, CDF, and the interpolated function
    if plot:
        plot_distributions (point, bincenter, hist, lower, upper)
        plot_cdf (point, bincenter, hist, cdf_limits)
        plot_icdf (point, icdf, upper, lower, cdf_limits)
    return [lower, upper]

def plot_distributions (point, bincenter, hist, lower, upper):

    ''' Function to plot the distribution of raw water levels from data before
        current point of interest. The raw water level of the point of interest
        is also plotted as a vertical line. The confident interval is presented
        by the shaded area.

        input param
        -----------
        point (row of dataframe): Current point of interest.
        bincenter (array): the x-value (i.e. raw water level) of the histogram
        hist (array): the y-value (i.e. number of entries) of the histogram
        lower (float): the lower range of the confidernt interval
        upper (float): the upper range of the confidernt interval
    '''

    ### Define canvas and subplot
    h  = plt.figure (figsize=(7.5, 5.5))
    gs = gridspec.GridSpec (1, 1)
    axis = h.add_subplot (gs[0])
    ### Plot distribution
    axis.plot (bincenter, hist, drawstyle='steps-pre', color='black', alpha=1.0,
               linewidth=2.0, label='distribution')
    ### Plot raw WL at current point of interest
    axis.axvline (x=point.raw, color='blue', alpha=0.3, linestyle='-', linewidth=2)
    ### Shade lower and upper limits
    axis.fill_betweenx (y=[0, max(hist) + 2], x1=lower, x2=upper, color='red', alpha=0.1)
    ### Draw grid lines
    for ytick in axis.yaxis.get_majorticklocs():
        axis.axhline (y=ytick, color='gray', alpha=0.5, linestyle=':', linewidth=0.3)
    for xtick in axis.xaxis.get_majorticklocs():
        axis.axvline (x=xtick, color='gray', alpha=0.5, linestyle=':', linewidth=0.3)
    ### Add text of data
    text = 'Good range: [{0:.4f}, {1:.4f}]\nPoint of Interest: {2:.4f}'.format(lower, upper,
                                                                               point.raw)
    at = AnchoredText (text, prop=dict (size=8), frameon=True, loc=1)
    at.patch.set_boxstyle ("round,pad=0.,rounding_size=0.5")
    axis.add_artist(at)
    ### Add legend
    axis.legend (loc=2, prop={'size':8})
    ### Format y-axis
    axis.set_ylim (0, max(hist) + 2)
    axis.tick_params (axis='y', labelsize=8)
    axis.set_ylabel ('enteries', fontsize=12)
    ### Format x-axis
    xmin = min (point.raw, bincenter[0]) if numpy.isfinite (point.raw) else bincenter[0]
    xmax = max (point.raw, bincenter[-1]) if numpy.isfinite (point.raw) else bincenter[-1]
    axis.set_xlim (xmin, xmax)
    axis.tick_params (axis='x', labelsize=8)
    axis.set_xlabel ('Water Level [meter]', fontsize=12)
    ### Store plot as PDF
    plt.suptitle ('Distribution ({0})'.format(point.Time.strftime('%Y/%m/%d %H:%M')),
                  fontsize=15)
    h.savefig (outpath + 'distribution_' + point.Time.strftime('%Y%m%d%H%M') + '.pdf')
    plt.close ('all')
    return

def plot_cdf (point, bincenter, hist, cdf_limits):

    ''' Function to plot the cumulative distribution function, which is the cumulated,
        normalized histogram from left to right. The raw water level of the current
        point of interest is presented as a vertical line, and the confidence level
        in percentage is also shaded.

        input param
        -----------
        point (row of dataframe): Current point of interest.
        bincenter (array): the x-value (i.e. raw water level) of the histogram
        hist (array): the y-value (i.e. number of entries) of the histogram
        cdf_limits (list): [lower, upper] confidence level in percentage
    '''

    ### Define canvas and subplot
    h  = plt.figure (figsize=(7.5, 5.5))
    gs = gridspec.GridSpec (1, 1)
    axis = h.add_subplot (gs[0])
    ### Plot CDF
    cdf = numpy.cumsum (hist/numpy.sum(hist))
    axis.plot (bincenter, cdf, color='blue', linewidth=2.0)
    ### Shade the confident levels
    axis.fill_between (x=[min(bincenter), max(bincenter)], y1=cdf_limits[0], y2=cdf_limits[1],
                      color='red', alpha=0.1)
    ### Draw grid lines.
    for ytick in axis.yaxis.get_majorticklocs():
        axis.axhline (y=ytick, color='gray', alpha=0.5, linestyle=':', linewidth=0.3)
    for xtick in axis.xaxis.get_majorticklocs():
        axis.axvline (x=xtick, color='gray', alpha=0.5, linestyle=':', linewidth=0.3)
    ### Format y-axis
    axis.set_ylim (-0.1, 1.1)
    axis.tick_params (axis='y', labelsize=8)
    axis.set_ylabel ('cumulative distribution function', fontsize=12)
    ### Format x-axis
    axis.set_xlim (bincenter[0]-0.05, bincenter[-1]+0.05)
    axis.tick_params (axis='x', labelsize=8)
    axis.set_xlabel ('Water Level [meter]', fontsize=12)
    ### Store plot as PDF
    title = 'Cumulative distribution function ({0})'.format(point.Time.strftime('%Y/%m/%d %H:%M'))
    plt.suptitle (title, fontsize=15)
    h.savefig (outpath + 'cdf_' + point.Time.strftime('%Y%m%d%H%M') + '.pdf')
    plt.close ('all')
    return

def plot_icdf (point, icdf, upper, lower, cdf_limits):

    ''' Function to plot interpolate inverted cumulative distribution function. Unlike
        plot_cdf function, the x-axis here is the probaility, and the y-axis is the
        raw water level. The confident levels and intervals are also plotted as shaded
        area.

        input param
        -----------
        point (row of dataframe): Current point of interest.
        icdf (function): the interpolated inverted cumulative distribution function.
        upper (float): the confidence interval at the upper confidence level
        lower (float): the confidence interval at the lower confidence level
        cdf_limits (list): [lower, upper] confidence level in percentage
    '''

    ### Define canvas and subplot
    h  = plt.figure (figsize=(7.5, 5.5))
    gs = gridspec.GridSpec (1, 1)
    axis = h.add_subplot (gs[0])
    ### Plot the interpolated inverted cumulative distribution function
    ##  Define 1000 points between a probability of 0 and 1
    probabilities = numpy.linspace (0, 1, 1000)
    ##  Obtain the raw water levels for those 1000 probabilities
    raw_water_levels = icdf(probabilities)
    ##  Actually plot the inverted CDF.
    xvalues = probabilities[numpy.isfinite (raw_water_levels)]
    yvalues = raw_water_levels[numpy.isfinite (raw_water_levels)]
    axis.plot (xvalues, yvalues, color='blue', linewidth=2.0)
    ### Fill the confidence interval and range.
    axis.fill_betweenx (y=cdf_limits, x1=lower, x2=upper, color='red', alpha=0.1)
    ### Add a text box with actual numbers
    text = 'Good range: [{0:.4f}, {1:.4f}]\nPoint of Interest: {2:.4f}'.format(lower, upper,
                                                                               point.raw)
    at = AnchoredText (text, prop=dict (size=8), frameon=True, loc=2)
    at.patch.set_boxstyle ("round,pad=0.,rounding_size=0.5")
    axis.add_artist(at)
    ### Draw grid lines
    for ytick in axis.yaxis.get_majorticklocs():
        axis.axhline (y=ytick, color='gray', alpha=0.5, linestyle=':', linewidth=0.3)
    for xtick in axis.xaxis.get_majorticklocs():
        axis.axvline (x=xtick, color='gray', alpha=0.5, linestyle=':', linewidth=0.3)
    ### Format x-axis
    axis.set_xlim (-0.1, 1.1)
    axis.tick_params (axis='x', labelsize=8)
    axis.set_xlabel ('cumulative distribution function', fontsize=12)
    ### Format y-axis
    axis.set_ylim (min (yvalues)-0.05, max (yvalues)+0.05)
    axis.tick_params (axis='y', labelsize=8)
    axis.set_ylabel ('Water Level [meter]', fontsize=12)
    ### Store plot as PDF
    title = 'Inverted cumulative distribution function ({0})'.format(point.Time.strftime('%Y/%m/%d %H:%M'))
    plt.suptitle (title, fontsize=15)
    h.savefig (outpath + 'icdf_' + point.Time.strftime('%Y%m%d%H%M') + '.pdf')
    plt.close ('all')
    return

if __name__ == "__main__":

    #########################################################
    ### tidy up data
    #########################################################
    ##  Read dataframe from CSV file.
    data = pandas.read_csv(csvFile, skiprows=1)
    ##  Tidy up data with 5 columns:
    #     * Time: python-recognized date time
    #     * raw: observed water level
    #     * accepted: QC-ed water level
    #     * delta: accepted - raw
    #     * true_is_spike: truth label of is or is not spike. True spike are those with
    #                      absolute delta greater than a buffer allowance.
    time = pandas.to_datetime (data['Time'])
    data = data.drop (['Time', 'Time.1'], axis=1)
    data.columns  = ['raw', 'accepted']
    data['Time']  = time
    data['delta'] = data.accepted - data.raw
    data['true_is_spike'] = abs (data.delta) > buffer
    ##  To get a sense of data: plot the absolute delta and full time series.
    plot_abs_delta (outpath, deepcopy(data))
    plot_time_series (outpath, deepcopy(data), point=None, spikes=None)

    #########################################################
    ### run statistical analysis on each point
    #########################################################
    ##  Define a holder of predictions. By default, all data points are not spikes.
    are_spikes = [False] * len (data)

    ## Loop through each data point. Perform analysis and predict if the point is spike.
    for index, point in data.iterrows():
        # Skip the period so there is enough points for histogram
        if index < min_entries: continue
        # Extract the subset of data prior current point of interest.
        begin_index =  max (0, index-interval)
        end_index = max (0, index-1)
        subdata = data.iloc[begin_index:end_index, :]
        # Make sure enough valid entries is available before continue.
        nValid = len (subdata.raw[numpy.isfinite (subdata.raw)])
        if nValid < min_entries-1:
            print ('+------------------------------------------')
            print ('| {0}-th row: {1} ... '.format(index, point.Time))
            print ('|      {0} valid data before current point.'.format(nValid))
            print ('|      Not enough data points - skipping ...')
            continue
        # Figure out previously identified spikes.
        previous_spikes = numpy.array (are_spikes[begin_index:end_index])
        nspikes = len (previous_spikes[previous_spikes])
        # Compute the confident interval by performing a statistical analysis.
        limit_range = build_histogram (subdata, previous_spikes, point, plot=False)
        # Determine if this point is a spike based on confident interval
        this_is_spike = False
        if not pandas.isnull (point.raw):
            this_is_spike = point.raw < limit_range[0] or point.raw > limit_range[1]
        are_spikes[index] = this_is_spike
        # Report if 1. this point is identified as a spike OR
        #           2. this point is a true spike.
        if this_is_spike or index in data[data.true_is_spike.values].index:
            asubdata = data.iloc[begin_index:end_index+10, :]
            aprevious_spikes = numpy.array (are_spikes[begin_index:end_index+10])
            plot_time_series (outpath, asubdata, spikes=aprevious_spikes, point=point)
            alimit_range = build_histogram (asubdata, aprevious_spikes, point, plot=True)
            print ('+------------------------------------------')
            print ('| {0}-th row: {1} ... '.format(index, point.Time))
            print ('|      selected row index: {0} - {1}'.format(begin_index, end_index))
            print ('|      last {0} rows has {1} spikes'.format(len (subdata), nspikes))
            print ('|      histogram limits are {0} - {1}'.format(round (limit_range[0], 5),
                                                                  round (limit_range[1], 5) ))
            print ('|      this point is {0}; is spike? {1}'.format(round (point.raw, 5),
                                                                    this_is_spike))
            print ('|      Is this really a spike? {0} ({1}).'.format(data.true_is_spike.values[index], data.delta.values[index]))

    ## Add a column of prediction to the data frame
    data['pred_is_spike']= numpy.array (are_spikes)
    print ('')

    #########################################################
    ### Summary of model performance
    #########################################################
    ## Confusion Matrix
    #  Define arrays of boolean for is/not true spikes vs is/not predicted spikes
    istrue_notpred  = numpy.logical_and (data.true_is_spike, ~data.pred_is_spike)
    nottrue_ispred  = numpy.logical_and (~data.true_is_spike, data.pred_is_spike)
    nottrue_notpred = numpy.logical_and (~data.true_is_spike, ~data.pred_is_spike)
    istrue_ispred   = numpy.logical_and(data.true_is_spike, data.pred_is_spike)
    #  Find the number of entries for all 4 catogaries
    n_predno_trueno   = len (nottrue_notpred[nottrue_notpred])
    n_predyes_trueno  = len (nottrue_ispred[nottrue_ispred])
    n_predno_trueyes  = len (istrue_notpred[istrue_notpred])
    n_predyes_trueyes = len (istrue_ispred[istrue_ispred])
    #  Print them out as a confusion matrix
    print ('Confusion Matrix')
    print ('+-{0:8}-+-{0:8}-+-{0:8}-+'.format('-'*8))
    print ('| {0:8} | {1:8} | {2:8} |'.format(' '*8, 'pred no ', 'pred yes'))
    print ('+-{0:8}-+-{0:8}-+-{0:8}-+'.format('-'*8))
    print ('| {0:8} | {1:8} | {2:8} |'.format('true no ', n_predno_trueno, n_predyes_trueno))
    print ('+-{0:8}-+-{0:8}-+-{0:8}-+'.format('-'*8))
    print ('| {0:8} | {1:8} | {2:8} |'.format('true yes', n_predno_trueyes, n_predyes_trueyes))
    print ('+-{0:8}-+-{0:8}-+-{0:8}-+'.format('-'*8))
    print ('')

    ## Common measurement of model performance
    #  Calculate total number of different catogaries
    total = n_predno_trueno + n_predno_trueyes + n_predyes_trueno + n_predyes_trueyes
    total_trueyes = n_predno_trueyes + n_predyes_trueyes
    total_trueno  = n_predno_trueno  + n_predyes_trueno
    total_predyes = n_predyes_trueyes + n_predyes_trueno
    #  Caclulate different measurement
    accuracy = (n_predno_trueno + n_predyes_trueyes) / total
    precision = n_predyes_trueyes / total_predyes
    sensitivity = n_predyes_trueyes / total_trueyes
    error_rate = (n_predyes_trueno + n_predno_trueyes) / total
    true_negative_rate = n_predno_trueno / total_trueno
    false_positive_rate = n_predyes_trueno / total_trueno
    prevalence = (n_predno_trueyes + n_predyes_trueyes) / total
    #  Print them all out
    print ('{0:20}: {1:.5f}'.format('accuracy', accuracy))
    print ('{0:20}: {1:.5f}'.format('precision', precision))
    print ('{0:20}: {1:.5f}'.format('sensitivity', sensitivity))
    print ('{0:20}: {1:.5f}'.format('error rate', error_rate))
    print ('{0:20}: {1:.5f}'.format('true negative rate', true_negative_rate))
    print ('{0:20}: {1:.5f}'.format('false positive rate', false_positive_rate))
    print ('{0:20}: {1:.5f}'.format('prevalence', prevalence))
