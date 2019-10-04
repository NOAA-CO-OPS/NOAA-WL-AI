# Overview

This task aims at identifying obvious spikes via statistical approach. When QC-ing water levels, analysts usually know the rough range of water level at a given station. This exercise puts that statistical aspect into codes. The basic concept involves building a distribution function at a given data point using data a day before the data point. A confidence interval is determined given some confidence level threshold, and the data point is considered as a spike if its observed water level is beyond the interval. This exercise is under the guidance of Katerina Glebushko with the help from Lindsay Abrams. They both have the script to explore e.g. different parameters and different stations. Eventually, it can be added to the overall AI project as a feature from the statistical point of view.

# Requirements

As of the first draft, only 1 script, identify_spikes.py, is written. It is expected to run with python3. The required packages are
- pandas
- numpy
- scipy
- matplotlib
- mpl_toolkits

If anaconda is installed, the first 4 packages are already available, except mpl_toolkits. In that case, import AnchoredText with matplotlib.offsetbox instead of mpl_toolkits. 

# Execute

### Step 1. Change internal paths
Open your preferred text editor, and change lines 33 and 35. Line 33 should be the folder where the script can dump out plots. Line 35 is the location of the input CSV file. An example CSV file can be found in supplementary/8772471_20180801_20190731.csv. This CSV file is downloaded directly from WALI.


### Step 2. Execute
To run it, do

``` $ python3 identify_spikes.py
```

Depending on how you set up your environment, the program name may be python or python3 (especially if your system has python 2.7 installed previously...).

# Results

Using the CSV file in supplementary/, the first draft of statistical analysis identifies 195 spikes given a total of 199 true spikes. On the down side, it misses 25 true spikes and mis-identifies 4 (0.002%) false-positive spikes. In short, it has a sensitivity (or true positive rate) of 88.6% and an accuracy of 99.983%. 

A few parameters are tweaked to obtain the best performance. These includes nbins (number of bins when building histograms), cdf_limits (the confidence levels to determine the confident range), buffer (to accept small differences in water levels), and the min_entries (the threshold on number of entries when building histogram). Model performance is most sensitive to nbins, which has a default value of 80, because it changes the resolution of the histogram and, hence, the CDF and the confidence interval. Buffer allowance also reduces a lot of the false positive cases due to small differences between accepted and raw water levels. Cdf_limits has a limited effect if it is set higher 5σ. In the near future, one can analyze a variety of stations with different behavior to ensure those values are valid across all stations. If it is not the case, a minimizer / more complicated procedure can be built to find the best values for those parameters to maximize performance per station.

# More information

For more information on the steps, please visit the juputer-notebook in supplementary/.