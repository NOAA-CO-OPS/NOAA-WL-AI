## Data Cleaner

Data Cleaner is a package that performs data cleaning for WL-AI project. All cleaning steps are documented in [this Google doc](https://docs.google.com/document/d/1BfyIQE9GXPCRbBSkyurd3UeGqpGkAr1UYkMZzh5LBNk/edit?usp=sharing).

### Installation

This package was created using python 3.7. Typical packages are required.
* numpy 1.19.0
* pandas 1.0.5
* matplotlib 3.2.2
* scipy 1.5.2

### Usage

Prior executing this script, one must do the following steps.

1. Download the latest WL-AI Station List from [this Google spreadsheet](https://docs.google.com/spreadsheets/d/1tLoaNPWNnHneWOZlpS38S7ldlkSs0wiCq_E6_39u_Qg/edit?usp=sharing) as a csv file. 

2. Copy Armin's zipped raw file from CO-OPS Common (CO-OPS_Common\CODE\AI-data-retrieval\data) to your local desktop. Unzip all files to a location. For each station, 3 files must exist: _raw_ver_merged_wl.csv, _offsets.csv, and _B1_gain_offsets.csv.

After that, open a terminal or command prompt. 

```
> cd C:\to\where\data\cleaning\package\is
> python clean_data.py --raw_path 'C:\\where\\armin\\raw\\data\\is\\' \
                       --proc_path 'C:\\where\\you\\want\\processed\\data\\to\\live\\' \
                       --station_info_csv 'C:\\path\\to\\WLAIStationList.csv' \
                       --log_level info \
                       --do_midstep_files 
```

* --log-level accepts info, debug, warn, and error
* --do_midstep_files is optional. If raised, store stats csv files and generate plots.