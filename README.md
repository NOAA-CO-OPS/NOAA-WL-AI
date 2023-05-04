# NOAA-WL-AI
Repository for the development of an AI QC and data fill system for NOAA water level observations

## NOAA AI Water Level Processing

* PI: Greg Dusek
* CO-OPS Team: Armin Pruessner, George Story, Elim Thompson, Lindsay Abrams, Erick DiFiore (OCS)
* External Team: Philippe Tissot - CBI, Texas A&M CC; Xu Chen - COAPS, Florida State
* Git repository: https://github.com/NOAA-CO-OPS/NOAA-WL-AI
* Google doc: https://docs.google.com/document/d/1YvMKjcHizZp5HMJsiyQvzoB4vWxkbQqT-bZIJAyaHAY/edit

### Important Note
Please leave this initial jupyter notebooks I developed un-edited, as I want to retain those results for future reference.  Instead please create new notebooks if attempting new approaches or using new data.  Thanks!

### Important Updates
2021/4/21 - Removed AI-data-retrieval directory from GitHub repository. This directory will be maintained and accessed via the CODE folder on CO-OPS Common for now. 

### Goal

To create an AI system which will complete much of the manual processing and verification steps presently needed for NOAA 6 minute water level observations.  The system should be able to: 1) successfully classify 6 minute water level observations from the primary sensor as good or bad and 2) fill gaps in the resultant time series following standard CO-OPS protocols.  The system should be able to perform both of these steps in near real-time and at an accuracy level close to what is done by humans.

### Background

Generating high-quality, continuous time series of 6-minute water level observations from raw tide gauge data is notoriously challenging. Data can be impacted by sensor, communication or other problems due to a range of issues such as extreme events, aging instrumentation and accidents. Though some problematic or spurious data are removed during initial automated quality-control (QC) steps, questionable water level values often remain in the records prior to human investigation. Once problematic data points are removed from the primary water level sensor time series, considerable effort and care are still required to fill the resultant data gaps with observations from the back-up sensor, nearby neighbor sensors, tide predictions, statistical fit of data around the gap or a combination thereof. The NOAA National Ocean Service (NOS) Center for Operational Oceanographic Products and Services (CO-OPS) operates and maintains over 250 real-time water level stations across the coastal U.S. and Great Lakes. This equates to over 1.8 million 6-minute water level observations per month which are processed, quality controlled and reviewed. CO-OPS presently relies on a combination of automated and manual processes, however a substantial amount of human intervention is required to generate a complete, verified water level time series. An AI water level processing system has the potential to substantially reduce the work-hours required to produce high-quality water level time series, while improving the lag between data collection and verification to near real-time.

This project represents an initial investigation into the application of Artificial Intelligence (AI) approaches to process, QC and fill water level observations. A range of Machine Learning (ML) techniques are explored, including regression, random forests and artificial neural networks to assess performance and skill compared to target data sets of manually verified water level time series. We begin with prototyping using a subset of 5 NWLON stations, and once initial skill is determined and the initial approach is demonstrated to be valid, we will expand to additional and sometimes more complicated station types.  Eventually the AI system should be applicable to all coastal tidal stations.  Great Lakes stations will be explored after this initial application, as the lack of tide predictions will make assessing those stations more challenging.

### Proposed Timeline

* Aug 2019 - Initial scoping and prototype data compiled (DONE)
* Sep 2019 - Initial data cleaning of prototype data completed and some initial ideas explored at OceanHackWeek 2019 (DONE)
* Oct 2019 - Project documentation, github repository started (DONE)
* Dec 2019 - Initial results should be completed for 5 prototype stations (DONE)
* Jan 2020 - AMS annual meeting presentation on initial results (DONE - [find the recorded pres here](https://ams.confex.com/ams/2020Annual/meetingapp.cgi/Paper/365782))
* Mar 2020 - Finalize initial approach and begin scoping application to larger dataset (working on this now)
* Jun 2020 - Create HPC Cloud Jupyter Hub for ML training and clean 50 station training/validation/testing data set
* TBD - Complete the prototype model for data QC 
* TBD - Complete the prototype model for data filling
* TBD - Complete peer reviewed journal article or NOAA tech report documenting the approach
* TBD - Demonstrate approach on all NWLON stations.  If successful, begin planning for operational implementation in FY22


For additional information, contact:
Gregory Dusek,
NOAA Center for Operational Oceanographic Products and Services,
gregory.dusek@noaa.gov

## NOAA Open Source Disclaimer

This repository is a scientific product and is not official communication of the National Oceanic and Atmospheric Administration, or the United States Department of Commerce. All NOAA GitHub project code is provided on an ?as is? basis and the user assumes responsibility for its use. Any claims against the Department of Commerce or Department of Commerce bureaus stemming from the use of this GitHub project will be governed by all applicable Federal law. Any reference to specific commercial products, processes, or services by service mark, trademark, manufacturer, or otherwise, does not constitute or imply their endorsement, recommendation or favoring by the Department of Commerce. The Department of Commerce seal and logo, or the seal and logo of a DOC bureau, shall not be used in any manner to imply endorsement of any commercial product or activity by DOC or the United States Government.

## License

Software code created by U.S. Government employees is not subject to copyright in the United States (17 U.S.C. �105). The United States/Department of Commerce reserve all rights to seek and obtain copyright protection in countries other than the United States for Software authored in its entirety by the Department of Commerce. To this end, the Department of Commerce hereby grants to Recipient a royalty-free, nonexclusive license to use, copy, and create derivative works of the Software outside of the United States.
