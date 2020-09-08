
"""
This is a module with functions that I repeatedly call when running ML models in jupyter lab.  
This helps make the notebooks cleaner since these are changed very often
"""
from scipy import io
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import datetime

# A function to assess the bad and good data points (0 or 1) for the training, testing and total data sets
def assessTrainTestData(trainOrTestData):
    
    totalData = trainOrTestData.shape[0]
    badData = totalData-trainOrTestData['TARGET'].sum()
    fracBadData=badData / totalData
    
    return totalData,badData



# a function to plot the confusion matrix
def plotConfusionMatrix(cnfMatrix, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                        ):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = cnfMatrix
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    
    #Calculate the total accuracy - for all points
    print('total accuracy = '+
          '{:.4f}'.format((cnfMatrix[0,0]+cnfMatrix[1,1])/cnfMatrix.sum())
         )
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)

    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 verticalalignment='center',
                 color="white" if cm[i, j] > thresh else "black")

    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    xtick_marks = np.array([0,1])
    plt.xticks(xtick_marks, classes)
    ytick_marks=np.array([-.5,0,1,1.5])
    ylabels=['',classes[0],classes[1],'']
    plt.yticks(ytick_marks, ylabels,rotation=0)
    plt.colorbar()
    return

#A function to save pandas dataframes into .mat files - specifically the modelOut, predictFeatures and Time
def pandasToMat(modelOut, predictFeatures, outfileName):

    #Outfile name is the name appended to the front of the .mat output filenames

    timeOut = modelOut.reset_index()['time']
    test=timeOut.dt.strftime('%d-%b-%Y %H:%M:%S')
    test=pd.DataFrame(test)
    test=test.to_dict('list')
    io.savemat(file_name = outfileName + '_time.mat', mdict = test)

    io.savemat(file_name = outfileName + '_modelOut.mat', mdict = modelOut.to_dict('list'))
    io.savemat(file_name = outfileName + '_predictFeatures.mat', mdict = predictFeatures.to_dict('list'))
    
    return

# Function to calculate the Brier Skill Score
def BSS(targClim, modPred, observed):
    
    #print(type(modPred))
    #print(type(observed))
    bs=np.mean((modPred-observed)**2)

    #Calculate the BSS
    climate=np.mean(targClim)

    climForecast=np.ones(np.shape(observed)) * climate

    bsClimate=np.mean((climForecast-observed)**2)

    bssOut=1-(bs/bsClimate)
    
    return bssOut

# Function to load the cleaned datafile for a station
def loadCleanedData(stationNum, fileType, dataDirectory):

    # Where stationNum is the wl station number to load
    # filetype is 'test','train' or 'validation'
    # dataDirectory is the directory where the test, train, validation sub-directories are located
    # ex: dataDirectory ='/jupyter/userhomes/dusek/waterlevelAI/data'
    
    filenameIn = dataDirectory + '/' + fileType + '/' + stationNum + '_processed_ver_merged_wl_' + fileType + '.csv'
    
    dataIn = pd.read_csv(filenameIn, index_col=1, parse_dates=True,
                        usecols=['STATION_ID','DATE_TIME','SENSOR_USED_PRIMARY','PRIMARY','PRIMARY_TRUE','PRIMARY_SIGMA',
                                'PRIMARY_SIGMA_TRUE','PRIMARY_RESIDUAL','BACKUP','BACKUP_TRUE','BACKUP_SIGMA',
                                'BACKUP_SIGMA_TRUE','BACKUP_RESIDUAL','PREDICTION','VERIFIED','TARGET',
                                'NEIGHBOR_PRIMARY','NEIGHBOR_PREDICTION','NEIGHBOR_PRIMARY_RESIDUAL','NEIGHBOR_TARGET']
                        )
    dataIn.index.name ='time'
    
    return dataIn   

#Function to resample the yes/no (1,0) distrubution of data points for each station
def resampleGoodPointsSetBad(fractionNo, dataToAdjust):
    
    #Where fractionNo is the fraction of no that we want, in initial prototype this was = 0.10
    #Where dataToAdjust is the pandas dataframe of the training data that we want to resample
    
    #Count the bad data points first
    totalData = dataToAdjust.shape[0]
    badDataCount = totalData-dataToAdjust['TARGET'].sum()

    #What is the number of yes points we need to have given that No fraction
    #Calculate the no fraction OR at least 10% of the total data
    numYes = int(badDataCount // fractionNo - badDataCount)

    # Divide by class
    class_0 = dataToAdjust[dataToAdjust['TARGET'] == 0]
    class_1 = dataToAdjust[dataToAdjust['TARGET'] == 1]
    
    #Now resample the yes or 1s to be 90% of total training data
    class_1_resample = class_1.sample(numYes)
    dataResample = pd.concat([class_1_resample, class_0], axis=0)

    print('Random under-sampling:')
    print(dataResample.TARGET.value_counts())
    
    return dataResample


#Function to resample the yes/no (1,0) distrubution of data points for each station
def resampleGoodPointsSetNum(totalPoints, dataToAdjust):
    
    #Where totalPoints is the total Number of data points we want from each station
    #Where dataToAdjust is the pandas dataframe of the training data that we want to resample
    
    #Count the bad data points first
    totalData = dataToAdjust.shape[0]
    badDataCount = totalData-dataToAdjust['TARGET'].sum()

    #What is the number of yes points we need to reach the desired total
    numYes=int(totalPoints-badDataCount);
    
    # Divide by class
    class_0 = dataToAdjust[dataToAdjust['TARGET'] == 0]
    class_1 = dataToAdjust[dataToAdjust['TARGET'] == 1]
    
    #Now resample the yes or 1s to be 90% of total training data
    class_1_resample = class_1.sample(numYes)
    dataResample = pd.concat([class_1_resample, class_0], axis=0)

    print('Random under-sampling:')
    print(dataResample.TARGET.value_counts())
    
    return dataResample

