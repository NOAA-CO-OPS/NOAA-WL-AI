
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

# A function to assess the bad and good data points (0 or 1) for the training, testing and total data sets
def assessTrainTestData(trainData, testData):
    
    totalTrainData = trainData.shape[0]
    badTrainData = totalTrainData-trainData['targets'].sum()
    fracTrainData=badTrainData / totalTrainData

    totalTestData = testData.shape[0]
    badTestData = totalTestData-testData['targets'].sum()
    fracTestData=badTestData / totalTestData

    print('Training Data Points')
    print(totalTrainData)

    print('Data manually QCed as 0 or bad')
    print(badTrainData)

    print('Fraction of data points which are bad')
    print(fracTrainData)

    print('\n')

    print('Testing Data Points')
    print(totalTestData)

    print('Data manually QCed as 0 or bad')
    print(badTestData)

    print('Fraction of data points which are bad')
    print(fracTestData)
    
    return totalTrainData,badTrainData



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