# AUTHORS
# Jose PATINO, EURECOM, Sophia-Antipolis, France, 2019
# http://www.eurecom.fr/en/people/patino-jose
# Contact: patino[at]eurecom[dot]fr, josempatinovillar[at]gmail[dot]com

import os, sys, glob
import configparser
from diarizationFunctions import *
import numpy as np

def runDiarization(showName,config):      
    print('showName\t\t',showName)
    print('Extracting features')  
    
    if config.getint('GENERAL','performFeatureExtraction'):
        allData=extractFeatures(config['PATH']['audio']+showName+config['EXTENSION']['audio'],config.getfloat('FEATURES','framelength'),config.getfloat('FEATURES','frameshift'),config.getint('FEATURES','nfilters'),config.getint('FEATURES','ncoeff'))    
    else:
        allData=getFeatures(config['PATH']['features']+showName+config['EXTENSION']['features'])
    nFeatures = allData.shape[0]    
    print('Initial number of features\t',nFeatures) 
    
    if os.path.isfile(config['PATH']['UEM']+showName+config['EXTENSION']['UEM']):
        maskUEM = readUEMfile(config['PATH']['UEM'],showName,config['EXTENSION']['UEM'],nFeatures,config.getfloat('FEATURES','frameshift'))
    else:
        print('UEM file does not exist. The complete audio content is considered.')
        maskUEM = np.ones([1,nFeatures])     
        
    if os.path.isfile(config['PATH']['SAD']+showName+config['EXTENSION']['SAD']) and not(config.getint('GENERAL','performVAD')):
        maskSAD = readSADfile(config['PATH']['SAD'],showName,config['EXTENSION']['SAD'],nFeatures,config.getfloat('FEATURES','frameshift'),config['GENERAL']['SADformat']) 
    else:
        print('SAD file does not exist or automatic VAD is enabled in the config. VAD is applied and saved at %s.\n'%(config['PATH']['SAD']+showName+'.lab'))
        maskSAD = getSADfile(config,showName,nFeatures)
    mask = np.logical_and(maskUEM,maskSAD)    
    mask = mask[0][0:nFeatures]
    nSpeechFeatures=np.sum(mask)
    speechMapping = np.zeros(nFeatures)
    #you need to start the mapping from 1 and end it in the actual number of features independently of the indexing style
    #so that we don't lose features on the way
    speechMapping[np.nonzero(mask)] = np.arange(1,nSpeechFeatures+1)
    data=allData[np.where(mask==1)]
    del allData        
    segmentTable=getSegmentTable(mask,speechMapping,config.getint('SEGMENT','length'),config.getint('SEGMENT','increment'),config.getint('SEGMENT','rate'))
    numberOfSegments=np.size(segmentTable,0)
    print('Number of speech features\t',nSpeechFeatures)
    #create the KBM
    print('Training the KBM... ')
    #set the window rate in order to obtain "minimumNumberOfInitialGaussians" gaussians
    if np.floor((nSpeechFeatures-config.getint('KBM','windowLength'))/config.getint('KBM','minimumNumberOfInitialGaussians')) < config.getint('KBM','maximumKBMWindowRate'):
        windowRate = int(np.floor((np.size(data,0)-config.getint('KBM','windowLength'))/config.getint('KBM','minimumNumberOfInitialGaussians')))
    else:
        windowRate = int(config.getint('KBM','maximumKBMWindowRate'))        
    poolSize = np.floor((nSpeechFeatures-config.getint('KBM','windowLength'))/windowRate)
    if  config.getint('KBM','useRelativeKBMsize'):
        kbmSize = int(np.floor(poolSize*config.getfloat('KBM','relKBMsize')))
    else:
        kbmSize = int(config.getint('KBM','kbmSize'))        
    print('Training pool of',int(poolSize),'gaussians with a rate of',int(windowRate),'frames')    
    kbm, gmPool = trainKBM(data,config.getint('KBM','windowLength'),windowRate,kbmSize )    
    print('Selected',kbmSize,'gaussians from the pool')    
    Vg = getVgMatrix(data,gmPool,kbm,config.getint('BINARY_KEY','topGaussiansPerFrame'))    
    print('Computing binary keys for all segments... ')
    segmentBKTable, segmentCVTable = getSegmentBKs(segmentTable, kbmSize, Vg, config.getfloat('BINARY_KEY','bitsPerSegmentFactor'), speechMapping)    
    print('Performing initial clustering... ')
    initialClustering = np.digitize(np.arange(numberOfSegments),np.arange(0,numberOfSegments,numberOfSegments/config.getint('CLUSTERING','N_init')))
    print('done')
    print('Performing agglomerative clustering... ')    
    if config.getint('CLUSTERING','linkage'):
        finalClusteringTable, k = performClusteringLinkage(segmentBKTable, segmentCVTable, config.getint('CLUSTERING','N_init'), config['CLUSTERING']['linkageCriterion'], config['CLUSTERING']['metric'])
    else:
        finalClusteringTable, k = performClustering(speechMapping, segmentTable, segmentBKTable, segmentCVTable, Vg, config.getfloat('BINARY_KEY','bitsPerSegmentFactor'), kbmSize, config.getint('CLUSTERING','N_init'), initialClustering, config['CLUSTERING']['metric'])        
    print('Selecting best clustering...')
    if config['CLUSTERING_SELECTION']['bestClusteringCriterion'] == 'elbow':
        bestClusteringID = getBestClustering(config['CLUSTERING_SELECTION']['metric_clusteringSelection'], segmentBKTable, segmentCVTable, finalClusteringTable, k)
    elif config['CLUSTERING_SELECTION']['bestClusteringCriterion'] == 'spectral':
        bestClusteringID = getSpectralClustering(config['CLUSTERING_SELECTION']['metric_clusteringSelection'],finalClusteringTable,config.getint('CLUSTERING','N_init'),segmentBKTable,segmentCVTable,k,config.getint('CLUSTERING_SELECTION','sigma'),config.getint('CLUSTERING_SELECTION','percentile'),config.getint('CLUSTERING_SELECTION','maxNrSpeakers'))+1        
    print('Best clustering:\t',bestClusteringID.astype(int))
    print('Number of clusters:\t',np.size(np.unique(finalClusteringTable[:,bestClusteringID.astype(int)-1]),0))    
    if config.getint('RESEGMENTATION','resegmentation') and np.size(np.unique(finalClusteringTable[:,bestClusteringID.astype(int)-1]),0)>1:
        print('Performing GMM-ML resegmentation...')
        finalClusteringTableResegmentation,finalSegmentTable = performResegmentation(data,speechMapping, mask,finalClusteringTable[:,bestClusteringID.astype(int)-1],segmentTable,config.getint('RESEGMENTATION','modelSize'),config.getint('RESEGMENTATION','nbIter'),config.getint('RESEGMENTATION','smoothWin'),nSpeechFeatures)
        print('done')
        getSegmentationFile(config['OUTPUT']['format'],config.getfloat('FEATURES','frameshift'),finalSegmentTable, np.squeeze(finalClusteringTableResegmentation), showName, config['EXPERIMENT']['name'], config['PATH']['output'], config['EXTENSION']['output'])
    else:
        getSegmentationFile(config['OUTPUT']['format'],config.getfloat('FEATURES','frameshift'),segmentTable, finalClusteringTable[:,bestClusteringID.astype(int)-1], showName, config['EXPERIMENT']['name'], config['PATH']['output'], config['EXTENSION']['output'])      
    
    if config.getint('OUTPUT','returnAllPartialSolutions'):
        if not os.path.isdir(config['PATH']['output']):
            os.mkdir(config['PATH']['output'])
        outputPathInd = config['PATH']['output']+ config['EXPERIMENT']['name'] + '/' + showName + '/'
        if not os.path.isdir(config['PATH']['output'] + config['EXPERIMENT']['name']):
            os.mkdir(config['PATH']['output'] + config['EXPERIMENT']['name'])
        if not os.path.isdir(outputPathInd):            
            os.mkdir(outputPathInd)
        for i in np.arange(k):
            getSegmentationFile(config['OUTPUT']['format'],config.getfloat('FEATURES','frameshift'), segmentTable, finalClusteringTable[:,i], showName, showName+'_'+str(np.size(np.unique(finalClusteringTable[:,i]),0))+'_spk', outputPathInd, config['EXTENSION']['output'])        
            
    print('\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    
if __name__ == "__main__":     
    # If a config file in INI format is passed by argument line then it's used. 
    # For INI config formatting please refer to https://docs.python.org/3/library/configparser.html
    if len(sys.argv) >= 2:
        configFile = sys.argv[1]
    else:
        configFile = 'config.ini'    
    config = configparser.ConfigParser()
    config.read(configFile)
    
    if config.getint('GENERAL','performFeatureExtraction'):
        # Audio files are searched at the corresponding folder
        showNameList = sorted(glob.glob(config['PATH']['audio']+'*'+config['EXTENSION']['audio']))
    else:
        # Feature files are searched if feature extraction is disabled:
        showNameList = sorted(glob.glob(config['PATH']['features']+'*'+config['EXTENSION']['features']))
        
    # If the output file already exists from a previous call it is deleted
    if os.path.isfile(config['PATH']['output']+config['EXPERIMENT']['name']+config['EXTENSION']['output']):
        os.remove(config['PATH']['output']+config['EXPERIMENT']['name']+config['EXTENSION']['output'])
                
    # Output folder is created
    if not os.path.isdir(config['PATH']['output']):
        os.mkdir(config['PATH']['output'])

    # Files are diarized one by one
    for idx,showName in enumerate(showNameList):
        print('\nProcessing file',idx+1,'/',len(showNameList))
        runDiarization(os.path.splitext(os.path.basename(showName))[0],config)