# AUTHORS
# Jose PATINO, EURECOM, Sophia-Antipolis, France, 2019
# http://www.eurecom.fr/en/people/patino-jose
# Contact: patino[at]eurecom[dot]fr, josempatinovillar[at]gmail[dot]com

# DIARIZATION FUNCTIONS

import numpy as np
    
def extractFeatures(audioFile,framelength,frameshift,nfilters,ncoeff):
    import librosa
    y, sr = librosa.load(audioFile,sr=None)
    frame_length_inSample=framelength*sr
    hop = int(frameshift*sr)
    NFFT=int(2**np.ceil(np.log2(frame_length_inSample)))
    if sr>=16000:
        features=librosa.feature.mfcc(y=y, sr=sr,dct_type=2,n_mfcc=ncoeff,n_mels=nfilters,n_fft=NFFT,hop_length=hop,fmin=20,fmax=7600).T
    else:
        features=librosa.feature.mfcc(y=y, sr=sr,dct_type=2,n_mfcc=ncoeff,n_mels=nfilters,n_fft=NFFT,hop_length=hop).T
    return features

def getFeatures(featureFile):
    features = HTKFile()
    features.load(featureFile)
    allData = np.asarray(features.data)
    return allData

def readUEMfile(path,filename,ext,nFeatures,frameshift):    
    uemFile = path + filename + ext
    notEvaluatedFramesMask=np.zeros([1, nFeatures])    
    f=open(uemFile,'r')
    C=f.read().splitlines()
    f.close()    
    initTime = []
    endTime = []
    if C:
        if len(C[0])>1:
            for idx,x in enumerate(C):        
                initTime=np.append(initTime,float(x.strip().split(' ')[2]))
                endTime=np.append(endTime,float(x.strip().split(' ')[3]))
    else:
        print('UEM annotations file was empty. The complete file is considered evaluable.')
        
    if type(initTime) is list:
        notEvaluatedFramesMask = np.ones([1,nFeatures])
    else:
        initTime = np.floor(initTime/frameshift)        
        endTime = np.floor(endTime/frameshift)        
        for idxT,x in np.ndenumerate(initTime):
            it=initTime[idxT]        
            if endTime[idxT]>nFeatures:
                et = nFeatures
            else:
                et = endTime[idxT]        
            notEvaluatedFramesMask[0,int(it):int(et)]=1
    return notEvaluatedFramesMask
  
def readSADfile(path,filename,ext,nFeatures,frameshift, format):   
    sadFile = path + filename + ext
    notEvaluatedFramesMask=np.zeros([1, nFeatures])    
    f=open(sadFile,'r')
    C=f.read().splitlines()
    f.close()    
    initTime = []
    endTime = []
    if C:
        if len(C[0])>1:
            if format=='LBL':
                for idx,x in enumerate(C):        
                    initTime=np.append(initTime,float(x.strip().split(' ')[0]))
                    endTime=np.append(endTime,float(x.strip().split(' ')[1]))
            elif format=='RTTM':
                for idx,x in enumerate(C):        
                    initTime=np.append(initTime,float(x.strip().split(' ')[3]))
                    endTime=np.append(endTime,float(x.strip().split(' ')[3])+float(x.strip().split(' ')[4]))         
            elif format=='MDTM':
                for idx,x in enumerate(C):        
                    initTime=np.append(initTime,float(x.strip().split(' ')[2]))
                    endTime=np.append(endTime,float(x.strip().split(' ')[2])+float(x.strip().split(' ')[3]))
    else:
        print('SAD annotations file was empty. The complete file is considered speech.')
    
    if type(initTime) is list:
        notEvaluatedFramesMask = np.ones([1,nFeatures])        
    else:        
        initTime = np.round(initTime/frameshift)        
        endTime = np.round(endTime/frameshift)        
        for idxT,x in np.ndenumerate(initTime):
            it=initTime[idxT]        
            if endTime[idxT]>nFeatures:
                et = nFeatures
            else:
                et = endTime[idxT]        
            notEvaluatedFramesMask[0,int(it):int(et)]=1
    return notEvaluatedFramesMask

def getSADfile(config,filename,nFeatures):
    
    """ The following VAD applying procedure is adapted from the repository
    provided as part of the DIHARD II challenge speech enhancement system:    
    https://github.com/staplesinLA/denoising_DIHARD18/blob/master/main_get_vad.py"""
    
    import os
    from librosa import load
    data, sr = load(config['PATH']['audio']+filename+config['EXTENSION']['audio'],sr=None)    
    va_framed = py_webrtcvad(data, fs=sr, fs_vad=sr, hoplength=30, vad_mode=0)
    segments = get_py_webrtcvad_segments(va_framed,sr)
    if not os.path.isdir(config['PATH']['SAD']):
        os.mkdir(config['PATH']['SAD'])     
    if os.path.isfile(config['PATH']['SAD']+filename+'.lab'):
        os.remove(config['PATH']['SAD']+filename+'.lab')    
    output_file = open(config['PATH']['SAD']+filename+'.lab','w')
    for i in range(segments.shape[0]):
        start_time = segments[i][0]
        end_time =  segments[i][1]
        output_file.write("%.3f %.3f speech\n" % (start_time, end_time))
    output_file.close()
    maskSAD = readSADfile(config['PATH']['SAD'],filename,'.lab',nFeatures,config.getfloat('FEATURES','frameshift'),'LBL')   
    return maskSAD

def py_webrtcvad(data, fs, fs_vad, hoplength=30, vad_mode=0):    
    import webrtcvad
    from librosa.core import resample
    from librosa.util import frame
    """ Voice activity detection.
    This was implementioned for easier use of py-webrtcvad.
    Thanks to: https://github.com/wiseman/py-webrtcvad.git
    Parameters
    ----------
    data : ndarray
        numpy array of mono (1 ch) speech data.
        1-d or 2-d, if 2-d, shape must be (1, time_length) or (time_length, 1).
        if data type is int, -32768 < data < 32767.
        if data type is float, -1 < data < 1.
    fs : int
        Sampling frequency of data.
    fs_vad : int, optional
        Sampling frequency for webrtcvad.
        fs_vad must be 8000, 16000, 32000 or 48000.
        Default is 16000.
    hoplength : int, optional
        Step size[milli second].
        hoplength must be 10, 20, or 30.
        Default is 0.1.
    vad_mode : int, optional
        set vad aggressiveness.
        As vad_mode increases, it becomes more aggressive.
        vad_mode must be 0, 1, 2 or 3.
        Default is 0.
    Returns
    -------
    vact : ndarray
        voice activity. time length of vact is same as input data.
        If 0, it is unvoiced, 1 is voiced.
    """

    # check argument
    if fs_vad not in [8000, 16000, 32000, 48000]:
        raise ValueError('fs_vad must be 8000, 16000, 32000 or 48000.')

    if hoplength not in [10, 20, 30]:
        raise ValueError('hoplength must be 10, 20, or 30.')

    if vad_mode not in [0, 1, 2, 3]:
        raise ValueError('vad_mode must be 0, 1, 2 or 3.')

    # check data
    if data.dtype.kind == 'i':
        if data.max() > 2**15 - 1 or data.min() < -2**15:
            raise ValueError(
                'when data type is int, data must be -32768 < data < 32767.')
        data = data.astype('f')

    elif data.dtype.kind == 'f':
        if np.abs(data).max() >= 1:
            data = data / np.abs(data).max() * 0.9
            warnings.warn('input data was rescaled.')
        data = (data * 2**15).astype('f')
    else:
        raise ValueError('data dtype must be int or float.')

    data = data.squeeze()
    if not data.ndim == 1:
        raise ValueError('data must be mono (1 ch).')

    # resampling
    if fs != fs_vad:
        resampled = resample(data, fs, fs_vad)
    else:
        resampled = data

    resampled = resampled.astype('int16')

    hop = fs_vad * hoplength // 1000
    framelen = resampled.size // hop + 1
    padlen = framelen * hop - resampled.size
    paded = np.lib.pad(resampled, (0, padlen), 'constant', constant_values=0)
    framed = frame(paded, frame_length=hop, hop_length=hop).T    
    
    vad = webrtcvad.Vad()
    vad.set_mode(vad_mode)
    valist = [vad.is_speech(tmp.tobytes(), fs_vad) for tmp in framed]

    hop_origin = fs * hoplength // 1000
    va_framed = np.zeros([len(valist), hop_origin])
    va_framed[valist] = 1

    return va_framed.reshape(-1)[:data.size]

def get_py_webrtcvad_segments(vad_info,fs):
    vad_index = np.where( vad_info==1.0) # find the speech index   
    vad_diff = np.diff( vad_index)
            
    vad_temp = np.zeros_like(vad_diff)
    vad_temp[ np.where(vad_diff==1) ]  = 1       
    vad_temp =  np.column_stack( (np.array([0]), vad_temp, np.array([0]) ))
    final_index = np.diff(vad_temp)
            
    starts = np.where( final_index == 1)
    ends = np.where( final_index == -1)
      
    sad_info = np.column_stack( (starts[1], ends[1]) ) 
    vad_index = vad_index[0]
    
    segments = np.zeros_like(sad_info,dtype=np.float)
    for i in range(sad_info.shape[0]):
        segments[i][0] = float( vad_index [ sad_info[i][0] ] ) / fs
        segments[i][1] =  float( vad_index[ sad_info[i][1] ]  +1 ) / fs
 
    return  segments  # present in seconds
  
def getSegmentTable(mask, speechMapping, wLength, wIncr, wShift):
    changePoints,segBeg,segEnd,nSegs = unravelMask(mask)  
    segmentTable = np.empty([0,4])
    for i in range(nSegs):
        begs = np.arange(segBeg[i],segEnd[i],wShift)
        bbegs = np.maximum(segBeg[i],begs-wIncr)
        ends = np.minimum(begs+wLength-1,segEnd[i])
        eends = np.minimum(ends+wIncr,segEnd[i])
        segmentTable=np.vstack((segmentTable,np.vstack((bbegs, begs, ends, eends)).T))
    return segmentTable

def unravelMask(mask):
    changePoints = np.diff(1*mask)
    segBeg=np.where(changePoints==1)[0]+1
    segEnd=np.where(changePoints==-1)[0]    
    if mask[0]==1:
        segBeg = np.insert(segBeg,0,0)
    if mask[-1]==1:
        segEnd = np.append(segEnd,np.size(mask)-1)
    nSegs = np.size(segBeg)  
    return changePoints,segBeg,segEnd,nSegs  
    
def trainKBM(data, windowLength, windowRate, kbmSize):  
    from scipy.stats import multivariate_normal
    from scipy.spatial.distance import cdist           
    # Calculate number of gaussian components in the whole gaussian pool
    numberOfComponents = int(np.floor((np.size(data,0)-windowLength)/windowRate))
    # Add new array for storing the mvn objects
    gmPool = []
    likelihoodVector = np.zeros((numberOfComponents, 1))
    muVector = np.zeros((numberOfComponents,np.size(data,1)))
    sigmaVector = np.zeros((numberOfComponents,np.size(data,1)))    
    for i in range(numberOfComponents):
        mu = np.mean(data[np.arange((i*windowRate),(i*windowRate+windowLength),1,int)],axis=0)
        std = np.std(data[np.arange((i*windowRate),(i*windowRate+windowLength),1,int)],axis=0)
        muVector[i], sigmaVector[i] = mu, std
        mvn = multivariate_normal(mu,std)
        gmPool.append(mvn)        
        likelihoodVector[i] = -np.sum(mvn.logpdf(data[np.arange((i*windowRate),(i*windowRate+windowLength),1,int)]))        
    # Define the global dissimilarity vector
    v_dist = np.inf*np.ones((numberOfComponents,1))
    # Create the kbm itself, which is a vector of kbmSize size, and contains the gaussian IDs of the components
    kbm = np.zeros((kbmSize,1));    
    # As the stored likelihoods are negative, get the minimum likelihood
    bestGaussianID = np.where(likelihoodVector==np.min(likelihoodVector))[0]    
    currentGaussianID = bestGaussianID
    kbm[0]=currentGaussianID
    v_dist[currentGaussianID]=-np.inf
    # Compare the current gaussian with the remaining ones
    dpairsAll = cdist(muVector,muVector,metric='cosine')
    np.fill_diagonal(dpairsAll,-np.inf)
    for j in range(1,kbmSize):
        dpairs = dpairsAll[currentGaussianID]    
        v_dist=np.minimum(v_dist,dpairs.T)
        # Once all distances are computed, get the position with highest value
        # set this position to 1 in the binary KBM ponemos a 1 en el vector kbm
        # store the gaussian ID in the KBM
        currentGaussianID = np.where(v_dist==np.max(v_dist))[0]
        kbm[j]=currentGaussianID
        v_dist[currentGaussianID]=-np.inf  
    return [kbm, gmPool]
    
def getVgMatrix(data, gmPool, kbm, topGaussiansPerFrame):
    print('Calculating log-likelihood table... ',end='')
    logLikelihoodTable = getLikelihoodTable(data,gmPool,kbm)
    print('done')    
    print('Calculating Vg matrix... ',end='')
    Vg = np.argsort(-logLikelihoodTable)[:,0:topGaussiansPerFrame]
    print('done')
    return Vg

def getLikelihoodTable(data,gmPool,kbm):
    # GETLIKELIHOODTABLE computes the log-likelihood of each feature in DATA
    # against all the Gaussians of GMPOOL specified by KBM vector    
    # Inputs:
    #   DATA = matrix of feature vectors
    #   GMPOOL = pool of Gaussians of the kbm model
    #   KBM = vector of the IDs of the actual Gaussians of the KBM
    # Output:
    #   LOGLIKELIHOODTABLE = NxM matrix storing the log-likelihood of each of
    #   the N features given each of th M Gaussians in the KBM    
    kbmSize = np.size(kbm,0)
    logLikelihoodTable = np.zeros([np.size(data,0),kbmSize])    
    for i in range(kbmSize):
        logLikelihoodTable[:,i]=gmPool[int(kbm[i])].logpdf(data)
    return logLikelihoodTable
  
def getSegmentBKs(segmentTable, kbmSize, Vg, bitsPerSegmentFactor, speechMapping):
    # GETSEGMENTBKS converts each of the segments in SEGMENTTABLE into a binary key
    # and/or cumulative vector.

    # Inputs:
    #   SEGMENTTABLE = matrix containing temporal segments returned by 'getSegmentTable' function
    #   KBMSIZE = number of components in the kbm model
    #   VG = matrix of the top components per frame returned by 'getVgMatrix' function
    #   BITSPERSEGMENTFACTOR = proportion of bits that will be set to 1 in the binary keys
    # Output:
    #   SEGMENTBKTABLE = NxKBMSIZE matrix containing N binary keys for each N segments in SEGMENTTABLE
    #   SEGMENTCVTABLE = NxKBMSIZE matrix containing N cumulative vectors for each N segments in SEGMENTTABLE  
    
    numberOfSegments = np.size(segmentTable,0)
    segmentBKTable = np.zeros([numberOfSegments,kbmSize])
    segmentCVTable = np.zeros([numberOfSegments,kbmSize])    
    for i in range(numberOfSegments):
        # Conform the segment according to the segmentTable matrix       
        beginningIndex = int(segmentTable[i,0])
        endIndex = int(segmentTable[i,3])
        # Store indices of features of the segment
        # speechMapping is substracted one because 1-indexing is used for this variable
        A = np.arange(speechMapping[beginningIndex]-1,speechMapping[endIndex],dtype=int)
        segmentBKTable[i], segmentCVTable[i] = binarizeFeatures(kbmSize, Vg[A,:], bitsPerSegmentFactor)
    print('done')
    return segmentBKTable, segmentCVTable

def binarizeFeatures(binaryKeySize, topComponentIndicesMatrix, bitsPerSegmentFactor):
    # BINARIZEMATRIX Extracts a binary key and a cumulative vector from the the
    # rows of VG specified by vector A
    
    # Inputs:
    #   BINARYKEYSIZE = binary key size
    #   TOPCOMPONENTINDICESMATRIX = matrix of top Gaussians per frame
    #   BITSPERSEGMENTFACTOR = Proportion of positions of the binary key which will be set to 1
    # Output:
    #   BINARYKEY = 1xBINARYKEYSIZE binary key
    #   V_F = 1xBINARYKEYSIZE cumulative vector
    numberOfElementsBinaryKey = np.floor(binaryKeySize * bitsPerSegmentFactor)    
    # Declare binaryKey
    binaryKey = np.zeros([1, binaryKeySize])
    # Declare cumulative vector v_f
    v_f = np.zeros([1, binaryKeySize])    
    unique, counts = np.unique(topComponentIndicesMatrix, return_counts=True)    
    # Fill CV
    v_f[:,unique]=counts    
    # Fill BK
    binaryKey[0,np.argsort(-v_f)[0][0:int(numberOfElementsBinaryKey)]]=1    
    # CV normalization
    v_f = v_f/np.sum(v_f)    
    return binaryKey, v_f

def performClusteringLinkage(segmentBKTable, segmentCVTable, N_init, linkageCriterion,linkageMetric ):
    from scipy.cluster.hierarchy import linkage
    from scipy import cluster
    if linkageMetric == 'jaccard':
      observations = segmentBKTable
    elif linkageMetric == 'cosine':
      observations = segmentCVTable
    else:
      observations = segmentCVTable      
    clusteringTable = np.zeros([np.size(segmentCVTable,0),N_init]) 
    Z = linkage(observations,method=linkageCriterion,metric=linkageMetric)
    for i in np.arange(N_init):
      clusteringTable[:,i] = cluster.hierarchy.cut_tree(Z,N_init-i).T+1  
    k=N_init
    print('done')
    return clusteringTable, k
   
def performClustering( speechMapping, segmentTable, segmentBKTable, segmentCVTable, Vg, bitsPerSegmentFactor, kbmSize, N_init, initialClustering, clusteringMetric):
    numberOfSegments = np.size(segmentTable,0)
    clusteringTable = np.zeros([numberOfSegments, N_init])
    finalClusteringTable = np.zeros([numberOfSegments, N_init])
    activeClusters = np.ones([N_init,1]) 
    clustersBKTable = np.zeros([N_init, kbmSize])
    clustersCVTable = np.zeros([N_init, kbmSize])    
    clustersBKTable, clustersCVTable = calcClusters(clustersCVTable,clustersBKTable,activeClusters,initialClustering,N_init,segmentTable,kbmSize,speechMapping,Vg,bitsPerSegmentFactor)   
    ####### Here the clustering algorithm begins. Steps are:
    ####### 1. Reassign all data among all existing signatures and retrain them
    ####### using the new clustering
    ####### 2. Save the resulting clustering solution
    ####### 3. Compare all signatures with each other and merge those two with
    ####### highest similarity, creating a new signature for the resulting
    ####### cluster
    ####### 4. Back to 1 if #clusters > 1      
    for k in range(N_init):
    ####### 1. Data reassignment. Calculate the similarity between the current segment with all clusters and assign it to the one which maximizes
    ####### the similarity. Finally re-calculate binaryKeys for all cluster   
    # before doing anything, check if there are remaining clusters
    # if there is only one active cluster, break    
        if np.sum(activeClusters)==1:
            break          
        clustersStillActive=np.zeros([1,N_init])
        segmentToClustersSimilarityMatrix = binaryKeySimilarity_cdist(clusteringMetric,segmentBKTable,segmentCVTable,clustersBKTable,clustersCVTable)
        # clusteringTable[:,k] = finalClusteringTable[:,k] = np.argmax(segmentToClustersSimilarityMatrix,axis=1)+1
        clusteringTable[:,k] = finalClusteringTable[:,k] = np.nanargmax(segmentToClustersSimilarityMatrix,axis=1)+1
        # clustersStillActive[:,np.unique(clusteringTable[:,k]).astype(int)-1] = 1
        clustersStillActive[:,np.unique(clusteringTable[:,k]).astype(int)-1] = 1       
        ####### update all binaryKeys for all new clusters        
        activeClusters = clustersStillActive
        clustersBKTable, clustersCVTable = calcClusters(clustersCVTable,clustersBKTable,activeClusters.T,clusteringTable[:,k].astype(int),N_init,segmentTable,kbmSize,speechMapping,Vg,bitsPerSegmentFactor)                
        ####### 2. Compare all signatures with each other and merge those two with highest similarity, creating a new signature for the resulting        
        clusterSimilarityMatrix = binaryKeySimilarity_cdist(clusteringMetric,clustersBKTable,clustersCVTable,clustersBKTable,clustersCVTable)        
        np.fill_diagonal(clusterSimilarityMatrix,np.nan)        
        value = np.nanmax(clusterSimilarityMatrix)
        location = np.nanargmax(clusterSimilarityMatrix)        
        R,C = np.unravel_index(location,(N_init,N_init))        
        ### Then we merge clusters R and C
        #print('Merging clusters',R+1,'and',C+1,'with a similarity score of',np.around(value,decimals=4))
        print('Merging clusters','%3s'%str(R+1),'and','%3s'%str(C+1),'with a similarity score of',np.around(value,decimals=4))
        activeClusters[0,C]=0        
        ### 3. Save the resulting clustering and go back to 1 if the number of clusters >1
        mergingClusteringIndices = np.where(clusteringTable[:,k]==C+1)
        # update clustering table
        clusteringTable[mergingClusteringIndices[0],k]=R+1
        # remove binarykey for removed cluster
        clustersBKTable[C,:]=np.zeros([1,kbmSize])
        clustersCVTable[C,:]=np.nan
        # prepare the vector with the indices of the features of thew new cluster and then binarize
        segmentsToBinarize = np.where(clusteringTable[:,k]==R+1)[0]
        M=[]
        for l in np.arange(np.size(segmentsToBinarize,0)):
            M = np.append(M,np.arange(int(segmentTable[segmentsToBinarize][:][l,1]),int(segmentTable[segmentsToBinarize][:][l,2])+1))
        clustersBKTable[R,:], clustersCVTable[R,:]=binarizeFeatures(kbmSize,Vg[np.array(speechMapping[np.array(M,dtype='int')],dtype='int')-1].T,bitsPerSegmentFactor)
    print('done')
    return clusteringTable, k
  
def calcClusters(clustersCVTable,clustersBKTable,activeClusters,clusteringTable,N_init,segmentTable,kbmSize,speechMapping,Vg,bitsPerSegmentFactor):    
    for i in np.arange(N_init):
        if activeClusters[i]==1:
            segmentsToBinarize = np.where(clusteringTable==i+1)[0]
            M = []
            for l in np.arange(np.size(segmentsToBinarize,0)):
                M = np.append(M,np.arange(int(segmentTable[segmentsToBinarize][:][l,1]),int(segmentTable[segmentsToBinarize][:][l,2])+1))
            clustersBKTable[i], clustersCVTable[i] = binarizeFeatures(kbmSize,Vg[np.array(speechMapping[np.array(M,dtype='int')],dtype='int')-1].T,bitsPerSegmentFactor)
        else:
            clustersBKTable[i]=np.zeros([1,kbmSize])
            clustersCVTable[i]=np.nan
    return clustersBKTable, clustersCVTable
  
def binaryKeySimilarity_cdist(clusteringMetric,bkT1, cvT1, bkT2, cvT2):
    from scipy.spatial.distance import cdist      
    if clusteringMetric == 'cosine':
      S = 1 - cdist(cvT1,cvT2,metric=clusteringMetric)
    elif clusteringMetric == 'jaccard':
      S = 1 - cdist(bkT1,bkT2,metric=clusteringMetric)
    else:
      print('Clustering metric must be cosine or jaccard')  
    return S
  
def getBestClustering(bestClusteringMetric, bkT, cvT, clusteringTable, n ):
    from scipy.spatial.distance import cdist  
    wss = np.zeros([1,n])
    overallMean = np.mean(cvT,0)    
    if bestClusteringMetric == 'cosine':
        distances = cdist(np.expand_dims(overallMean,axis=0),cvT,bestClusteringMetric)
    elif bestClusteringMetric == 'jaccard':
        nBitsTol = np.sum(bkT[0,:])
        indices = np.argsort(-overallMean)
        overallMean = np.zeros([1,np.size(bkT,1)])
        overallMean[0,indices[np.arange(nBitsTol).astype(int)]]=1
        distances = cdist(overallMean,bkT,bestClusteringMetric)    
    distances2 = np.square(distances)    
    wss[0,n-1] = np.sum(distances2)    
    for i in np.arange(n-1):
        T = clusteringTable[:,i]
        clusterIDs = np.unique(T)
        variances = np.zeros([np.size(clusterIDs,0),1])
        for j in np.arange(np.size(clusterIDs,0)):
            clusterIDsIndex = np.where(T==clusterIDs[j])
            meanVector = np.mean(cvT[clusterIDsIndex,:],axis=1)
            if bestClusteringMetric=='cosine':
                distances = cdist(meanVector,cvT[clusterIDsIndex,:][0],bestClusteringMetric)
            elif bestClusteringMetric=='jaccard':
                indices = np.argsort(-meanVector[0,:])
                meanVector = np.zeros([1,np.size(bkT,1)])
                meanVector[0,indices[np.arange(nBitsTol).astype(int)]]=1
                distances = cdist(meanVector,bkT[clusterIDsIndex,:][0],bestClusteringMetric)
            distances2 = np.square(distances)    
            variances[j] = np.sum(distances2)
        wss[0,i]=np.sum(variances)        
    nPoints = np.size(wss,1)    
    allCoord = np.vstack((np.arange(1,nPoints+1),wss)).T
    firstPoint = allCoord[0,:]
    allCoord = allCoord[np.arange(np.where(allCoord[:,1]==np.min(allCoord[:,1]))[0],nPoints),:]
    nPoints = np.size(allCoord,0)
    lineVec = allCoord[-1,:] - firstPoint
    lineVecN = lineVec / np.sqrt(np.sum(np.square(lineVec)))
    vecFromFirst = np.subtract(allCoord,firstPoint)
    scalarProduct = vecFromFirst*lineVecN
    scalarProduct = scalarProduct[:,0]+scalarProduct[:,1]
    vecFromFirstParallel = np.expand_dims(scalarProduct,axis=1) * np.expand_dims(lineVecN,0)
    vecToLine = vecFromFirst - vecFromFirstParallel
    distToLine = np.sqrt(np.sum(np.square(vecToLine),axis=1))
    bestClusteringID = allCoord[np.argmax(distToLine)][0]
    return bestClusteringID
  
def getSpectralClustering(bestClusteringMetric,clusteringTable,N_init, bkT, cvT, n, sigma, percentile,maxNrSpeakers):
    from scipy.ndimage.filters import gaussian_filter
    simMatrix = binaryKeySimilarity_cdist(bestClusteringMetric,bkT,cvT,bkT,cvT)
    np.fill_diagonal(simMatrix,np.nan)       
    np.fill_diagonal(simMatrix,np.nanmax(simMatrix,1))    
    # Similarity matrix smoothed through Gaussian filter
    simMatrix_1 = gaussian_filter(simMatrix,sigma)
    # Similarity matrix thresholded to the percentile-th components leaving a small amount instead of 0 following Google's implementation
    thresholds = np.tile(percentile*np.nanmax(simMatrix,1)/100,(np.size(bkT,0),1)).T
    simMatrix_2 = np.copy(simMatrix_1)
    mask = simMatrix_2<thresholds
    simMatrix_2[np.where(mask==True)] = simMatrix_2[np.where(mask==True)]*0.01
    # Similarity matrix made symmetric
    simMatrix_3 = np.copy(simMatrix_2)
    for k in np.arange(np.size(simMatrix_3,0)):
        # for j in np.arange(np.size(simMatrix_3,)):
        maximum = np.maximum(simMatrix_3[:,k],simMatrix_3[k,:])
        simMatrix_3[:,k] = simMatrix_3[k,:] = maximum
    # Similarity matrix diffussion
    simMatrix_4 = np.dot(simMatrix_3,simMatrix_3)   
    # Row-wise max normalization
    simMatrix_5 = simMatrix_4 / np.tile(simMatrix_4.max(axis=0),(np.size(simMatrix_4,0),1)).T
    # Decomposition in eigenvalues, we don't use eigenvectors for the moment
    eigenvalues,eigenvectors = np.linalg.eigh(simMatrix_5)
    new_N_init = np.minimum(maxNrSpeakers,N_init)
    eigenvalues = np.flip(np.sort(eigenvalues),axis=0)[0:new_N_init]
    if new_N_init > 1:
        eigengaps = eigenvalues[0:new_N_init-1]/eigenvalues[1:new_N_init]
        eigengapssubstract = eigenvalues[0:new_N_init-1] - eigenvalues[1:new_N_init] 
    else:
        eigengaps = eigengapssubstract = eigenvalues       
    if eigengapssubstract[0] > 340:
        kclusters = 1
    else:
        eigengaps[0]=0  
        kclusters = np.flip(np.argsort(eigengaps),axis=0)[0]+1  
    
    nrElements = np.zeros([N_init,1])
    for k in np.arange(N_init):
        nrElements[k]=np.size(np.unique(clusteringTable[:,k]),0)
    distances = np.abs(nrElements-kclusters)    
    minidx = np.argmin(distances)
    bestClusteringID = minidx  
    return bestClusteringID

def smooth(a,WSZ):
    # a: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    #From https://stackoverflow.com/a/40443565
    out0 = np.convolve(a,np.ones(WSZ,dtype=int),'valid')/WSZ    
    r = np.arange(1,WSZ-1,2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((start,out0,stop))
  
def performResegmentation(data, speechMapping,mask,finalClusteringTable,segmentTable,modelSize,nbIter,smoothWin,numberOfSpeechFeatures):        
    from sklearn import mixture
    np.random.seed(0)
    
    changePoints,segBeg,segEnd,nSegs = unravelMask(mask) 
    speakerIDs = np.unique(finalClusteringTable)
    trainingData = np.empty([2,0])      
    for i in np.arange(np.size(speakerIDs,0)):
        spkID = speakerIDs[i]
        speakerFeaturesIndxs = []
        idxs = np.where(finalClusteringTable==spkID)[0]
        for l in np.arange(np.size(idxs,0)):
            speakerFeaturesIndxs = np.append(speakerFeaturesIndxs,np.arange(int(segmentTable[idxs][:][l,1]),int(segmentTable[idxs][:][l,2])+1))
        formattedData = np.vstack((np.tile(spkID,(1,np.size(speakerFeaturesIndxs,0))),speakerFeaturesIndxs))
        trainingData = np.hstack((trainingData,formattedData))
    
    llkMatrix = np.zeros([np.size(speakerIDs,0),numberOfSpeechFeatures])
    for i in np.arange(np.size(speakerIDs,0)):
        spkIdxs = np.where(trainingData[0,:]==speakerIDs[i])[0]
        spkIdxs = speechMapping[trainingData[1,spkIdxs].astype(int)].astype(int)-1
        msize = np.minimum(modelSize,np.size(spkIdxs,0))
        w_init = np.ones([msize])/msize
        m_init = data[spkIdxs[np.random.randint(np.size(spkIdxs,0), size=(1, msize))[0]],:]
        gmm=mixture.GaussianMixture(n_components=msize,covariance_type='diag',weights_init=w_init,means_init=m_init,verbose=0)
        gmm.fit(data[spkIdxs,:])
        llkSpk = gmm.score_samples(data)
        llkSpkSmoothed = np.zeros([1,numberOfSpeechFeatures])      
        for jx in np.arange(nSegs):
            sectionIdx = np.arange(speechMapping[segBeg[jx]]-1,speechMapping[segEnd[jx]]).astype(int)
            sectionWin = np.minimum(smoothWin,np.size(sectionIdx))
            if sectionWin % 2 ==0:
                sectionWin = sectionWin - 1
            if sectionWin>=2:
                llkSpkSmoothed[0,sectionIdx] = smooth(llkSpk[sectionIdx], sectionWin)
            else:
                llkSpkSmoothed[0,sectionIdx]=llkSpk[sectionIdx]
        llkMatrix[i,:] = llkSpkSmoothed[0].T
    segOut = np.argmax(llkMatrix,axis=0)+1
    segChangePoints = np.diff(segOut)
    changes = np.where(segChangePoints!=0)[0]
    relSegEnds = speechMapping[segEnd]
    relSegEnds = relSegEnds[0:-1]
    changes = np.sort(np.unique(np.hstack((changes,relSegEnds))))    
    
    # Create the new segment and clustering tables
    currentPoint = 0
    finalSegmentTable = np.empty([0,4])
    finalClusteringTableResegmentation = np.empty([0,1])
    
    for i in np.arange(np.size(changes,0)):
        addedRow = np.hstack((np.tile(np.where(speechMapping==np.maximum(currentPoint,1))[0],(1,2)),  np.tile(np.where(speechMapping==np.maximum(1,changes[i].astype(int)))[0],(1,2))))
        finalSegmentTable = np.vstack((finalSegmentTable,addedRow[0]))    
        finalClusteringTableResegmentation = np.vstack((finalClusteringTableResegmentation,segOut[(changes[i]).astype(int)]))
        currentPoint = changes[i]+1
    addedRow = np.hstack((np.tile(np.where(speechMapping==currentPoint)[0],(1,2)),  np.tile(np.where(speechMapping==numberOfSpeechFeatures)[0],(1,2))))
    finalSegmentTable = np.vstack((finalSegmentTable,addedRow[0]))
    finalClusteringTableResegmentation = np.vstack((finalClusteringTableResegmentation,segOut[(changes[i]+1).astype(int)]))    
    return finalClusteringTableResegmentation,finalSegmentTable  

def getSegmentationFile(format, frameshift,finalSegmentTable, finalClusteringTable, showName, filename, outputPath, outputExt):
    numberOfSpeechFeatures = finalSegmentTable[-1,2].astype(int)+1
    solutionVector = np.zeros([1,numberOfSpeechFeatures])
    for i in np.arange(np.size(finalSegmentTable,0)):
        solutionVector[0,np.arange(finalSegmentTable[i,1],finalSegmentTable[i,2]+1).astype(int)]=finalClusteringTable[i]
    seg = np.empty([0,3]) 
    solutionDiff = np.diff(solutionVector)[0]
    first = 0
    for i in np.arange(0,np.size(solutionDiff,0)):
        if solutionDiff[i]:
            last = i+1
            seg1 = (first)*frameshift
            seg2 = (last-first)*frameshift
            seg3 = solutionVector[0,last-1]
            if seg3:
                seg = np.vstack((seg,[seg1,seg2,seg3]))
            first = i+1
    last = np.size(solutionVector,1)
    seg1 = (first-1)*frameshift
    seg2 = (last-first+1)*frameshift
    seg3 = solutionVector[0,last-1]
    seg = np.vstack((seg,[seg1,seg2,seg3]))    
    solution = []
    if format=='MDTM':
        for i in np.arange(np.size(seg,0)):
            solution.append(showName+' 1 '+str(np.around(seg[i,0],decimals=4))+' '+str(np.around(seg[i,1],decimals=4))+' speaker NA unknown speaker'+str(seg[i,2].astype(int)))+'\n'
    elif format=='RTTM':
        for i in np.arange(np.size(seg,0)):
            solution.append('SPEAKER '+showName+' 1 '+str(np.around(seg[i,0],decimals=4))+' '+str(np.around(seg[i,1],decimals=4))+' <NA> <NA> speaker'+str(seg[i,2].astype(int))+' <NA>\n')
    else:
        print('Output file format must be MDTM or RTTM.')
    solution[-1]=solution[-1][0:-1]        
        
    outf = open(outputPath+filename+outputExt,"a")    
    outf.writelines(solution)
    outf.write('\n')
    outf.close()
    
class HTKFile:
    
    #Code from: https://github.com/danijel3/PyHTK
    
    """ Class to load binary HTK file.

        Details on the format can be found online in HTK Book chapter 5.7.1.

        Not everything is implemented 100%, but most features should be supported.

        Not implemented:
            CRC checking - files can have CRC, but it won't be checked for correctness

            VQ - Vector features are not implemented.
    """

    data = None
    nSamples = 0
    nFeatures = 0
    sampPeriod = 0
    basicKind = None
    qualifiers = None

    def load(self, filename):
        import struct
        """ Loads HTK file.

            After loading the file you can check the following members:

                data (matrix) - data contained in the file

                nSamples (int) - number of frames in the file

                nFeatures (int) - number if features per frame

                sampPeriod (int) - sample period in 100ns units (e.g. fs=16 kHz -> 625)

                basicKind (string) - basic feature kind saved in the file

                qualifiers (string) - feature options present in the file

        """
        with open(filename, "rb") as f:

            header = f.read(12)
            self.nSamples, self.sampPeriod, sampSize, paramKind = struct.unpack(">iihh", header)
            basicParameter = paramKind & 0x3F

            if basicParameter is 0:
                self.basicKind = "WAVEFORM"
            elif basicParameter is 1:
                self.basicKind = "LPC"
            elif basicParameter is 2:
                self.basicKind = "LPREFC"
            elif basicParameter is 3:
                self.basicKind = "LPCEPSTRA"
            elif basicParameter is 4:
                self.basicKind = "LPDELCEP"
            elif basicParameter is 5:
                self.basicKind = "IREFC"
            elif basicParameter is 6:
                self.basicKind = "MFCC"
            elif basicParameter is 7:
                self.basicKind = "FBANK"
            elif basicParameter is 8:
                self.basicKind = "MELSPEC"
            elif basicParameter is 9:
                self.basicKind = "USER"
            elif basicParameter is 10:
                self.basicKind = "DISCRETE"
            elif basicParameter is 11:
                self.basicKind = "PLP"
            else:
                self.basicKind = "ERROR"

            self.qualifiers = []
            if (paramKind & 0o100) != 0:
                self.qualifiers.append("E")
            if (paramKind & 0o200) != 0:
                qualifiers.append("N")
            if (paramKind & 0o400) != 0:
                self.qualifiers.append("D")
            if (paramKind & 0o1000) != 0:
                self.qualifiers.append("A")
            if (paramKind & 0o2000) != 0:
                self.qualifiers.append("C")
            if (paramKind & 0o4000) != 0:
                self.qualifiers.append("Z")
            if (paramKind & 0o10000) != 0:
                self.qualifiers.append("K")
            if (paramKind & 0o20000) != 0:
                self.qualifiers.append("0")
            if (paramKind & 0o40000) != 0:
                self.qualifiers.append("V")
            if (paramKind & 0o100000) != 0:
                self.qualifiers.append("T")

            if "C" in self.qualifiers or "V" in self.qualifiers or self.basicKind is "IREFC" or self.basicKind is "WAVEFORM":
                self.nFeatures = sampSize // 2
            else:
                self.nFeatures = sampSize // 4

            if "C" in self.qualifiers:
                self.nSamples -= 4

            if "V" in self.qualifiers:
                raise NotImplementedError("VQ is not implemented")

            self.data = []
            if self.basicKind is "IREFC" or self.basicKind is "WAVEFORM":
                for x in range(self.nSamples):
                    s = f.read(sampSize)
                    frame = []
                    for v in range(self.nFeatures):
                        val = struct.unpack_from(">h", s, v * 2)[0] / 32767.0
                        frame.append(val)
                    self.data.append(frame)
            elif "C" in self.qualifiers:

                A = []
                s = f.read(self.nFeatures * 4)
                for x in range(self.nFeatures):
                    A.append(struct.unpack_from(">f", s, x * 4)[0])
                B = []
                s = f.read(self.nFeatures * 4)
                for x in range(self.nFeatures):
                    B.append(struct.unpack_from(">f", s, x * 4)[0])

                for x in range(self.nSamples):
                    s = f.read(sampSize)
                    frame = []
                    for v in range(self.nFeatures):
                        frame.append((struct.unpack_from(">h", s, v * 2)[0] + B[v]) / A[v])
                    self.data.append(frame)
            else:
                for x in range(self.nSamples):
                    s = f.read(sampSize)
                    frame = []
                    for v in range(self.nFeatures):
                        val = struct.unpack_from(">f", s, v * 4)
                        frame.append(val[0])
                    self.data.append(frame)

            if "K" in self.qualifiers:
                print("CRC checking not implememented...")