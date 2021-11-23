import numpy as np
import math
import scipy
from scipy.fftpack import fft


#A.Tuning Frequency Estimation
def get_spectral_peaks(X):
    spec = X.T
    spectralPeaks = []
    for i, frame in enumerate(spec):
        spectralPeaks.append(frame.argsort()[frame.shape()-20, frame.shape()-1])
    spectralPeaks = np.array(spectralPeaks)
    return spectralPeaks


def block_audio(x, blockSize, hopSize, fs):
    numBlocks = math.ceil(x.size / hopSize)
    xb = np.zeros([numBlocks, blockSize])
    t = (np.arange(0, numBlocks) * hopSize) / fs
    x = np.concatenate((x, np.zeros(blockSize)), axis=0)
    for n in range(0, numBlocks):
        i_start = n * hopSize
        i_stop = np.min([x.size - 1, i_start + blockSize - 1])
        xb[n][np.arange(0, blockSize)] = x[np.arange(i_start, i_stop + 1)] * np.hamming(blockSize)
    return xb, t


def compute_spectrogram(xb, fs):
    numBlocks = xb.shape[0]
    afWindow = 0.5 - (0.5 * np.cos(2 * np.pi / xb.shape[1] * np.arange(xb.shape[1])))
    X = np.zeros([math.ceil(xb.shape[1]/2+1), numBlocks])
    
    for n in range(0, numBlocks):
        # apply window
        tmp = abs(fft(xb[n,:] * afWindow))*2/xb.shape[1]
    
        # compute magnitude spectrum
        X[:,n] = tmp[range(math.ceil(tmp.size/2+1))] 
        # normalize
        X[[0,math.ceil(tmp.size/2)],n]= X[[0,math.ceil(tmp.size/2)],n]/np.sqrt(2) 

    return X


def estimate_tuning_freq(x, blockSize, hopSize, fs):
    xb, t = block_audio(x, blockSize, hopSize, fs)
    X = compute_spectrogram(xb, fs)
    spectralPeaks = get_spectral_peaks(X)
    tfInHz = []
    for i, value in enumerate(spectralPeaks):
        valueInHz = fs*value/blockSize
        valueInCent = 1200 * math.log2(valueInHz/440)
        if np.max(np.absolute(valueInCent%100)) >= np.max(100-np.absolute(valueInCent%100)):
            tfInHz.append(np.max(valueInCent%100))
        else:
            tfInHz.append(np.absolute(valueInCent%100)-100)
    tfInHz = max(tfInHz)
    tfInHz = 440 * math.pow(2, tfInHz/1200)
    return tfInHz


#B.Key Detection
def freq2class(freq):
    if freq != 0:
        key = round(12 * math.log(freq / 440, 2)) + 69
    else:
        key = 0
    return key % 12


def extract_pitch_chroma(X, fs, tfInHz):
    pitchChroma = np.zeros((12, X.shape[1]))
    for i, frame in enumerate(X):
        X[:, i] = X[:, i] - tfInHz
        for freq in np.arange(start=int(X.shape[1] * 130.81 / fs), stop=int(X.shape[1] * 987.77 / fs)):
            if X[freq][i] > 0:
                pitchChroma[freq2class(fs * (freq / X.shape[1]))][i] = \
                pitchChroma[freq2class(fs * (freq / X.shape[1]))][i] + X[freq][i]
        if np.max(pitchChroma[:, i]) == 0:
            continue
        pitchChroma[:, i] = pitchChroma[:, i] / np.linalg.norm(pitchChroma[:, i])
    return pitchChroma


def detect_key(x, blockSize, hopSize, fs, bTune):
    
    t_pc = np.array([[6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]])
    
    if bTune == 1:
        tfInHz = estimate_tuning_freq(x, blockSize, hopSize, fs)
    elif bTune == 0:
        tfInHz = 440
    else: 
        raise NameError('Input Argument (bTune) should be either 0 or 1 ')
    
    xb = block_audio(x, blockSize, hopSize, fs)
    X = compute_spectrogram(xb, fs)
    pitchChroma = extract_pitch_chroma(X, fs, tfInHz)
    
    
    for i in arrange(0, 1):
        
    return keyEstimate

