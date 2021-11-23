import numpy as np
import math
import scipy


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


def estimate_tuning_freq(x, blockSize, hopSize, fs):
    xb, t = block_audio(x, blockSize, hopSize, fs)
    X = scipy.fft(xb, blockSize=blockSize)
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



