import numpy as np
import math
import os
from scipy.fftpack import fft
from scipy.io.wavfile import read


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
    
    major_template = t_pc[0]
    major_norm = major_template/np.sum(major_template)
    
    minor_template = t_pc[0]
    minor_norm = minor_template/np.sum(minor_template)
    
    if bTune == True:
        tfInHz = estimate_tuning_freq(x, blockSize, hopSize, fs)
    elif bTune == False:
        tfInHz = 440
    else: 
        raise NameError('Input Argument (bTune) should be either 0 or 1 ')
    
    xb = block_audio(x, blockSize, hopSize, fs)
    X = compute_spectrogram(xb, fs)
    pitchChroma = extract_pitch_chroma(X, fs, tfInHz)
    
    minimum_dist = np.ones(1) + 0.1
    minimum_ndx = np.zeros(1) 
    for i in np.arange(0, 12):
        
        pitchChroma_Shifted = np.concatenate((pitchChroma[i:len(pitchChroma)], pitchChroma[0:i]))
        majDistance = np.sqrt(np.sum((pitchChroma_Shifted - major_norm)**2))
        
        if majDistance < minimum_dist:
            minimum_dist = majDistance
            minimum_ndx = i
            
        minDistance = np.sqrt(np.sum((pitchChroma_Shifted - minor_norm)**2))
        
        if minDistance < minimum_dist:
            minimum_dist = majDistance
            minimum_ndx = i + 12
        
    keyEstimate = minimum_ndx
    
    return keyEstimate

#C. Evaluation

def eval_tfe(pathToAudio, pathToGT):
    audio_files = os.listdir(pathToAudio)
    gt_files = os.listdir(pathToGT)
    audio_files.sort()
    gt_files.sort()

    deviations = np.zeros(len(audio_files))

    for i, (audio_file, gt_file) in enumerate(zip(audio_files, gt_files)):
        sr, y = read(os.path.join(pathToAudio, audio_file))
        tfInHz_est = estimate_tuning_freq(y, 4096, 2048, sr)
        tfInHz_gt = np.loadtxt(os.path.join(pathToGT, gt_file))
        diff_in_cents = 1200 * np.log(tfInHz_est/tfInHz_gt)
        deviations[i] = diff_in_cents

    avgDeviation = np.mean(deviations)

    return avgDeviation

def eval_key_detection(pathToAudio, pathToGT):
    audio_files = os.listdir(pathToAudio)
    gt_files = os.listdir(pathToGT)
    audio_files.sort()
    gt_files.sort()

    results_with_est = np.zeros(len(audio_files))
    results_without_est = np.zeros(len(audio_files))

    for i, (audio_file, gt_file) in enumerate(zip(audio_files, gt_files)):
        sr, y = read(os.path.join(pathToAudio, audio_file))
        with_est = detect_key(y, 4096, 2048, sr, True)
        without_est = detect_key(y, 4096, 2048, sr, False)
        gt = np.loadtxt(os.path.join(pathToGT, gt_file))

        if with_est==gt:
            results_with_est[i] = 1
        else:
            results_with_est[i] = 0

        if without_est==gt:
            results_without_est[i] = 1
        else:
            results_without_est[i] = 0

    accuracy_with_est = np.sum(results_with_est) / len(results_with_est)
    accuracy_without_est = np.sum(results_without_est) / len(results_without_est)

    accuracy = np.array([[accuracy_with_est, accuracy_without_est]])

    return accuracy

def evaluate(pathToAudioKey, pathToGTKey,pathToAudioTf, pathToGTTf):
    avg_accuracy = eval_key_detection(pathToAudioKey, pathToGTKey)
    avg_deviationInCent = eval_tfe(pathToAudioTf, pathToGTTf)

    return avg_accuracy, avg_deviationInCent






