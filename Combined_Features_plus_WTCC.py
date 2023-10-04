import csv
import os
from librosa.feature.spectral import spectral_centroid, spectral_rolloff
import scipy.stats
import librosa
import numpy as np
import matplotlib.pyplot as plt
from Wavelet_Calculator import extract_WTCC

AUDIO_DIR = "<AUDIO_PATH>"
ANNOTATIONS_DIR = "<LABELS_PATH>"


def spectral_cent(signal, sr): 
    spec_cen = librosa.feature.spectral_centroid(
        signal,
        sr=sr,
        n_fft=8192,
        hop_length=4096
    )

    spec_cen = np.squeeze(spec_cen)  
    
    avg = np.average(spec_cen)

    spec_cen = librosa.feature.spectral_centroid(
        signal,
        sr=sr,
        n_fft=1024,
        hop_length=512
    )

    spec_cen = np.squeeze(spec_cen)  
    
    mean = np.mean(spec_cen)
    var = np.var(spec_cen)
    skew = np.average(scipy.stats.skew(spec_cen))
    kurt = np.average(scipy.stats.kurtosis(spec_cen))
    
    return avg, mean, var, skew, kurt
def root_mean_square(signal):
    root_ms = librosa.feature.rms(
        signal,
        frame_length=8192,
        hop_length=4096
    )
    root_ms = np.squeeze(root_ms)
    
    avg = np.average(root_ms)

    root_ms = librosa.feature.rms(
        signal,
        frame_length=1024,
        hop_length=512
    )
    root_ms = np.squeeze(root_ms)
    
    mean = np.mean(root_ms)
    var = np.var(root_ms)
    skew = np.average(scipy.stats.skew(root_ms))
    kurt = np.average(scipy.stats.kurtosis(root_ms))
    return avg, mean, var, skew, kurt
def zcr(signal):
    zero_cr = librosa.feature.zero_crossing_rate(
        signal,
        frame_length=8192,
        hop_length=4096
    )
    zero_cr = np.squeeze(zero_cr)
    
    avg = np.average(zero_cr)

    zero_cr = librosa.feature.zero_crossing_rate(
        signal,
        frame_length=1024,
        hop_length=512
    )
    zero_cr = np.squeeze(zero_cr)
    
    mean = np.mean(zero_cr)
    var = np.var(zero_cr)
    skew = np.average(scipy.stats.skew(zero_cr))
    kurt = np.average(scipy.stats.kurtosis(zero_cr))
    return avg, mean, var, skew, kurt
def spectral_roll(signal, sr):
    spec_roll = librosa.feature.spectral_rolloff(
        signal,
        sr=sr,
        n_fft=8192,
        hop_length=4096
    )
    spec_roll = np.squeeze(spec_roll)
    
    avg = np.average(spec_roll)

    spec_roll = librosa.feature.spectral_rolloff(
        signal,
        sr=sr,
        n_fft=1024,
        hop_length=512
    )
    spec_roll = np.squeeze(spec_roll)
    
    mean = np.mean(spec_roll)
    var = np.var(spec_roll)
    skew = np.average(scipy.stats.skew(spec_roll))
    kurt = np.average(scipy.stats.kurtosis(spec_roll))
    return avg, mean, var, skew, kurt
def spectral_flat(signal):
    spec_flat = librosa.feature.spectral_flatness(
        signal,
        n_fft=8192,
        hop_length=4096
    )
    spec_flat = np.squeeze(spec_flat)
    
    avg = np.average(spec_flat)

    spec_flat = librosa.feature.spectral_flatness(
        signal,
        n_fft=1024,
        hop_length=512
    )
    spec_flat = np.squeeze(spec_flat)
    
    mean = np.mean(spec_flat)
    var = np.var(spec_flat)
    skew = np.average(scipy.stats.skew(spec_flat))
    kurt = np.average(scipy.stats.kurtosis(spec_flat))
    return avg, mean, var, skew, kurt
def spectral_band(signal, sr):
    spec_band = librosa.feature.spectral_bandwidth(
        signal,
        sr=sr,
        n_fft=8192,
        hop_length=4096
    )
    spec_band = np.squeeze(spec_band)
    
    avg = np.average(spec_band)

    spec_band = librosa.feature.spectral_bandwidth(
        signal,
        sr=sr,
        n_fft=1024,
        hop_length=512
    )
    spec_band = np.squeeze(spec_band)
    
    mean = np.mean(spec_band)
    var = np.var(spec_band)
    skew = np.average(scipy.stats.skew(spec_band))
    kurt = np.average(scipy.stats.kurtosis(spec_band))
    return avg, mean, var, skew, kurt
def spectral_contr(signal, sr):
    spec_contr = librosa.feature.spectral_contrast(
        signal,
        sr=sr,
        n_fft=8192,
        hop_length=4096
    )
    spec_contr = np.squeeze(spec_contr)
    
    avg = np.average(spec_contr)

    spec_contr = librosa.feature.spectral_contrast(
        signal,
        sr=sr,
        n_fft=1024,
        hop_length=512
    )
    spec_contr = np.squeeze(spec_contr)
    
    mean = np.mean(spec_contr)
    var = np.var(spec_contr)
    skew = np.average(scipy.stats.skew(spec_contr))
    kurt = np.average(scipy.stats.kurtosis(spec_contr))
    return avg, mean, var, skew, kurt
def chroma(signal, sr):
    chrm = librosa.feature.chroma_stft(
        signal,
        sr=sr,
        n_fft=8192,
        hop_length=4096
    )
    chrm = np.squeeze(chrm)
    
    avg = np.average(chrm)

    chrm = librosa.feature.chroma_stft(
        signal,
        sr=sr,
        n_fft=1024,
        hop_length=512
    )
    chrm = np.squeeze(chrm)
    
    mean = np.mean(chrm)
    var = np.var(chrm)
    skew = np.average(scipy.stats.skew(chrm))
    kurt = np.average(scipy.stats.kurtosis(chrm))
    return avg, mean, var, skew, kurt
def MFCCs(signal, sr, index):
    Mfcc = librosa.feature.mfcc(
        signal,
        sr=sr,
        n_mfcc=13
    )
    temp = Mfcc[index]
    temp = np.squeeze(temp)
    
    avg = np.average(temp)
    mean = np.mean(temp)
    var = np.var(temp)
    skew = np.average(scipy.stats.skew(temp))
    kurt = np.average(scipy.stats.kurtosis(temp))
    return avg, mean, var, skew, kurt
def WTCCs(signal):
    Wtcc = extract_WTCC(signal)
    full_list =[]
    for i in range (24):
        wtcc_temp = Wtcc[i]
        wtcc_temp = np.squeeze(wtcc_temp)

        mean = np.mean(wtcc_temp)
        var = np.var(wtcc_temp)
        skew = np.average(scipy.stats.skew(wtcc_temp))
        kurt = np.average(scipy.stats.kurtosis(wtcc_temp))
        Wtcc_list = [mean, var, skew, kurt]
        full_list = full_list + Wtcc_list

    return full_list

header = ["filename", "label",
 "Cen", "Cen_M", "Cen_V", "Cen_S", "Cen_K",
 "Rms", "Rms_M", "Rms_V", "Rms_S", "Rms_K",
 "Zcr", "Zcr_M", "Zcr_V", "Zcr_S", "Zcr_K",
 "Rol", "Rol_M", "Rol_V", "Rol_S", "Rol_K",
 "Fla", "Fla_M", "Fla_V", "Fla_S", "Fla_K",
 "Ban", "Ban_M", "Ban_V", "Ban_S", "Ban_K",
 "Con", "Con_M", "Con_V", "Con_S", "Con_K",
 "Chr", "Chr_M", "Chr_V", "Chr_S", "Chr_K",
 "Wtcc1_M", "Wtcc1_V", "Wtcc1_S", "Wtcc1_K", 
 "Wtcc2_M", "Wtcc2_V", "Wtcc2_S", "Wtcc2_K",
 "Wtcc3_M", "Wtcc3_V", "Wtcc3_S", "Wtcc3_K",
 "Wtcc4_M", "Wtcc4_V", "Wtcc4_S", "Wtcc4_K",
 "Wtcc5_M", "Wtcc5_V", "Wtcc5_S", "Wtcc5_K",
 "Wtcc6_M", "Wtcc6_V", "Wtcc6_S", "Wtcc6_K",
 "Wtcc7_M", "Wtcc7_V", "Wtcc7_S", "Wtcc7_K",
 "Wtcc8_M", "Wtcc8_V", "Wtcc8_S", "Wtcc8_K",
 "Wtcc9_M", "Wtcc9_V", "Wtcc9_S", "Wtcc9_K",
 "Wtcc10_M", "Wtcc10_V", "Wtcc10_S", "Wtcc10_K",
 "Wtcc11_M", "Wtcc11_V", "Wtcc11_S", "Wtcc11_K",
 "Wtcc12_M", "Wtcc12_V", "Wtcc12_S", "Wtcc12_K",
 "Wtcc13_M", "Wtcc13_V", "Wtcc13_S", "Wtcc13_K",
 "Wtcc14_M", "Wtcc14_V", "Wtcc14_S", "Wtcc14_K",
 "Wtcc15_M", "Wtcc15_V", "Wtcc15_S", "Wtcc15_K",
 "Wtcc16_M", "Wtcc16_V", "Wtcc16_S", "Wtcc16_K",
 "Wtcc17_M", "Wtcc17_V", "Wtcc17_S", "Wtcc17_K",
 "Wtcc18_M", "Wtcc18_V", "Wtcc18_S", "Wtcc18_K",
 "Wtcc19_M", "Wtcc19_V", "Wtcc19_S", "Wtcc19_K",
 "Wtcc20_M", "Wtcc20_V", "Wtcc20_S", "Wtcc20_K",
 "Wtcc21_M", "Wtcc21_V", "Wtcc21_S", "Wtcc21_K",
 "Wtcc22_M", "Wtcc22_V", "Wtcc22_S", "Wtcc22_K",
 "Wtcc23_M", "Wtcc23_V", "Wtcc23_S", "Wtcc23_K",
 "Wtcc24_M", "Wtcc24_V", "Wtcc24_S", "Wtcc24_K",
 ]
data = []

for i ,(dirpath, dirnames, filenames) in enumerate(os.walk(AUDIO_DIR)):
    
    if dirpath is not AUDIO_DIR:
        dirpath_compontents = os.path.split(dirpath) 
        folder = dirpath_compontents[-1]
        temp = []
        for f in filenames:
            file_path = os.path.join(dirpath, f)
            signal, sr = librosa.load(file_path)
           
            labels = ([f, folder])
            
            x, y, z, t, e = spectral_cent(signal, sr)
            spec_cen = [x, y, z, t, e]        
            temp = labels + spec_cen

            x, y, z, t, e = root_mean_square(signal)
            root_ms = [x, y, z, t, e]        
            temp = temp + root_ms

            x, y, z, t, e = zcr(signal)
            zero_cr = [x, y, z, t, e]        
            temp = temp + zero_cr

            x, y, z, t, e = spectral_roll(signal, sr)
            spec_roll = [x, y, z, t, e]        
            temp = temp + spec_roll

            x, y, z, t, e = spectral_flat(signal)
            spec_flat = [x, y, z, t, e]        
            temp = temp + spec_flat

            x, y, z, t, e = spectral_band(signal, sr)
            spec_band = [x, y, z, t, e]        
            temp = temp + spec_band

            x, y, z, t, e = spectral_contr(signal, sr)
            spec_contr = [x, y, z, t, e]        
            temp = temp + spec_contr

            x, y, z, t, e = chroma(signal, sr)
            chrm = [x, y, z, t, e]        
            temp = temp + chrm

            Wtcc = WTCCs(signal)
            temp = temp + Wtcc
            a = 1
                    
            data.append(temp)      
            a = 1  
            
            
                  

with open(ANNOTATIONS_DIR, 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    
    writer.writerow(header)

    writer.writerows(data)


"""
plt.figure(figsize=(15,12))

    time = np.linspace(0, sr, len(spec_cen))
    num_time_bins = int(len(time))
    
    plt.plot(time[:num_time_bins], spec_cen[:num_time_bins])
    plt.xlabel("Time")
    plt.ylabel("Spectral Centroid")
    plt.show()
"""
