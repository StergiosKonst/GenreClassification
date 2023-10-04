import numpy as np
import pywt
import tensorflow as tf
from scipy.fftpack import dct
from scipy.signal import get_window
import librosa
from tensorflow.python.ops.gen_math_ops import sign

mel_wavelet_bands = {
    0: "aaaaaa",
    1: "aaaaad",
    2: "aaaada",
    3: "aaaadd",
    4: "aaadaa",
    5: "aaadad",
    6: "aaadda",
    7: "aaaddd",
    8: "aadaa",
    9: "aadad",
    10: "aadda",
    11: "aaddd",
    12: "adaaa",
    13: "adaad",
    14: "adada",
    15: "adadd",
    16: "addaa",
    17: "addad",
    18: "addd",
    19: "daaa",
    20: "daad",
    21: "dad",
    22: "dda",
    23: "ddd"
}

def frame_time_series(signal, frame_length, hop_length):
    '''
    A function to divide signal into overlaping frames.
      
    Parameters
    ----------
    time_series : nympy array
        Time series array to frame.
    fr_length : int 
        Frame length. 
    hop_length : int 
        Number of samples to hop in each overlaping frame.
    Output
    ------
    framed_time_series : np.array (no_frames, fr_length)
        The framed signal
    '''
    # Pad signal
    padded_series = np.pad(signal, hop_length, mode='reflect')
    # Frame Signal
    framed_series = tf.signal.frame(padded_series,
                                    frame_length,
                                    hop_length,
                                    pad_end=True)
    framed_series = framed_series.numpy()
    framed_series = framed_series.astype(np.float64)
    # Specify window function
    window = get_window("triang", frame_length, fftbins=False)
    # Multiply each frame with specified window
    for fr_idx in range(0, framed_series.shape[0]):
        framed_series[fr_idx, :] *= window
    return framed_series

def calculate_frame_length(level):
    '''
    Function to determine frame and hop length according to
        the decomposition level
        
        Parameters
        ----------
            level : int
                Decomposition level of the wavelet
                    transform
        Returns
        -------
            fr_length : int
                Frame length in the specified 
                    decomposition level
            hop_length : int
                Hop length in the specified 
                    decomposition level
    '''
    fr_length = int(512 // np.power(2, level))
    hop_length = int(fr_length / 2)
    return fr_length, hop_length

def wavelet_coefficients(signal):

    # Normalize signals
    signal = librosa.util.normalize(signal)
    signal = np.squeeze(signal)
    
    # Calculate VAD non - linearity
    temp = librosa.stft(signal, n_fft=512, hop_length=256)
    frames_num = temp.shape[1]
    
    # Calculate wavelet packet trasform for each signal (clean & noisy)
    wp = pywt.WaveletPacket(data=signal,
                                  wavelet="db6",
                                  mode='symmetric')
   

    # Initialize gb and WTTC arrays
    wtcc = np.zeros((frames_num, len(mel_wavelet_bands)), dtype="float32")

    # Iterate througn all filtering bands
    for i in range(0, len(mel_wavelet_bands)):
        # Get band time series data for clean and noisy speach
        data = wp[mel_wavelet_bands[i]].data

        # Calculate frame length according to wavelet band level
        band_level = wp[mel_wavelet_bands[i]].level
        frame_length, hop_length = calculate_frame_length(band_level)

        # Frame band time series data for clean and noisy
        framed_data = frame_time_series(data, frame_length, hop_length)
        
        # Square band data for clean and noisy
        framed_data = np.square(framed_data)
        
        # Calculate mean
        energy = np.mean(framed_data, axis=1)
        

        # Put Powered scalogram coeff in WTCC array
        wtcc[:, i] = energy[0:frames_num]

    # Convert power scalogram to db scalogram.
    wtcc = librosa.power_to_db(np.transpose(wtcc))
    # Compute dct transform along frequency axis.
    wtcc = dct(wtcc, axis=0, type=2, norm="ortho")

    return wtcc

def extract_WTCC(signal):
    wtcc = wavelet_coefficients(signal)
    return wtcc
