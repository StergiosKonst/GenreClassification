import os
from numpy.core.fromnumeric import resize

import torch
from torch.utils.data import Dataset, random_split
import pandas as pd
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import SubsetRandomSampler
import torchaudio
from torchaudio import transforms
from torchvision import transforms
from torchvision.transforms import Resize
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, dataloader
import matplotlib.pyplot as plt
import librosa
import librosa.display

class GTZANDataset(Dataset):

    def __init__(self, annotations_file, audio_dir, transformation, target_sample_rate, num_samples, device):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        label = self._transform_label_into_number(label)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = self._resample_if_necessary(signal ,sr)
        signal = signal.to(self.device)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)
        return signal, label

    def _get_audio_sample_path(self, index):
        fold = f"{self.annotations.iloc[index, 1]}"
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[index, 0])
        return path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 1]

    def _transform_label_into_number(self, label):
        if label == 'blues':
            label = 0
        elif label == 'classical':
            label = 1    
        elif label == 'country':
            label = 2
        elif label == 'disco':
            label = 3           
        elif label == 'hiphop':
            label = 4 
        elif label == 'jazz':
            label = 5
        elif label == 'metal':
            label = 6 
        elif label == 'pop':
            label = 7 
        elif label == 'reggae':
            label = 8 
        elif label == 'rock':
            label = 9 

        return label    
           
    def _transform_label_into_number_2(self, label):
        
        if label == 'classical':
            label = 0   
        elif label == 'jazz':
            label = 1
        elif label == 'rock':
            label = 2 

        return label    
    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:    
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal    

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        lenght_signal = signal.shape[1]
        if lenght_signal < self.num_samples:
            num_missing_samples = self.num_samples - lenght_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal


if __name__=="__main__":

    AUDIO_DIR = "C:/Users/Stergios/Desktop/Datasets/GTZAN/genres_modified"
    ANNOTATIONS_FILE = "C:/Users/Stergios/Desktop/Datasets/GTZAN/annotations_file_3.csv"
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 3*22050
    """
    if torch.cuda.is_available():
        device = "cuda"
    else:"""
    device = "cpu"    
    print(f"Using Device: {device}")    

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, 
        n_fft=2048,
        hop_length=512,
        n_mels=128
        )
    
    gtzan = GTZANDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE,  NUM_SAMPLES, device)
    
    train_length = int(0.75*len(gtzan))
    test_length = len(gtzan) - train_length

    train_dataset, test_dataset = random_split(gtzan, [train_length, test_length], generator=torch.Generator().manual_seed(42))
    
    train_set = DataLoader(train_dataset, batch_size=1)
    
    for inputs, targets in train_set:
        
        inputs = np.array(inputs)
        inputs = np.squeeze(inputs)
        plt.figure(figsize=(25,10))
        librosa.display.specshow(
        inputs,
        sr=22050,
        y_axis="mel",
        x_axis= "time"
        )
        plt.colorbar(format="%+2.f")
        plt.show()
        a=2