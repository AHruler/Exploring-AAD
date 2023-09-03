import librosa
import speechpy
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LogNorm
from scipy.signal import spectrogram
from IPython.display import Audio
from IPython.display import display
import torch
import torchaudio

import numpy as np

##------ LOAD & PLAY -------##
def load_audio(filename, normalize=True, channel=0):
    waveform, sample_rate = torchaudio.load(filename, normalize=normalize)
    num_channels, num_frames = waveform.shape
    if num_channels != 1:
        x = librosa.to_mono(waveform.numpy())
        waveform = torch.tensor(x).float()
        waveform = waveform.unsqueeze(0)
    return waveform, sample_rate

def play_audio(waveform, sample_rate):
    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape
    if num_channels == 1:
        display(Audio(waveform[0], rate=sample_rate))
    elif num_channels == 2:
        display(Audio((waveform[0], waveform[1]), rate=sample_rate))
    else:
        raise ValueError("Waveform with more than 2 channels are not supported.")
        
##---------Features ext. --------##
def get_features(file, n_fft, sr):
    n_fft = n_fft
    x, sr = load_audio(file)
    
    lmfe, lml = get_lmfe(x, n_fft, sr)

    mfcc, mfl = extract_mfcc(x, n_fft, sr)

    chroma, cl = get_chroma(x, sr)

    feat = np.concatenate(([lmfe], [mfcc], [chroma]), axis=1).flatten()
    labels = np.concatenate(([lml], [mfl], [cl]), axis=1)
    return feat.flatten(), labels

def get_lmfe(y, n_fft, sr):
    n_fft=n_fft
    y = y[0].numpy()
    n=54

    # Compute log mel energy
    lmfe = speechpy.feature.lmfe(y, sr, fft_length=n_fft, frame_length=1, frame_stride=(0.75), num_filters=n)
    labels = []
    for i in range(1, n+1):
        for j in range(lmfe.shape[0]):
            labels.append('lmfe{}_{}'.format(i+1, j+1))
    return lmfe.flatten(), labels


def get_mfcc(x, n_fft, sr):
    ''' MFCC - effective in capturing spectral features that are relevant to human perception
    '''
    n_fft=sr

    win_length = sr // 4 
    hop_length = int(win_length*0.75)

    n_mfcc = 13
    mfcc_transform = torchaudio.transforms.MFCC(sample_rate=sr,n_mfcc=n_mfcc, log_mels=True,
                                                melkwargs={
                                                    "n_fft": n_fft,
                                                    "win_length": win_length,
                                                    "hop_length": hop_length,
                                                    "n_mels": 128,
                                                    "mel_scale": "htk",
                                                },)

    mfcc = mfcc_transform(x)
    mfcc = mfcc[:, 1:, :]  # Drop the first MFCC coefficient
    
    labels = []
    for i in range(1, n_mfcc):
        for j in range(mfcc.shape[2]):
            labels.append('mfcc{}_{}'.format(i+1, j+1))

    return np.squeeze(mfcc.numpy()).flatten(), labels

                     
def get_chroma(x, sr):
    win_length = sr // 4 # 1 sec window
    hop_length = int(win_length*0.75) # 25sec overlap
    C = np.abs(librosa.cqt(y=x.numpy(), sr=sr, pad_mode='reflect', hop_length=hop_length)) # magnitude of constant C
    chroma = librosa.feature.chroma_cens(C=C, sr=sr, win_len_smooth=2, hop_length=hop_length)

    
    labels = []
    for i in range(chroma.shape[1]):
        for j in range(chroma.shape[2]):
            labels.append('chroma{}_{}'.format(i+1, j+1))
    
    return np.squeeze(chroma).flatten(), labels




def get_tempo(x, sr):
    onset_env = librosa.onset.onset_strength(y=x.numpy(), sr=sr)
    tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
    return int(tempo.flatten()[0])


## --------- Ploting feat. ------------ ##
def plot_chroma(waveforms, titles, ylabel="freq_bin", sr = 16000, typ='fan', sig_pitch=[]):
    fig, axes = plt.subplots(nrows=2, ncols=int(np.ceil(len(titles) / 2)), figsize=(8.7, 6.6))
    path=f'chroma_{typ}_sample'
    win_length = sr // 4 # 1 sec window
    hop_length = int(win_length*0.75) # 25sec overlap
    # Create a common color bar for all subplots
    highlight_row3 = sig_pitch
    p_classes = ['C', 'C♯', 'D', 'D♯', 'E', 'F', 'F♯', 'G', 'G♯', 'A', 'A♯', 'B']

    for i, x in enumerate(waveforms):
        C = np.abs(librosa.cqt(y=x.numpy(), sr=sr, pad_mode='reflect', hop_length=hop_length)) # magnitude of constant C
        chroma = np.squeeze(librosa.feature.chroma_cens(C=C, sr=sr, win_len_smooth=2, hop_length=hop_length))
        
        row = i // int(np.ceil(len(titles) / 2))
        col = i % int(np.ceil(len(titles) / 2))
        
        ax = axes[row, col]
        ax.set_title(titles[i])
        if row == 0:  # For row 1
            ax.set_xlabel("Frame")
            ax.set_xticks([])  # Hide x-axis ticks for row 1
            ax.set_yticks(np.arange(12))
            
            
        elif row == 1:  # For row 3
            ax.set_xlabel("Frame")
        else:
            
            ax.set_xticks([])
            
        # ax.set_title(titles[i])
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Frame")
        ax.set_xticks([])
        ax.set_yticks([])
        
        if i < 3:
            ax.label_outer()
            ax.set_yticks([])
            
        ax.set_yticks([])
        ax.label_outer()
        im = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax)
    
    # # Remove empty subplots, if any
    # if len(titles) < axes.size:
    #     for i in range(len(titles), axes.size):
    #         fig.delaxes(axes.flatten()[i])
    plt.colorbar(im, ax=axes)
    if highlight_row3:

        for pitch_class in highlight_row3:
            axes[1,0].set_yticks(np.arange(12))
            axes[1,0].set_yticklabels(p_classes)
            axes[1,0].get_yticklabels()[pitch_class].set_color('red')
        for pitch_class in highlight_row3:
            axes[1,0].set_yticks(np.arange(12))
            axes[1,0].set_yticklabels(p_classes)
            
    for ax in axes[1,:]:
        ax.set_xticks([])

    plt.savefig('figs/' + path + ".png", transparent=True, dpi=400)
    plt.show()
    

def plot_lmfe(waveforms, titles, ylabel="freq_bin", sr=16000, n_fft = 2048, typ='fan'):
    fig, axes = plt.subplots(nrows=2, ncols=int(np.ceil(len(titles) / 2)), figsize=(8.5, 6.6))
    path = f'lmfe_{typ}_sample'
    n_fft=n_fft
    n=54

    # Compute log mel energy
    

    for i, x in enumerate(waveforms):
        y = x[0].numpy()
        lmfe = speechpy.feature.lmfe(y, sr, fft_length=n_fft, frame_length=1, frame_stride=(0.75), num_filters=n).T
        
        row = i // int(np.ceil(len(titles) / 2))
        col = i % int(np.ceil(len(titles) / 2))
        
        ax = axes[row, col]
        ax.set_title(titles[i])
        
        ax.set_yticks([])
        ax.label_outer()
        ax.set_xticks([])
        im = librosa.display.specshow(lmfe, x_axis='time', y_axis="log", ax=ax)
         # Remove ticks from inner subplots
        ax.tick_params(axis='both', which='both', bottom=False, left=False)
        ax.tick_params(axis='both', which='both', bottom=False, left=False)
        
    
    # # Remove empty subplots, if any
    # if len(titles) < axes.size:
    #     for i in range(len(titles), axes.size):
    #         fig.delaxes(axes.flatten()[i])
    
    plt.colorbar(im, ax=axes)
    for ax in axes[:,0]:
        ax.set_ylabel('log Mel-frequency')
        
    for ax in axes[1,:]:
        ax.set_xticks([])
    
    plt.savefig('figs/' + path+ ".png", transparent=True, dpi=400)
    plt.show()

def plot_mfcc(waveforms, titles, ylabel="freq_bin", sr=16000, n_fft = 2048, typ='fan'):
    fig, axes = plt.subplots(nrows=2, ncols=int(np.ceil(len(titles) / 2)), figsize=(8.5, 6.6))
    
    path = f'mfcc_{typ}_sample'

    n_fft=sr

    win_length = sr // 4 
    hop_length = int(win_length*0.75)

    n_mfcc = 13
    mfcc_transform = torchaudio.transforms.MFCC(sample_rate=sr,n_mfcc=n_mfcc, log_mels=True,
                                                melkwargs={
                                                    "n_fft": n_fft,
                                                    "win_length": win_length,
                                                    "hop_length": hop_length,
                                                    "n_mels": 128,
                                                    "mel_scale": "htk",
                                                },)

    # Compute MFCC
    

    for i, x in enumerate(waveforms):
       
        mfcc = mfcc_transform(x)
        
        row = i // int(np.ceil(len(titles) / 2))
        col = i % int(np.ceil(len(titles) / 2))
        
        ax = axes[row, col]
        ax.set_title(titles[i])
        
        ax.set_yticks([])
        ax.label_outer()
        ax.set_xticks([])
        im = librosa.display.specshow(np.squeeze(mfcc.numpy()), x_axis='time', ax=ax)
         # Remove ticks from inner subplots
        ax.tick_params(axis='both', which='both', bottom=False, left=False)
        ax.tick_params(axis='both', which='both', bottom=False, left=False)
        
    
    # # Remove empty subplots, if any
    # if len(titles) < axes.size:
    #     for i in range(len(titles), axes.size):
    #         fig.delaxes(axes.flatten()[i])
    
    plt.colorbar(im, ax=axes)
    for ax in axes[:,0]:
        ax.set_ylabel('MFCC')
        
    for ax in axes[1,:]:
        ax.set_xticks([])
    
    plt.savefig('figs/' + path+ ".png", transparent=True, dpi=400)
    plt.show()