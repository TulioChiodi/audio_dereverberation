import math
import os
import pickle
import random
import sys

import librosa
import numpy as np
import pandas as pd
import torch
from scipy.signal import stft
from torch.utils.data import Dataset
from pathlib import Path

'''
Miscellaneous utilities
'''

class AudioDataset(Dataset):
    def __init__(self, dataset_rootdir, metadata_filepath, sample_rate=48000, audio_duration=5.970666667):
        self.dataset_rootdir = dataset_rootdir
        self.metadata = pd.read_csv(metadata_filepath)
        self.sample_rate = sample_rate
        self.audio_duration = audio_duration

        
    def __len__(self):
        return len(self.metadata)

    
    def __getitem__(self, idx):
        
        noisy_filename = self.metadata.loc[idx, 'noisy_filename']
        clean_filename = self.metadata.loc[idx, 'clean_filename']
        
        noisy_audio_file = os.path.join(self.dataset_rootdir, "noisy", noisy_filename)
        clean_audio_file = os.path.join(self.dataset_rootdir, "clean", clean_filename)
        
        noisy_waveform, _ = librosa.load(noisy_audio_file, sr=self.sample_rate, duration=self.audio_duration)
        clean_waveform, _ = librosa.load(clean_audio_file, sr=self.sample_rate, duration=self.audio_duration)
        
        noisy_waveform_padded = self.pad_audio(noisy_waveform)
        clean_waveform_padded = self.pad_audio(clean_waveform)
            
        noisy_waveform_tensor = torch.from_numpy(noisy_waveform_padded).unsqueeze(dim=0)
        clean_waveform_tensor = torch.from_numpy(clean_waveform_padded).unsqueeze(dim=0)
        
        return noisy_waveform_tensor, clean_waveform_tensor
    
    
    def pad_audio(self, waveform):
        waveform_len = len(waveform)
        desired_len = int(self.sample_rate * self.audio_duration)
        if waveform_len < desired_len:
            num_samples = desired_len - waveform_len
            waveform = np.pad(waveform, (0, num_samples))
        return waveform


def get_data_path_wavs(dataset_name, data_dir):
    data_dir = Path(data_dir)
    wavs_dir = data_dir / dataset_name / "wavs"
    return wavs_dir


def get_data_path_metadata(dataset_name, data_dir, split):
    data_dir = Path(data_dir)
    metadata_dir = data_dir / dataset_name / f"{split}_metadata.csv"
    return metadata_dir


def get_data_path(dataset_name, data_dir):
    splits = [
        "train",
        "val",
        "test",
    ]

    paths = [get_data_path_wavs(dataset_name, data_dir)]
    metadata_paths = [
        get_data_path_metadata(dataset_name, data_dir, split) for split in splits
    ]

    paths.extend(metadata_paths)

    return paths


def calc_frames_num(signal_length, window_size=512, hop_length=128):
    return 5 + ((signal_length - window_size) // hop_length)


def adjust_signal_length(sample_rate, desired_audio_duration=6, window_size=512, hop_length=128):
    """Adjust signal length to be exactly divisible by 8."""
    signal_length = sample_rate * desired_audio_duration
    frames_num = calc_frames_num(signal_length, window_size, hop_length)
    if frames_num % 8 != 0:
        frames_num = round(frames_num / 8) * 8

    return int(np.floor(((frames_num - 1) * hop_length) + window_size - (4 * hop_length)))


def save_model(model, optimizer, state, path):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module  # save state dict of wrapped module
    if len(os.path.dirname(path)) > 0 and not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'state': state,  # state of training loop (was 'step')
    }, path)


def load_model(model, optimizer, path, cuda):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module  # load state dict of wrapped module
    if cuda:
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location='cpu')
    try:
        #model.load_state_dict(checkpoint['model_state_dict'])
        model.load_state_dict(torch.load(checkpoint['model_state_dict'],
                                    map_location=lambda storage, location: storage),
                                    strict=False)
    except:
        # work-around for loading checkpoints where DataParallel was saved instead of inner module
        from collections import OrderedDict
        model_state_dict_fixed = OrderedDict()
        prefix = 'module.'
        for k, v in checkpoint['model_state_dict'].items():
            if k.startswith(prefix):
                k = k[len(prefix):]
            model_state_dict_fixed[k] = v
        model.load_state_dict(model_state_dict_fixed)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'state' in checkpoint:
        state = checkpoint['state']
    else:
        # older checkpoints only store step, rest of state won't be there
        state = {'step': checkpoint['step']}
    return state


def spectrum_fast(x, nperseg=512, noverlap=128, window='hamming', cut_dc=True,
                  output_phase=True, cut_last_timeframe=True):
    '''
    Compute magnitude spectra from monophonic signal
    '''

    f, t, seg_stft = stft(x,
                        window=window,
                        nperseg=nperseg,
                        noverlap=noverlap)

    #seg_stft = librosa.stft(x, n_fft=nparseg, hop_length=noverlap)

    output = np.abs(seg_stft)

    if output_phase:
        phase = np.angle(seg_stft)
        output = np.concatenate((output,phase), axis=-3)

    if cut_dc:
        output = output[:,1:,:]

    if cut_last_timeframe:
        output = output[:,:,:-1]

    #return np.rot90(np.abs(seg_stft))
    return output


def segment_waveforms(predictors, target, length):
    '''
    segment input waveforms into shorter frames of
    predefined length. Output lists of cut frames
    - length is in samples
    '''

    def pad(x, d):
        pad = np.zeros((x.shape[0], d))
        pad[:,:x.shape[-1]] = x
        return pad

    cuts = np.arange(0,predictors.shape[-1], length)  #points to cut
    X = []
    Y = []
    for i in range(len(cuts)):
        start = cuts[i]
        if i != len(cuts)-1:
            end = cuts[i+1]
            cut_x = predictors[:,start:end]
            cut_y = target[:,start:end]
        else:
            end = predictors.shape[-1]
            cut_x = pad(predictors[:,start:end], length)
            cut_y = pad(target[:,start:end], length)
        X.append(cut_x)
        Y.append(cut_y)
    return X, Y


def gen_dummy_waveforms(n, out_path):
    '''
    Generate random waveforms as example for the submission
    '''
    sr = 16000
    max_len = 10  #secs

    for i in range(n):
        len = int(np.random.sample() * max_len * sr)
        sound = ((np.random.sample(len) * 2) - 1) * 0.9
        filename = os.path.join(out_path, str(i) + '.npy')
        np.save(filename, sound)


def gen_fake_task1_dataset():
    l = []
    target = []
    for i in range(4):
        n = 160000
        n_target = 160000
        sig = np.random.sample(n)
        sig_target = np.random.sample(n_target).reshape((1, n_target))
        target.append(sig_target)
        sig = np.vstack((sig,sig,sig,sig))
        l.append(sig)

    output_path = '../prova_pickle'
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    with open(os.path.join(output_path,'training_predictors.pkl'), 'wb') as f:
        pickle.dump(l, f)
    with open(os.path.join(output_path,'training_target.pkl'), 'wb') as f:
        pickle.dump(target, f)
    with open(os.path.join(output_path,'validation_predictors.pkl'), 'wb') as f:
        pickle.dump(l, f)
    with open(os.path.join(output_path,'validation_target.pkl'), 'wb') as f:
        pickle.dump(target, f)
    with open(os.path.join(output_path,'test_predictors.pkl'), 'wb') as f:
        pickle.dump(l, f)
    with open(os.path.join(output_path,'test_target.pkl'), 'wb') as f:
        pickle.dump(target, f)
    '''
    np.save(os.path.join(output_path,'training_predictors.npy'), l)
    np.save(os.path.join(output_path,'training_target.npy'), l)
    np.save(os.path.join(output_path,'validation_predictors.npy'), l)
    np.save(os.path.join(output_path,'validation_target.npy'), l)
    np.save(os.path.join(output_path,'test_predictors.npy'), l)
    np.save(os.path.join(output_path,'test_target.npy'), l)
    '''

    with open(os.path.join(output_path,'training_predictors.pkl'), 'rb') as f:
        data = pickle.load(f)
    with open(os.path.join(output_path,'training_target.pkl'), 'rb') as f:
        data2 = pickle.load(f)

    print (data[0].shape)
    print (data2[0].shape)
