# perceived_human_hearing.py
# About: 最小可聴レベル(dBから振幅値に変換したもの)を返す関数

import pandas as pd
import scipy as sp
import numpy as np
import librosa as lb
import matplotlib as plt

n_fft=1024
fs=16000
bias=0
sup=20

def Hearing_Threshold(n_fft, fs, bias, sup):

    # n_fft : FFT点数
    # fs    : サンプリング周波数
    # bias  : 0dB基準 (音圧[dB]をbias[dB]で引いて調整する)
    # sup   : 最大音圧

    # Create Interpolated Curve
    data    = pd.read_csv('equal_loudness.csv')
    print(f'Results: line 22 (pd.read_csv)\n{data}')
    freq    = np.array(data['frequency'])
    print(f'Results: line 24 np.array\n{freq}')
    level   = np.array(data['level'])
    print(f'Results: line 26 np.array\n{level}')
    curve   = sp.interpolate.CubicSpline(freq, level, bc_type='natural')
    print(f'Results: line 28 sp.interpolate.CubicSpline\n{curve}')

    # Frequencies that lebel estimates
    freq_   = lb.fft_frequencies(sr=fs, n_fft=n_fft)
    level_  = curve(freq_)-bias
    level_[level_>sup] = sup

    # dB -> linear
    Mag     = 10**((level_)/20)
    
    # DC成分除去
    Mag[0]  = 0

    return Mag

Hearing_Threshold(n_fft, fs, bias, sup)
