# perceived_human_hearing.py
# About: 最小可聴レベル(dBから振幅値に変換したもの)を返す関数

import pandas as pd
import scipy as sp
import numpy as np
import librosa as lb

def Hearing_Threshold(n_fft=1024, fs=16000, bias=0, sup=20):

    # n_fft : FFT点数
    # fs    : サンプリング周波数
    # bias  : 0dB基準 (音圧[dB]をbias[dB]で引いて調整する)
    # sup   : 最大音圧

    # Create Interpolated Curve
    data    = pd.read_csv('equal_loudness.csv')
    freq    = np.array(data['frequency'])
    level   = np.array(data['level'])
    curve   = sp.interpolate.CubicSpline(freq, level, bc_type='natural')

    # Frequencies that lebel estimates
    freq_   = lb.fft_frequencies(sr=fs, n_fft=n_fft)
    level_  = curve(freq_)-bias
    level_[level_>sup] = sup

    # dB -> linear
    Mag     = 10**((level_)/20)

    # DC成分除去
    Mag[0]  = 0

    return Mag

