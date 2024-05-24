"""
    オーディオ透かしの埋め込み
    from Y. Nakashima et. al., "Indoor Positioning System Using Digital AudioWatermarking," IEICE Trans., vol.E.94-D, no.11, Nov. 2011.
"""

'''---------------
    インポート
---------------'''
import os
import urllib.request  # ファイルを読み込むためのモジュール
import numpy as np
# 信号処理関係
import scipy as sp
import scipy.signal as sg
import librosa as lb
# プロット関係
import matplotlib.pyplot as plt
# オーディオ関係
import soundfile as sf
from Perceived_Human_Hearing import Hearing_Threshold
from presudo_random import *
# パラメータは settings.py から読み込む
from settings import *

'''---------------
    オーディオデータ読み込み
---------------'''
file_name = 'music1.wav'
if not os.path.isfile(file_name):
    url = 'https://github.com/Shimamura-Lab-SU/Sharing-Knowledge-Database/blob/master/python_exercise/music1.wav?raw=true'  # Githubでrawデータを扱うためのurl
    urllib.request.urlretrieve(url, file_name)  # 音楽ファイル

x, fs = sf.read(file_name)
if np.array(x).ndim == 2:
    x = x[:, 0]  # ステレオ->モノラル変換

'''---------------
    パラメータ
---------------'''
# 周波数閾値
f_th = np.floor(N * TH / fs).astype(int)  # 周波数ビンに変換
# 埋め込み周波数上限
f_sp = Hb * Ht
# はじめの f_bias フレームには埋め込まない(center=Trueの影響)
f_bias = 1
# 埋め込み音声のパワー補正 (大きいほど埋め込みパワーが小さい)
Bias_power = 40
# seed値
seed = 2

k = 100
l = 100
n = 0
'''---------------
    事前準備
    埋め込みの手順1., 3., 4.の一部を先に行う
---------------'''
# 窓関数の作成 (手順1.)
win_t = sg.windows.hann(N)  # sin窓(=Hanning窓)の準備

# m_[f mod 2]の計算   (式(4)の一部)
m = np.array([-2 * (i % 2) + 1 for i in range(Wb * Wt)])

# 心理音響モデルに基づく振幅の計算 (手順4.の一部)
A = Hearing_Threshold(n_fft=N, fs=fs, bias=Bias_power, sup=20)
A_tile = np.tile(A[:Hb * Ht, np.newaxis], (1, Wb * Wt))



'''---------------
    埋め込み
---------------'''
#   手順2.  DFT (=STFT)   (式(3))
f, t, Zxx = sg.stft(x, nperseg=N, noverlap=N - S, nfft=N, window=win_t, boundary=None)

# 埋め込み
for i in range(1000):
   Zxx[k, l+i] = 0

# 逆FFT      (式(8))
#_, y = sg.istft(Y, nperseg=N, noverlap=N - S, window=win_t, boundary=True)
#y = np.real(y)
#outputs.append(y)

'''---------------
    ファイル保存
---------------'''
#file_base = os.path.splitext(file_name)[0]
#new_file_name = '{0}_embedded_seed{1}.wav'.format(file_base, seed)
#sf.write(new_file_name, y * 0.95 / np.max(abs(y)), fs)

'''---------------
    図表示
---------------'''
fig = plt.figure()
plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
