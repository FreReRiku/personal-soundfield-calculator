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
''''''
file_name = 'music2_mono.wav'
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
    seedごとに埋め込み
---------------'''
outputs = []
for seed in seeds:
    '''---------------
        事前準備
        seedごとに疑似乱数を生成
    ---------------'''
    # 乱数配列を作成し，縦にHt個，横にWt個拡張する
    wc = presudo_random(Hb, Wb, Ht, Wt, seed=seed)

    # signの計算   (式(4))
    sign = np.multiply(np.tile(m[np.newaxis, :], (Hb * Ht, 1)), wc)

    '''---------------
        埋め込み
    ---------------'''
    #   手順2.  DFT (=STFT)   (式(3))
    _, _, X = sg.stft(x, nperseg=N, noverlap=N - S, nfft=N, window=win_t, boundary=None)
    X = X[:, :2 * (X.shape[1] // 2)]
    Xa, Xp = lb.magphase(X)  # 振幅・位相を分離

    #   手順4.  フーリエ係数の変換(埋め込み)
    Ya = np.zeros(Xa.shape)

    # 閾値(TH=6000Hz)以下:  (式(5))
    # 埋め込む信号
    A_sgn_ = np.multiply(A_tile, sign)
    A_sgn = np.tile(A_sgn_, (1, Xa.shape[1] // (2 * Wb) + 1))
    A_sgn = np.roll(A_sgn, f_bias, axis=1)  # 不要ならコメント：実際のことを考えて数フレームずらしてみる
    A_sgn = A_sgn[:, :Xa.shape[1]]
    # 埋め込み
    Ya[:f_th, :] = Xa[:f_th, :] + A_sgn[:f_th, :]

    # 閾値(TH=6000Hz)以上，上限以下:  (式(6))
    # 埋め込み (sign = 1 -> Xa, sign = -1 -> 0)
    Ya[f_th:f_sp, :] = Xa[f_th:f_sp, :] * (A_sgn[f_th:f_sp, :] >= 0)

    # 上限以上:   (式にはないけど，超高周波数帯域は多分加工なし)
    Ya[f_sp:, :] = Xa[f_sp:, :]

    # 振幅は0以上に固定 (式にはないけど追加)
    Ya[Ya < 0] = 0

    # 逆FFT      (式(8))
    Y = Ya * np.exp(1.j * np.angle(Xp))
    _, y = sg.istft(Y, nperseg=N, noverlap=N - S, window=win_t, boundary=True)
    y = np.real(y)
    outputs.append(y)

    '''---------------
        ファイル保存
    ---------------'''
    file_base = os.path.splitext(file_name)[0]
    new_file_name = '{0}_embedded_seed{1}.wav'.format(file_base, seed)
    sf.write(new_file_name, y * 0.95 / np.max(abs(y)), fs)

'''---------------
    図表示
---------------'''
fig = plt.figure()
fig.subplots_adjust(hspace=0.6, wspace=0.4)  # 余白の調整
for i, y in enumerate(outputs):
    x = x[:len(y)]  # 長さ揃える
    time = np.arange(len(y)) / fs  # 時間インデックス
    # プロット
    ax = fig.add_subplot(len(seeds), 1, i + 1)
    ax.plot(time[::10], x[::10], label='Input')
    ax.plot(time[::10], y[::10], label='Embedded seed={0}'.format(seeds[i]))
    ax.legend(loc="lower right")
    ax.set_xlim([0, np.max(time)])
    ax.set_ylabel('Value', fontname="MS Gothic")
    ax.set_xlabel('Time [s]', fontname="MS Gothic")
plt.show()
