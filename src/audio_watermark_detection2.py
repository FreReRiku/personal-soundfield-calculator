"""
    オーディオ透かしの検知
    from Y. Nakashima et. al., "Indoor Positioning System Using Digital AudioWatermarking," IEICE Trans., vol.E.94-D, no.11, Nov. 2011.
"""

'''---------------
    インポート
---------------'''
import numpy as np
# 信号処理関係
import scipy as sp
import scipy.signal as sg
import librosa as lb
# プロット関係
import matplotlib.pyplot as plt
# オーディオ関係
import soundfile as sf
from presudo_random import presudo_random
# パラメータは settings.py から読み込む
from settings import *
from scipy.signal import find_peaks

'''---------------
    疑似乱数のseed値
---------------'''
# seed値
seed = 1

'''---------------
    オーディオデータ読み込み
---------------'''
#単一音源音声 (直接音源)
#z, fs = sf.read('music1_embedded_seed{0}.wav'.format(seed))
# 単一音源音声 (部屋再生)
z, fs = sf.read('music1_room_seed{0}.wav'.format(seed))
# 複数音源-混合音声 (部屋再生)
#z, fs = sf.read('music1_room_seed{0}&{1}.wav'.format(seeds[0], seeds[1]))

'''---------------
    パラメータ(一意に決まるもの)
---------------'''
# 周波数閾値
f_th = np.floor(N * TH / fs).astype(int)  # 周波数ビンに変換
# 埋め込み周波数上限
f_sp = Hb * Ht

# 1パターンブロックに含まれる時間方向のフレーム数
Wn = Wt * Wb
# 1パターンブロックに含まれるサンプル数
Nn = (Wn - 1) * S + N
# パターンブロックの周期
Peak_period = Wn * S

'''---------------
    事前準備
---------------'''
# ⊿i の範囲
i_range = 20000

## 窓関数の作成 (手順1.)
win_t = sg.windows.hann(N)  # sin窓(=Hanning窓)の準備

# 乱数配列を作成
wc = presudo_random(Hb, Wb, 1, 1, seed=seed)

'''---------------
    検知
---------------'''
sc_array = []
for i in range(0, i_range, delta):  # ⊿=8個飛ばし
    #   手順2.  DFT (=STFT)  (式(9)~(10))
    _, _, Z = sg.stft(z[i:i + Nn], nperseg=N, noverlap=N - S, nfft=N, window=win_t, boundary=None)
    Za, Zp = lb.magphase(Z)  # 振幅・位相を分離

    #   手順3.  正規化 (式(11))
    Za_ = Za / (np.mean(Za, axis=0, keepdims=True) + 10 ** (-6))

    #   手順4.  対数振幅のフレーム差分 (式(12))
    D = np.log(Za_[:, ::2] + 10 ** (-6)) - np.log(Za_[:, 1::2] + 10 ** (-6))  # 偶数番 - 奇数番

    #   手順5.  ρの計算 (式(13))
    rho = np.array([np.sum(D[i * Ht:(i + 1) * Ht, :], axis=0) for i in range(Hb)])  # 周波数方向にHtごとに和

    #   手順6.  s^cの計算 (式(14), (15))
    rho_ = np.mean(rho)  # パターンブロック内の平均 (式(15))
    sc = np.sum(wc * (rho - rho_)) / np.sqrt(np.sum((wc * (rho - rho_)) ** 2))

    sc_array.append(sc)

    if i % 1000 == 0:
        print('{0}/{1} is done.'.format(i, i_range))

# scを正規化
sc_array = sc_array / np.max(Wb)

# 検知によるピーク位置
peak, _   = find_peaks(sc_array, height=0.12)
print(peak[0])

# 時間軸
time = list(range(0, i_range, delta))
'''---------------
    表示系
---------------'''
x = list(range(341, i_range, Peak_period - 1))

plt.figure()
plt.vlines(x, ymin=-0.3, ymax=0.3, colors='r', label='True Position')
plt.plot(time, sc_array, label='Correlation')
plt.xlabel("時間[sample]", fontname="Arial", fontsize=24)
plt.ylabel("相関値", fontname="Arial", fontsize=24)
plt.legend(loc="lower right")
plt.show()
