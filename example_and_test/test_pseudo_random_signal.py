# playground_pseudo_random_signal.py

import numpy as np
from numpy.random import Generator, PCG64
import scipy.signal as sg
import scipy.ndimage.filters as flt
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt

'''
N = 512  # 周波数分解能
Wb = 15  # ブロックあたりのタイル数(横)
Hb = 42  # ブロックあたりのタイル数(縦)
Ht = 6  # 1タイルあたりの周波数ビン数(縦)
Wt = 2  # 1タイルあたりの周波数ビン数(横)
S = N // 2  # FFTのシフト幅
delta = 8  # 検知：サンプルの飛ばし幅
# 周波数閾値
TH = 6000
# 疑似乱数を生成するseed値の候補
seeds = [1, 1234]
'''

rg_mt = Generator(PCG64(seed=1234)) #seed=1234はseedという引数に1234という値を入れている

# タイルごとの疑似乱数配列を作成     (式(4)の一部)
wc = rg_mt.integers(0, 2, (5, 4))

wc = wc.reshape((5, 4)) # ランダム信号 (0 or 1)の配列

wc[wc == 0] = -1  # 0 -> -1

print(f'拡張前のタイル\n{wc}')


# 乱数配列を縦にht = 3個，横にwt =2 個拡張する
wc = np.repeat(wc, 3, axis=0)
wc = np.repeat(wc, 2, axis=1)

print(f'拡張後のタイル\n{wc}')
