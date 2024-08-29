# pseudo_random.py

import numpy as np
from numpy.random import Generator, PCG64
import scipy.signal as sg
import scipy.ndimage.filters as flt
from scipy.fft import fft, ifft 
import matplotlib.pyplot as plt


wb = 15  # ブロックあたりのタイル数(横)
hb = 42  # ブロックあたりのタイル数(縦)
ht = 6  # 1タイルあたりの周波数ビン数(縦)
wt = 2  # 1タイルあたりの周波数ビン数(横)

def pseudo_random(hb, wb, ht, wt, seed=1234):
    rg_mt = Generator(PCG64(seed=seed)) #引数seedにseed=1234を代入している

    # タイルごとの疑似乱数配列を作成     (式(4)の一部)
    wc = rg_mt.integers(0, 2, (hb, wb))     # Generator.integers
    # wc = wc.reshape((hb,wb)) # ランダム信号 (0 or 1)の配列
    wc[wc == 0] = -1  # 0 -> -1
    # 乱数配列を縦にht個，横にwt個拡張する
    wc = np.repeat(wc, ht, axis=0)
    wc = np.repeat(wc, wt, axis=1)

    return wc
