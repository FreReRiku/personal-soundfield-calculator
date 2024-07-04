'''
delay_time_estimation2.py

What can we find out in this program?

Make graph
- Cross-Correlation using some frequency components
- CSP using some frequency components
- Cross-Correlation using all frequency range
- CSP using all frequency range
- Impulse
'''

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from librosa import stft, magphase, display
from scipy.signal import istft, find_peaks
from scipy.fft import rfft, irfft, fftshift

# パラメータ
L   = 16000*10
N   = 1024
S   = 512
st  = 2000      # スタートポイント
ed  = st + L    # エンドポイント
K   = 5        # 参照するフレーム数
k   = 100       # スタートのフレーム位置(ここからKフレーム用いる)

# 　オーディオデータ読み込み
file_name_impulse1  = '../wav_data/impulse_mic1_ch1.wav'
file_name_impulse2  = '../wav_data/impulse_mic1_ch2.wav'
file_name_origin    = '../wav_data/music1_mono.wav'
file_name_received1 = '../wav_data/music1_room_seed1.wav'
file_name_received2 = '../wav_data/music1_room_seed1234.wav'

h1, _   = sf.read(file_name_impulse1)
h2, _   = sf.read(file_name_impulse2)
x, _    = sf.read(file_name_origin)
y1, _   = sf.read(file_name_received1)
y2, fs  = sf.read(file_name_received2)

# 時間軸
t = np.arange(N)/fs

# インパルスの真のピーク位置
pos = []
for h_ in [h1, h2]:
    pos_peaks, _ = find_peaks(h_, height=0.6)
    pos.append(pos_peaks[0])

# a.単一音声の場合：
#h       = h1
#y       = y1
# b.混合音声の場合：
h  = h1[:2500] + h2[:2500]
#y  = y1 + y2

#x = np.concatenate([np.zeros(300), x])

x = x[st:ed]

#ここでy1の特定の周波数を0にする
fi      = [500, 1000, 2000, 3000, 4000, 5000]  # [Hz]
bin_    = np.round(np.array(fi)/fs*N*2).astype(int)       # 周波数ビンに変換

y2spec = stft(y1, n_fft=2*N, hop_length=S, win_length=2*N)
y2spec[bin_, :] = 0
y1 = librosa.istft(y2spec)

y = y1 + y2
y = y[st:ed]


# スペクトログラム
Xspec = stft(x, n_fft=2*N, hop_length=S, win_length=N)
Yspec = stft(y, n_fft=2*N, hop_length=S, win_length=2*N)

# 時間長をそろえる
D     = np.min([Xspec.shape[1], Yspec.shape[1]])
Xspec = Xspec[:, :D]
Yspec = Yspec[:, :D]

#-------------------------
#   全周波数帯域を使った遅延時間推定
#-------------------------

# 相互相関：全周波数帯域
C       = Yspec[:, k:k+K] * np.conj(Xspec[:, k:k+K])
C_ave   = np.mean(C, axis=1)
c_ave   = irfft(C_ave)

# 白色化相互相関(CSP)：全周波数帯域
eps     = 1e-20
XY      = Yspec[:, k:k+K] * np.conj(Xspec[:, k:k+K])
XYamp   = np.abs(XY)
XYamp[XYamp < eps] = eps
CSP     = irfft(XY/XYamp, axis=0)
CSP_ave = np.mean(CSP, axis=1)

#-------------------------
#   特定帯域を使った遅延時間推定
#-------------------------

# 埋め込み周波数
#fi      = [500, 1000, 2000]  # [Hz]
#bin_    = np.round(np.array(fi)/fs*N*2).astype(int)       # 周波数ビンに変換

# 特定の周波数・特定のフレームを抽出する
Xext    = np.zeros((Xspec.shape[0], K), dtype=np.complex128)
Yext    = np.zeros((Yspec.shape[0], K), dtype=np.complex128)
Yext[bin_, :] = Yspec[bin_, k:k+K]
Xext[bin_, :] = Xspec[bin_, k:k+K]

# 相互相関：特定周波数帯域
C_      = Yext * np.conj(Xext)
C_ave_  = np.mean(C_, axis=1)
c_ave_  = irfft(C_ave_)

# 白色化相互相関(CSP)：特定周波数帯域
eps     = 1e-8
XY_     = Yext * np.conj(Xext)
XYamp_  = np.abs(XY_)
XYamp_[XYamp_<eps] = eps
CSP_    = irfft(XY_/XYamp_, axis=0)
CSP_ave_ = np.mean(CSP_, axis=1)

#-------------------------
# インパルス応答プロット
#-------------------------
fig = plt.figure(num='Impulse')
for p, c in zip(pos, ['r', 'g']):
    plt.axvline(p/fs, color=c, linestyle='--')
plt.plot(t, h[:N])

#-------------------------
# 全周波数帯域のグラフ表示
#-------------------------
fig = plt.figure(num='Correlation using all freq. range')
plt.subplots_adjust(wspace=0.4, hspace=0.6)
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

# 各計算結果をプロット
for p, c in zip(pos, ['r', 'g']):
    ax1.axvline(p/fs, color=c, linestyle='--')
    ax2.axvline(p/fs, color=c, linestyle='--')
ax1.plot(t, c_ave[:N])
ax2.plot(t, CSP_ave[:N])

# 各subplotにラベルを追加
ax1.set_xlabel("Time [s]", fontname="Arial")
ax1.set_ylabel("Cross-Corr.")
ax1.set_title('Cross-Correlation using all frequency range')
ax1.set_xlim([t[0], t[-1]])

ax2.set_xlabel("Time [s]", fontname="Arial")
ax2.set_ylabel("CSP")
ax2.set_title("CSP using all frequency range")
ax2.set_xlim([t[0], t[-1]])

#-------------------------
# 特定周波数帯域のグラフ表示
#-------------------------
fig = plt.figure(num='Correlation using some freq. components')
plt.subplots_adjust(wspace=0.4, hspace=0.6)
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

# 各計算結果をプロット
t = np.arange(N)/fs
for p, c in zip(pos, ['r', 'g']):
    ax1.axvline(p/fs, color=c, linestyle='--')
    ax2.axvline(p/fs, color=c, linestyle='--')
ax1.plot(t, c_ave_[:N])
ax2.plot(t, CSP_ave_[:N])

# 各subplotにラベルを追加
ax1.set_xlabel("Time [s]", fontname="Arial")
ax1.set_ylabel("Cross-Corr.")
ax1.set_title('Cross-Correlation using some frequency components')
ax1.set_xlim([t[0], t[-1]])

ax2.set_xlabel("Time [s]", fontname="Arial")
ax2.set_ylabel("CSP")
ax2.set_title("CSP using some frequency components")
ax2.set_xlim([t[0], t[-1]])

plt.show()
