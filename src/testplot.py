import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

# 　オーディオデータ読み込み
file_name = 'ir44100.wav'
h, fs = sf.read(file_name)

file_name = 'music1_embedded_seed1.wav'
x, fs = sf.read(file_name)

file_name = 'music1_room_seed1.wav'
y, fs = sf.read(file_name)

# 伝達特性を求める
N = 1024 * 4
dt = 1 / fs

H = np.fft.fft(h[:N], N * 2)
freq = np.fft.fftfreq(N * 2, d=dt)

X = np.fft.fft(x[:N], N * 2)
freq = np.fft.fftfreq(N * 2, d=dt)

Y = np.fft.fft(y[:N], N * 2)
freq = np.fft.fftfreq(N * 2, d=dt)

# フーリエ変換の結果を正規化
H = H / (N / 2)
X = X / (N / 2)
Y = Y / (N / 2)
Z = Y / X

# グラフを表示する領域を作成
fig = plt.figure()

# グラフを描画するsubplot領域を作成
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)

# 各subplot領域にデータを渡す
Amp = np.abs(H)
ax1.plot(freq[:N // 2], Amp[:N // 2])

Amp = np.abs(X)
ax2.plot(freq[:N // 2], Amp[:N // 2])

Amp = np.abs(Y)
ax3.plot(freq[:N // 2], Amp[:N // 2])

Amp = np.abs(Z)
ax4.plot(freq[:N // 2], Amp[:N // 2])

# 各subplotにラベルを追加
ax1.set_xlabel("Frequency [Hz]")
ax1.set_ylabel("Amplitude")

ax2.set_xlabel("Frequency [Hz]")
ax2.set_ylabel("Amplitude")

ax3.set_xlabel("Frequency [Hz]")
ax3.set_ylabel("Amplitude")

ax4.set_xlabel("Frequency [Hz]")
ax4.set_ylabel("Amplitude")

# グラフのタイトルを追加
ax1.set_title('H(ω)')
ax2.set_title('X(ω)')
ax3.set_title('Y(ω)')
ax4.set_title('Y(ω)/X(ω)')

plt.show()

