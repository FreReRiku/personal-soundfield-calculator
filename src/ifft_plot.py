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
t = np.arange(0, N*dt*2, dt)

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

# ifftをする
H_ifft = np.fft.ifft(H, N*2)
Z_ifft = np.fft.ifft(Z, N*2)

# グラフを表示する領域を作成
fig = plt.figure()

# グラフを描画するsubplot領域を作成
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

# 各subplot領域にデータを渡す
ax1.plot(t, H_ifft)
ax2.plot(t, Z_ifft)

# 各subplotにラベルを追加
ax1.set_xlabel("Time[s]", fontname="Arial")
ax1.set_ylabel("Signal")

ax2.set_xlabel("Time[s]", fontname="Arial")
ax2.set_ylabel("Signal")

plt.show()
