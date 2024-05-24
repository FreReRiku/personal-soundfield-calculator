import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

# 　オーディオデータ読み込み
file_name = 'impulse_mic1_ch1.wav'
h, fs = sf.read(file_name)

file_name = 'music1_mono.wav'
x, fs = sf.read(file_name)

file_name = 'music1_room_seed1.wav'
y, fs = sf.read(file_name)

# 伝達特性を求める
N = 1024 * 2
st = 1000
dt = 1 / fs
t = np.arange(0, N * dt * 2, dt)

H = np.fft.rfft(h[:N], N * 2)
freq = np.fft.rfftfreq(N * 2, d=dt)

X = np.fft.rfft(x[st:st + N], N * 2)
freq = np.fft.rfftfreq(N * 2, d=dt)

Y = np.fft.rfft(y[st:st + N], N * 2)
freq = np.fft.rfftfreq(N * 2, d=dt)

Z = Y / X

# ifftをかける
h_ = np.fft.irfft(H, N * 2)
z_ = np.fft.irfft(Z, N * 2)

# グラフを表示する領域を作成
fig = plt.figure()

# グラフを描画するsubplot領域を作成
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

# 各subplot領域にデータを渡す
ax1.plot(t, h_)
ax2.plot(t, z_)

# 各subplotにラベルを追加
ax1.set_xlabel("Time [s]", fontname="Arial")
ax1.set_ylabel("Value")
ax1.set_title('Ground Truth of Impulse Response')

ax2.set_xlabel("Time [s]", fontname="Arial")
ax2.set_ylabel("Estimated Impulse Response")

plt.show()


# Zと同じ要素数のlistを作成
A = np.zeros(Z.size, dtype=np.complex128)

# Zの特定の周波数の値を取り出す
Z_i_index = [10, 30, 50, 100, 200, 300]

for i in Z_i_index:
    A[i] = Z[i]


# ifftをかける
A_ = np.fft.irfft(A, N * 2)

# グラフにプロット
fig = plt.figure()
ax3 = fig.add_subplot(2, 1, 1)
ax3.plot(t, A_)

plt.show()
