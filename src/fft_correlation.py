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
N = 1024*4
st = 800
dt = 1 / fs
t = np.arange(0, N * dt * 2, dt)

print(len(t))

H = np.fft.rfft(h[:N], N * 2)
freq = np.fft.rfftfreq(N * 2, d=dt)

X = np.fft.rfft(x[st:st + N], N * 2)
freq = np.fft.rfftfreq(N * 2, d=dt)

Y = np.fft.rfft(y[st:st + N], N * 2)
freq = np.fft.rfftfreq(N * 2, d=dt)

S = Y * np.conj(X)

# ifftをかける
s_ = np.fft.irfft(S, N * 2)
print(s_)

# グラフにプロット
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(h)
ax1.plot(s_/np.max(s_))
plt.vlines(np.argmax(s_)*dt, ymin=-1, ymax=1, colors='r', label='True Position')


X_ext = np.zeros(X.size, dtype=np.complex128)
Y_ext = np.zeros(Y.size, dtype=np.complex128)
index = [10]

for i in index:
    X_ext[i] = X[i]
    Y_ext[i] = Y[i]


S_ext = Y_ext * np.conj(Y_ext)

# ifftをかける
s_ext = np.fft.irfft(S_ext, N * 2)

# グラフにプロット
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(t, s_ext)
plt.vlines(np.argmax(s_)*dt, ymin=-np.max(s_ext), ymax=np.max(s_ext), colors='r', label='True Position')

plt.show()

