import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

'''---------------
    インパルス応答から伝達特性を求める
---------------'''
file_name = 'ir44100.wav'
h, fs = sf.read(file_name)

# 伝達特性を求める
N = 1024*4
dt = 1/fs
a_range = N
#t = np.arange(0, N*dt, dt)

H = np.fft.fft(h[:a_range], N*2)
freq = np.fft.fftfreq(N*2, d=dt)

# フーリエ変換の結果を正規化
H = H / (N / 2)

# 振幅スペクトル
Amp = np.abs(H)
fig, ax = plt.subplots()
ax.plot(freq[:N // 2], Amp[:N // 2])
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Amplitude")
ax.grid()
plt.show()

'''---------------
    埋込音からX(ω)を求める
---------------'''
file_name2 = 'music1_embedded_seed1.wav'
x, fs = sf.read(file_name2)

# 伝達特性を求める
X = np.fft.fft(x[:a_range], N*2)
freq = np.fft.fftfreq(N*2, d=dt)

# フーリエ変換の結果を正規化
X = X / (N / 2)

# 振幅スペクトル
Amp = np.abs(X)
fig, ax = plt.subplots()
ax.plot(freq[:N // 2], Amp[:N // 2])
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Amplitude")
ax.grid()
plt.show()

'''---------------
    部屋での録音からY(ω)を求める
---------------'''
file_name3 = 'music1_room_seed1.wav'
y, fs = sf.read(file_name3)

# 伝達特性を求める
Y = np.fft.fft(y[:a_range], N*2)
freq = np.fft.fftfreq(N*2, d=dt)

# フーリエ変換の結果を正規化
Y = Y / (N / 2)

# 振幅スペクトル
Amp = np.abs(Y)
fig, ax = plt.subplots()
ax.plot(freq[:N // 2], Amp[:N // 2])
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Amplitude")
ax.grid()
plt.show()


# 伝達特性を求める
Z = X / Y

# 振幅スペクトル
Amp = np.abs(Z)
fig, ax = plt.subplots()
ax.plot(freq[:N // 2], Amp[:N // 2])
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Amplitude")
ax.grid()
plt.show()

# ifftをする
H_ifft = np.fft.ifft(H, N*2)
Z_ifft = np.fft.ifft(Z, N*2)

fig = plt.figure()

Amp = np.abs(H_ifft)
ax1 = fig.add_subplot()
ax1.plot(freq[:N // 2], Amp[:N // 2])
ax1.set_xlabel("Frequency [Hz]")
ax1.set_ylabel("Amplitude")

Amp = np.absolute(Z_ifft)
ax2 = fig.add_subplot()
ax2.plot(freq[:N // 2], Amp[:N // 2])
ax2.set_xlabel("Frequency [Hz]")
ax2.set_ylabel("Amplitude")

plt.show()
