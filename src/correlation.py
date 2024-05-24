import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import correlate

# 　オーディオデータ読み込み
file_name = '../wav_data/impulse_mic1_ch1.wav'
h, fs = sf.read(file_name)

file_name = '../wav_data/music1_mono.wav'
x, fs = sf.read(file_name)

file_name = '../wav_data/music1_room_seed1.wav'
y, fs = sf.read(file_name)

N = 1024*64
st = 0

# 相互相関
c = correlate(y[st:st+N], x[st:st+N])

# グラフにプロット
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(h)
#ax1.plot(c)
ax1.plot(c[len(c)//2:]/np.max(c))
#plt.vlines(np.argmax(s_)*dt, ymin=-1, ymax=1, colors='r', label='True Position')

plt.show()
