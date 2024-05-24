import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import soundfile as sf

# 　オーディオデータ読み込み
file_name = 'impulse_mic1_ch1.wav'
h, fs = sf.read(file_name)

file_name = 'music1_mono.wav'
x, fs = sf.read(file_name)

file_name = 'music1_room_seed1.wav'
y, fs = sf.read(file_name)

f, t, Zxx = signal.stft(x, fs, nperseg=1024)
plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')

plt.show()

f, t, Zyy = signal.stft(y, fs, nperseg=1024)
plt.pcolormesh(t, f, np.abs(Zyy), shading='gouraud')
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')

plt.show()