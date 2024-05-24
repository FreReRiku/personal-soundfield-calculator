import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import soundfile as sf
import librosa

# 　オーディオデータ読み込み
file_name = 'music1.wav'
h, fs = librosa.load(file_name)
time = np.arange(0,len(h))/ fs

fig = plt.figure()
plt.plot(time, h)
plt.xlabel("Time(s)")
plt.ylabel("Sound Amplitude")

plt.show()