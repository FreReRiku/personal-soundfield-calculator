import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from IPython.display import display, Audio
import pyroomacoustics as pra
import soundfile as sf

# 残響時間と部屋の寸法
rt60 = 0.1  # seconds
room_dim = [3.52, 3.52, 2.4]  # meters ここを二次元にすると二次平面の部屋になります

# Sabineの残響式から壁面の平均吸音率と鏡像法での反射回数の上限を決めます
e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)

# 壁の材質を決める
m = pra.make_materials(
    ceiling="ceiling_perforated_gypsum_board",
    floor="carpet_thin",
    east="plasterboard",
    west="plasterboard",
    north="plasterboard",
    south="plasterboard",
)

# 部屋をつくります
# fsは生成されるインパルス応答のサンプリング周波数です。入力する音源があるならそれに合わせる。
room = pra.ShoeBox(
    room_dim, t0=0.0, fs=16000, materials=m, max_order=max_order - 5
)

# マイク設置
mic_loc = [1.75, 1.75, 1.6]
room.add_microphone(mic_loc)

# wavファイルを読み込んで配置してみます
fs, audio1 = wavfile.read('embedded_music3_seed1234.wav')
fs, audio2 = wavfile.read('embedded_music3_seed1.wav')

# 音源ごとに座標情報を与え、`room`に追加していきます。
# オプションで delay を追加することもできます。
room.add_source([3.4, 0.5, 0.5], signal=audio1)
room.add_source([3.4, 2.3, 0.5], signal=audio2)

fig, ax = room.plot()
ax.set_xlim([0, 3.6])
ax.set_ylim([0, 3.6])
ax.set_zlim([0, 2.5])
plt.show()

# インパルス応答を計算
room.compute_rir()

# インパルス応答の波形データを保存
ir_signal = room.rir[0][0]
ir_signal /= np.max(np.abs(ir_signal))  # 可視化のため正規化
sf.write(f"ir{fs}.wav", ir_signal, fs)

# 伝達特性を求める
N = 1024
dt = 0.0005
F = np.fft.fft(ir_signal)
freq = np.fft.fftfreq(N, d=dt)

# フーリエ変換の結果を正規化
F = F / (N / 2)

# 振幅スペクトル
Amp = np.abs(F)
fig, ax = plt.subplots()
ax.plot(freq[:N // 2], Amp[:N // 2])
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Amplitude")
ax.grid()
plt.show()

# シミュレーション
room.simulate()
recorded = np.array(room.mic_array.signals[0, :])
rmax = np.max(recorded)
plt.plot(recorded)

# シミュレーションした音を書き出す
writefilename = "simu3_seed1&1234.wav"
sf.write(writefilename, recorded / rmax * 0.5, fs)
