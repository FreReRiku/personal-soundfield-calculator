import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import pyroomacoustics as pra
import soundfile as sf
# パラメータは settings.py から読み込む
from settings import *

'''---------------
    音声ファイルパラメータ
---------------'''
# wavファイルを読み込む
audios = []
'''
for seed in seeds:
    fs, audio = wavfile.read('music1_mono.wav'.format(seed))
    audios.append(audio)
'''
fs, audio = wavfile.read('music2_emb_min.wav')
audios.append(audio)
fs, audio = wavfile.read('music2_mono.wav')
audios.append(audio)

'''---------------
    部屋の設定
---------------'''
# 残響時間と部屋の寸法
rt60 = 0.3  # seconds
room_dim = [3.52, 3.52, 2.4]  # meters ここを二次元にすると二次平面の部屋になります

# Sabineの残響式から壁面の平均吸音率と鏡像法での反射回数の上限を決めます
e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)

# 壁の材質を決める
m = pra.make_materials(
    ceiling="plasterboard",
    floor="carpet_cotton",
    east="plasterboard",
    west="plasterboard",
    north="plasterboard",
    south="plasterboard",
)

# 部屋をつくります
# fsは生成されるインパルス応答のサンプリング周波数です。入力する音源があるならそれに合わせる。
room = pra.ShoeBox(
    room_dim, t0=0.0, fs=fs, materials=pra.Material(e_absorption), max_order=5
)

# マイク設置
mic_loc = [1.75, 1.75, 1.6]
room.add_microphone(mic_loc)

# 音源ごとに座標情報を与え、`room`に追加していきます。
# オプションで delay を追加することもできます。
room.add_source([3.4, 0.5, 0.5], signal=audios[0])
room.add_source([3.4, 2.3, 0.5], signal=audios[1])

# 部屋表示
fig, ax = room.plot()
ax.set_xlim([0, 3.6])
ax.set_ylim([0, 3.6])
ax.set_zlim([0, 2.5])

'''---------------
    シミュレーション & 保存
---------------'''

# インパルス応答を計算
room.compute_rir()
# インパルス応答を保存
for i, ir_ in enumerate(room.rir):
    for j, ir in enumerate(ir_):
        ir_signal = ir
        ir_signal /= np.max(np.abs(ir_signal)) # 可視化のため正規化
        sf.write('impulse2_mic{0}_ch{1}.wav'.format(i+1,j+1), ir_signal, fs)

# シミュレーション
separate_recordings = room.simulate(return_premix=True)

# 各音源のみを再生した音を保存
for i, sound in enumerate(separate_recordings):
    recorded        = sound[0, :]
    writefilename   = "music1_room_seed{0}.wav".format(seeds[i])
    sf.write(writefilename, recorded / np.max(recorded) * 0.95, fs)


# ミックスされた音源を保存
mixed_recorded  = np.sum(separate_recordings, axis=0)[0,:]
writefilename = "music2_room_seed{0}&{1}.wav".format(seeds[0], seeds[1])
sf.write(writefilename, mixed_recorded / np.max(mixed_recorded) * 0.95, fs)

'''---------------
    図示
---------------'''
plt.show()