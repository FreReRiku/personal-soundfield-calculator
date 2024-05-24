import pyroomacoustics as pra
import matplotlib.pyplot as plt
from IPython.display import display, Audio
import numpy as np
import soundfile as sf
from async_io import Audio_Player, Process

if __name__ == '__main__':

    # 残響時間と部屋の寸法
    rt60 = 0.5  # seconds
    room_dim = [3.52, 3.52, 2.4]  # meters ここを二次元にすると二次平面の部屋になります

    # Sabineの残響式から壁面の平均吸音率と鏡像法での反射回数の上限を決めます
    e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)

    # 部屋をつくります
    # fsは生成されるインパルス応答のサンプリング周波数です。入力する音源があるならそれに合わせる。
    room = pra.ShoeBox(
        room_dim, fs=16000, materials=pra.Material(e_absorption), max_order=max_order
    )

    # マイクの設置
    mic_loc = [1.76, 1.76, 1.6]  # mic1

    # room にマイクを追加します
    room.add_microphone(mic_loc)

    # wavファイルを読み込んで配置します
    from scipy.io import wavfile
    sr, audio1 = wavfile.read('music.wav')
    sr, audio2 = wavfile.read('music.wav')

    #読み込んだ音を分割します
    N = 1024
    audio1_split = np.split(audio1, len(audio1)/N)
    output_data = []
    for x in audio1_split:
        output_data.append(Process(x))

    #print(output_data)

    # 音源ごとに座標情報を与え、`room`に追加していきます。
    # オプションで delay を追加することもできます。
    # 音源の設置
    room.add_source([3.4, 0.5, 0.5], signal=audio1)
    room.add_source([3.4, 2.3, 0.5], signal=audio2)

    fig, ax = room.plot()
    ax.set_xlim([0, 4])
    ax.set_ylim([0, 4])
    ax.set_zlim([0, 3])

    #インパルス応答
    room.plot_rir()
    fig = plt.gcf()
    fig.set_size_inches(20, 10)

    # シミュレーション
    room.simulate()
    recorded = np.array(room.mic_array.signals[0, :])
    rmax = np.max(recorded)
    plt.plot(recorded)

    #シミュレーションした音を書き出す
    writefilename = "./write-audio.wav"
    sf.write(writefilename, recorded / rmax * 0.5, sr)

    plt.show()
