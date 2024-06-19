'''
async_io.py

The program uses an audio device of your choice to record audio and play it back in real time.
'''

import sounddevice as sd
import numpy as np
import soundfile as sf
import threading
from numpy.fft import rfft, irfft

from device_info import device_info

# 音声の収録・再生
class Audio_Player(threading.Thread):
    def __init__(self, device, blocksize=1024, sr=44100, in_ch=None, out_ch=None):
        super(Audio_Player, self).__init__()

        # パラメータの設定
        self.sr = sr                # サンプリング周波数(Sampling_Rate)
        self.block_n = blocksize    # ブロックサイズ
        self.id = device            # デバイスID (入力，出力)
        # チャンネル数 (入力，出力 デフォルトは2)
        self.in_ch = in_ch or 2     # 入力 in_ch が None あるいは 0 のとき 2 を代入する
        self.out_ch = out_ch or 2   # 出力

    def callback(self, indata, outdata, frames, time, status):

        # 入力の左・右チャンネルを平均
        input_data = np.mean(indata, axis=1)

        # 出力の各チャンネルに処理結果を代入
        for l in range(self.out_ch):
            outdata[:, l] = Process(input_data)

    def run(self):

        # スレッドにイベントを準備
        self.event = threading.Event()

        # スレッドにストリーミング開始
        with sd.Stream(
                        device=self.id,
                        samplerate=self.sr, blocksize=self.block_n,
                        channels=(self.in_ch, self.out_ch),
                        callback=self.callback, finished_callback=self.event.set):
            self.event.wait()   # 待機状態


# 音声に対する処理
def Process(x):
    X = rfft(x)
    Y = 0.8 * X
    y = irfft(Y)
    return y


if __name__ == '__main__':

    # デバイス取得・設定
    dv = sd.query_devices()  # デバイス群

    # ドライバごとのデバイス情報
    drivers, input_device, output_device = device_info()

    # 1. ドライバの選択
    print('\n **** Select Driver ****')
    for i, (name, inltc, outltc) in enumerate(zip(drivers['name'], drivers["in_latency"], drivers["out_latency"])):
        print('{0} : {1}\n   - Input latency : {2:.1f}ms, Output Latency: {3:.1f}ms'.format(i, name, inltc*1000, outltc*1000))
    driver_id = int(input('\n  >> Enter driver ID  '))
    driver = drivers['name'][driver_id]

    # Inputデバイス情報の表示
    print('**** Select Input Device ****')
    info_ = input_device[driver]
    for (id, inputs, ch) in zip(info_['id'], info_['name'], info_['ch']):
        print('{0} {1}   - {2}ch'.format(id, inputs, ch,))
    input_id    = int(input('\n  >> Enter input device ID  '))

    # Outputデバイス情報の表示
    print('**** Select Output Device ****')
    info_ = output_device[driver]
    for (id, outputs, ch) in zip(info_['id'], info_['name'], info_['ch']):
        print('{0} {1}   - {2}ch'.format(id, outputs, ch,))
    #output_id   = input('\n  >> Enter output device ID  (You can select more than two using ",")  ')
    #output_id   = [int(num) for num in output_id.split(',')]
    output_id = int(input('\n  >> Enter output device ID  '))

    print('  Driver : {0}'.format(driver))
    print('  - Input Device : {0}. {1}'.format(input_id, dv[input_id]['name']))
    print('  - Output Device : {0}. {1}'.format(output_id, dv[output_id]['name']))
    #for i in output_id:
    #    print('            {0}. {1}'.format(i, dv[i]['name']))

    device_id = [input_id, output_id]
    ch = [dv[input_id]['max_input_channels'], dv[output_id]['max_output_channels']]

    # 各種パラメータ
    block_size  = 256
    sampling_rate = 48000

    player = Audio_Player(device_id, block_size, sampling_rate, in_ch=ch[0], out_ch=ch[1])
    player.start()

    print("Start")
