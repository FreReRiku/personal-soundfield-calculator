# async_io.md について

## どんなプログラム?

このプログラムは実行すると、入力デバイスと出力デバイスを選択するよう表示されます。
ユーザーが選択したデバイスを使用して、音声を収録し、簡単な信号処理を行った後、その音声をリアルタイムで再生されます。

## 使用するライブラリ

- `sounddevice`: 音声の入出力を制御するライブラリ。
- `numpy`: 数値計算をサポートするライブラリ。
- `soundfile`: 音声ファイルの読み書きに使用するライブラリ。
- `threading`: スレッド処理を行うための標準ライブラリ。
- `rfft`, `irfft`: 高速フーリエ変換と逆高速フーリエ変換を行う関数。
- `device_info`: デバイス情報を取得するモジュール(/src内にあるpythonファイル。)

## クラス `Audio_Player`

音声を収録・再生するクラスで、スレッドを継承しています。

```python
class Audio_Player(threading.Thread):
    def __init__(self, device, blocksize=1024, sr=44100, in_ch=None, out_ch=None):
        super(Audio_Player, self).__init__()

        self.sr = sr
        self.block_n = blocksize
        self.id = device
        self.in_ch = in_ch or 2
        self.out_ch = out_ch or 2
```

`__init__`メソッドはオブジェクトの初期化を行います。

- `device`: デバイスID（入力と出力のペア）です。
- `blocksize`: ブロックサイズで、デフォルトは1024。
- `sr`: サンプリングレートで、デフォルトは44100[Hz]
- `in_ch`: 入力のチャンネル数。指定がなければ2。
- `out_ch`: 出力のチャンネル数。指定がなければ2。

```python
def callback(self, indata, outdata, frames, time, status):
    input_data = np.mean(in_data, axis=1)
    for i in range(self.out_ch):
        outdata[:, l] = Process(input_data)
```

`callback`メソッドは、ストリームの各ブロック毎に呼び出されるコールバック関数です。

- `indata`は入力データを表しています。
- `outdata`は出力データを表しています。
- `indata`の左右チャンネルを平均してモノラルに変換します。
- 平均したデータを`Process`関数で処理し、出力データの各チャンネルに設定します。

```python
def run(self):
    self.event = threading.Event()
    with sd.Stream(
        device = self.id,
        samplerate = self.sr,
        blocksize = self.block_n,
        channels = (self.in_ch, self.out_ch),
        callback = self.callback,
        finished_callback = self.event.set):
        serf.event.wait()
```
`run`メソッドは、スレッドが開始されると呼び出されます。

ストリーミングを開始し、終了するまで待機します。

## 関数`Process` : 音声に対する処理

```python
def Process(x):
    X = rfft(x)
    Y = 0.8 * X
    y = irfft(Y)
    return y
```

- 音声データ`x`に対して、高速フーリエ変換を行い、周波数成分`X`を取得します。
- その成分を0.8倍して、`Y`とし、逆高速フーリエ変換で時系列データ`y`を得て返します。
- 音声信号の振幅を0.8倍にしています。

## メイン処理

```python
if __name__ == '__main__':
    dv = sd.query_devices()
    drivers, input_device, output_device = device_info()
```

- 利用可能なデバイスを取得します。
- ドライバ毎のデバイス情報を取得します。

```python
print('\n **** Select Driver ****')
for i, (name, inltc, outltc) in enumerate(zip(drivers['name'], drivers["in_latency"], drivers["out_latency"])):
    print('{0} : {1}\n   - Input latency : {2:.1f}ms, Output Latency: {3:.1f}ms'.format(i, name, inltc*1000, outltc*1000))
driver_id = int(input('\n  >> Enter driver ID  '))
driver = drivers['name'][driver_id]
```

- ドライバを選択し、そのIDを入力させます。

```python
print('**** Select Input Device ****')
info_ = input_device[driver]
for (id, inputs, ch) in zip(info_['id'], info_['name'], info_['ch']):
    print('{0} {1}   - {2}ch'.format(id, inputs, ch,))
input_id = int(input('\n  >> Enter input device ID  '))
```

- 入力デバイスを選択し、そのIDを入力させます。

```python
print('**** Select Output Device ****')
info_ = output_device[driver]
for (id, outputs, ch) in zip(info_['id'], info_['name'], info_['ch']):
    print('{0} {1}   - {2}ch'.format(id, outputs, ch,))
output_id = int(input('\n  >> Enter output device ID  '))
```

- 出力デバイスを選択し、そのIDを入力させます。

```python
print('  Driver : {0}'.format(driver))
print('  - Input Device : {0}. {1}'.format(input_id, dv[input_id]['name']))
print('  - Output Device : {0}. {1}'.format(output_id, dv[output_id]['name']))
device_id = [input_id, output_id]
ch = [dv[input_id]['max_input_channels'], dv[output_id]['max_output_channels']]
block_size = 256
sampling_rate = 48000
player = Audio_Player(device_id, block_size, sampling_rate, in_ch=ch[0], out_ch=ch[1])
player.start()
print("Start")
```

- デバイスとチャンネル数を設定し、`Audio_Player`インスタンスを生成してスレッドを開始します。

全体として、このプログラムはユーザーが選択したオーディオデバイスを使用して、音声を収録し、振幅を0.8倍するという簡単な信号処理を行い、その音声をリアルタイムで再生するものです。
