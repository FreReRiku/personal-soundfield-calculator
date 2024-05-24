import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from pesq import pesq
from librosa import stft, istft, magphase, resample
from scipy.signal import find_peaks
from scipy.fft import rfft, irfft, fftshift
import scipy.signal as sg

'''---------------
    パラメータ
---------------'''
c   = 340.29    # 音速 [m/s]
L   = 16000*18
N   = 1024
S   = 512
st  = 1000      # スタートポイント
ed  = st + 4*L    # エンドポイント

# 埋め込み関連
Tn  = 100       # トライアル回数
K   = 40        # 連続して埋め込むフレーム数
D   = np.floor(N*0.35).astype(int)  # 埋め込み周波数のビン数
start_frame_pos = np.arange(0, Tn*3, 3)  # スタートのフレーム位置(ここからKフレーム用いる)


# 検知関連
TH  = 0.2       # CSPの最大値に対するノイズと判定する振幅比のしきい値

# 確認用の表示
print('')
print('[ 設定条件 ]')
print(' - ゼロを埋め込む周波数ビンの数：{0}bin/{1}bin中'.format(D, N+1))
print(' - １回の検知で埋め込むフレーム数：{0}フレーム'.format(K))
print(' - 試行回数：{0}回'.format(len(start_frame_pos)))
print('')

'''---------------
    オーディオファイルの読み込み
---------------'''
# ファイル名
file_name_impulse1  = 'impulse_mic1_ch1.wav'
file_name_impulse2  = 'impulse_mic1_ch2.wav'
file_name_origin    = 'music2_mono.wav'
file_name_received1 = 'music2_room_seed1.wav'
file_name_received2 = 'music2_room_seed1234.wav'
# 読み込み
h1, _   = sf.read(file_name_impulse1)
h2, _   = sf.read(file_name_impulse2)
x, _    = sf.read(file_name_origin)
y1, _   = sf.read(file_name_received1)
y2, fs  = sf.read(file_name_received2)
# 時間軸
t = np.arange(N)/fs

# a.単一音声の場合：
#h       = h1
#y       = y1
# b.混合音声の場合：
h  = h1[:2500] + h2[:2500]
#y  = y1 + y2


# 音声のトリミング (長すぎるので)
x_0       = x[st:ed]          # スピーカ出力音声のトリミング
y1_0      = y1[st:ed]         # マイク入力音声1のトリミング
y2_0      = y2[st:ed]         # マイク入力音声1のトリミング

# スペクトログラム
Xspec   = stft(x_0, n_fft=2*N, hop_length=S, win_length=N, center=False)
Y1spec  = stft(y1_0, n_fft=2*N, hop_length=S, win_length=2*N, center=False)
Y2spec  = stft(y2_0, n_fft=2*N, hop_length=S, win_length=2*N, center=False)
Y1zero  = stft(y1, n_fft=2*N, hop_length=S, win_length=2*N, center=False)

# ログ配列
CSP0_log, CSP_log, CSP1_log, CSP2_log, CSP_emb_log, CSP_sub_log, CSP_wtd_log , CSP_emb_sub_log, CSP_emb_wtd_log = [], [], [], [], [], [], [], [], []
outputs = []

# Top Position d の推定
for k in start_frame_pos:

    '''---------------
        0. CSP0
    ---------------'''
    # マイク入力音声のスペクトログラム
    Yspec = Y1spec + Y2spec

    # 白色化相互相関(CSP)：全周波数帯域
    eps = 1e-20
    XY = Yspec[:, k:k + K] * np.conj(Xspec[:, k:k + K])  # 相互相関(周波数領域)
    XYamp = np.abs(XY)  # 相互相関の絶対値(周波数領域)
    XYamp[XYamp < eps] = eps  # 分母がほぼ0になるのを防止
    CSP0_sp = XY / XYamp  # 白色化相互相関(周波数領域)
    CSP0 = np.mean(CSP0_sp, axis=1)  # 時間方向で平均
    CSP0_ave = irfft(CSP0, axis=0)  # 逆STFT

    # CSPの整形
    CSP0_ave = CSP0_ave[:N]  # いらない後半部を除去
    CSP0_ave = CSP0_ave / np.max(CSP0_ave)  # 最大で割り算

    # ログに溜め込む
    CSP0_log.append(CSP0_ave)

    # numpyに変更
    #  CSP0_log     = np.array(CSP0_log)         # CSP0

    # dを推定
    d = (np.argmax(CSP0_log)-25)

#  print(d)

# インパルス、CSP1,2の真のピーク位置
pos_imp = []
pos_imp2 = []
for h_ in [h1, h2]:
    pos_peaks, _ = find_peaks(h_, height=0.6)
    pos_imp.append(pos_peaks[0])
    pos_imp2.append(pos_peaks[0]-d)

pos_imp = np.array(pos_imp)
pos_imp2 = np.array(pos_imp2)

# 音声のトリミング (長すぎるので)
#x       = x[st-d:ed-d]

# スペクトログラム
Xspec   = stft(x, n_fft=2*N, hop_length=S, win_length=N, center=False)

for k in start_frame_pos:
    
    # 埋め込み用の配列
    Y1emb   = np.copy(Y1spec)
    Y2emb   = np.copy(Y2spec)
    
    '''---------------
        1. CSP1
    ---------------'''
    # マイク入力音声のスペクトログラム
    Yspec       = Y1spec + Y2spec

    # 白色化相互相関(CSP)：全周波数帯域
    eps         = 1e-20
    XY          = Yspec[:, k:k+K] * np.conj(Xspec[:, k:k+K])    # 相互相関(周波数領域)
    XYamp       = np.abs(XY)                # 相互相関の絶対値(周波数領域)
    XYamp[XYamp < eps] = eps                # 分母がほぼ0になるのを防止
    CSP_sp      = XY/XYamp                  # 白色化相互相関(周波数領域)
    CSP1         = np.mean(CSP_sp, axis=1)   # 時間方向で平均
    CSP1_ave     = irfft(CSP1, axis=0)        # 逆STFT

    # CSPの整形
    CSP1_ave     = CSP1_ave[:N]               # いらない後半部を除去
    CSP1_ave     = CSP1_ave/np.max(CSP1_ave)   # 最大で割り算

    '''---------------
        2. ゼロ埋め込み周波数の決定
    ---------------'''
    # CSPの周波数特性から，振幅の大きい順にD個の周波数を検知
    pos         = np.argsort(-np.abs(CSP1))  # 周波数の大きい順にインデックスを取得
    embedded_freq   = pos[:D]               # CSPの最大D個の周波数

    # 埋め込み位置の確認
    # plt.figure()
    # plt.plot(np.abs(CSP))
    # plt.scatter(embedded_freq,np.abs(CSP[embedded_freq]))

    '''---------------
        3  CSP1(埋込周波数のみ)の計算
    ---------------'''
    CSP1_emb = np.zeros_like(CSP1)
    CSP1_emb[embedded_freq]     = CSP1[embedded_freq]
    CSP1_emb_ave     = irfft(CSP1_emb, axis=0)        # 逆STFT

    # CSPの整形
    CSP1_emb_ave     = CSP1_emb_ave[:N]               # いらない後半部を除去
    CSP1_emb_ave     = CSP1_emb_ave/np.max(CSP1_emb_ave)   # 最大で割り算


    '''---------------
        4. ゼロ埋め込み
    ---------------'''
    # Y1 に対してゼロを埋め込み
    Y1emb[embedded_freq, :] = 0.3*Y1emb[embedded_freq, :]        # embedded_freqの周波数ビンのみ０に
    Yspec   = Y1emb + Y2spec

    # 音質検査用
    Y1zero[embedded_freq, k:k+3] = 0.2*Y1zero[embedded_freq, k:k+3]

    
    # ログに溜め込む
    CSP_log.append(CSP1_ave)
#CSP_emb_log.append(CSP2_ave)
    #CSP_sub_log.append(CSP_sub)
    #CSP_wtd_log.append(CSP_wt)
    CSP1_log.append(CSP1_emb_ave)
 #   CSP2_log.append(CSP2_emb_ave)
  #  CSP_emb_sub_log.append(CSP_emb_sub)
   # CSP_emb_wtd_log.append(CSP_emb_wt)

# numpyに変更
CSP_log     = np.array(CSP_log)         # CSP1
CSP_emb_log = np.array(CSP_emb_log)     # CSP2
CSP_sub_log = np.array(CSP_sub_log)     # 差分CSP
CSP_wtd_log = np.array(CSP_wtd_log)     # 重み付き差分CSP
CSP1_log     = np.array(CSP1_log)       # CSP1(埋込周波数のみ)
CSP2_log     = np.array(CSP2_log)       # CSP2(埋込周波数のみ)
CSP_emb_sub_log = np.array(CSP_emb_sub_log)     # 差分CSP
CSP_emb_wtd_log = np.array(CSP_emb_wtd_log)     # 重み付き差分CSP

_, y = sg.istft(Yspec,fs=fs,nperseg=None)
y = np.real(y)
outputs.append(y)

'''---------------
        ファイル保存
    ---------------'''
#file_base = os.path.splitext(file_name)[0]
new_file_name = 'music2_mono_embedded.wav'
sf.write(new_file_name, y * 0.95 / np.max(abs(y)), fs)


