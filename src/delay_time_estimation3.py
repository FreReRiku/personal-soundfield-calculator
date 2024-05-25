'''
delay_time_estimation2.py

What can we find out in this program?

Calculate
- the distance between the microphone and speaker and the arrival time
- the estimation error
- peak ratio(PR)
- sound quality (PESQ, SNR)

Make graph
- CSP1(without embedding)
- CSP2(with embedding)
- Weighted Difference-CSP
- Impulse
'''

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from librosa import stft, magphase, display
from scipy.signal import istft, find_peaks
from scipy.fft import rfft, irfft, fftshift

'''---------------
    パラメータ
---------------'''
c   = 340.29    # 音速 [m/s]
L   = 16000*10
N   = 1024
S   = 512
st  = 2000      # スタートポイント
ed  = st + L    # エンドポイント
start_frame_pos = np.arange(0,200,2) # スタートのフレーム位置(ここからKフレーム用いる)

# 埋め込み関連
K   = 10        # 連続して埋め込むフレーム数
D   = np.floor(N*0.05).astype(int) # 埋め込み周波数のビン数

# 検知関連
TH  = 0.2       # CSPの最大値に対するノイズと判定する振幅比のしきい値

# 確認用の表示
print('')
print('[ 設定条件 ]')
print(' - ゼロを埋め込む周波数ビンの数：{0}bin/{1}bin中'.format(D,N+1))
print(' - １回の検知で埋め込むフレーム数：{0}フレーム'.format(K))
print(' - 試行回数：{0}回'.format(len(start_frame_pos)))
print('')

'''---------------
    オーディオファイルの読み込み
---------------'''
# ファイル名
file_name_impulse1  = '../wav_data/impulse_mic1_ch1.wav'
file_name_impulse2  = '../wav_data/impulse_mic1_ch2.wav'
file_name_origin    = '../wav_data/music1_mono.wav'
file_name_received1 = '../wav_data/music1_room_seed1.wav'
file_name_received2 = '../wav_data/music1_room_seed1234.wav'
# 読み込み
h1, _   = sf.read(file_name_impulse1)
h2, _   = sf.read(file_name_impulse2)
x, _    = sf.read(file_name_origin)
y1, _   = sf.read(file_name_received1)
y2, fs  = sf.read(file_name_received2)
# 時間軸
t = np.arange(N)/fs

# インパルスの真のピーク位置
pos_imp = []
for h_ in [h1, h2]:
    pos_peaks, _ = find_peaks(h_, height=0.6)
    pos_imp.append(pos_peaks[0])

pos_imp = np.array(pos_imp)

print('[ マイク・スピーカ距離・到来時間 ]')
print(' - スピーカ１：{0:.2f}[m] , {1:.2f}[ms]'.format(pos_imp[0]/fs*c, 1000*pos_imp[0]/fs))
print(' - スピーカ２：{0:.2f}[m] , {1:.2f}[ms]'.format(pos_imp[1]/fs*c, 1000*pos_imp[1]/fs))
print('')

# a.単一音声の場合：
#h       = h1
#y       = y1
# b.混合音声の場合：
h  = h1[:2500] + h2[:2500]
#y  = y1 + y2


# 音声のトリミング (長すぎるので)
x       = x[st:ed]          # スピーカ出力音声のトリミング
y1      = y1[st:ed]         # マイク入力音声1のトリミング
y2      = y2[st:ed]         # マイク入力音声1のトリミング

# スペクトログラム
Xspec   = stft(x, n_fft=2*N, hop_length=S, win_length=N, center=False)
Y1spec  = stft(y1, n_fft=2*N, hop_length=S, win_length=2*N, center=False)
Y2spec  = stft(y2, n_fft=2*N, hop_length=S, win_length=2*N, center=False)

# ログ配列
CSP_log, CSP_emb_log, CSP_sub_log, CSP_wtd_log = [], [], [], []

for k in start_frame_pos:
    
    # 埋め込み用の配列
    Y1emb   = np.copy(Y1spec)
    Y2emb   = np.copy(Y2spec)
    
    '''---------------
        1. 残響推定
    ---------------'''
    # マイク入力音声のスペクトログラム
    Yspec       = Y1spec + Y2spec

    # 白色化相互相関(CSP)：全周波数帯域
    eps         = 1e-20
    XY          = Yspec[:,k:k+K] * np.conj(Xspec[:,k:k+K])    # 相互相関(周波数領域)
    XYamp       = np.abs(XY)                # 相互相関の絶対値(周波数領域)
    XYamp[XYamp < eps] = eps                # 分母がほぼ0になるのを防止
    CSP_sp      = XY/XYamp                  # 白色化相互相関(周波数領域)
    CSP         = np.mean(CSP_sp, axis=1)   # 時間方向で平均
    CSP_ave     = irfft(CSP, axis=0)        # 逆STFT

    # CSPの整形
    CSP_ave     = CSP_ave[:N]               # いらない後半部を除去
    CSP_ave     = CSP_ave/np.max(CSP_ave)   # 最大で割り算

    '''---------------
        2. ゼロ埋め込み周波数の決定
    ---------------'''
    # CSPの周波数特性から，振幅の大きい順にD個の周波数を検知
    pos         = np.argsort(-np.abs(CSP))  # 周波数の大きい順にインデックスを取得
    embedded_freq   = pos[:D]               # CSPの最大D個の周波数

    # 埋め込み位置の確認
    # plt.figure()
    # plt.plot(np.abs(CSP))
    # plt.scatter(embedded_freq,np.abs(CSP[embedded_freq]))

    '''---------------
        3. ゼロ埋め込み
    ---------------'''
    # Y1 に対してゼロを埋め込み
    Y1emb[embedded_freq, :] = 0        # embedded_freqの周波数ビンのみ０に
    Yspec   = Y1emb + Y2spec

    '''---------------
        4. 埋め込み信号を利用したCSPの計算
    ---------------'''
    # 白色化相互相関(CSP)
    eps         = 1e-20
    XY          = Yspec[:,k+K:k+2*K] * np.conj(Xspec[:,k+K:k+2*K]) # 相互相関(周波数領域)
    XYamp       = np.abs(XY)
    XYamp[XYamp < eps] = eps                # 分母がほぼ0になるのを防止
    CSP_sp      = XY/XYamp                  # 白色化相互相関(周波数領域)
    CSP         = np.mean(CSP_sp, axis=1)   # 時間方向で平均
    CSP_emb     = irfft(CSP, axis=0)        # 逆STFT
    
    # CSPの整形
    CSP_emb     = CSP_emb[:N]               # いらない後半部を除去
    CSP_emb     = CSP_emb/np.max(CSP_emb)   # 最大で割り算

    '''---------------
        4. 重み付け差分CSP用の重み計算
    ---------------'''
    # CSPのピーク位置を計算
    pk_csp, _   = find_peaks(CSP_ave, threshold=0.01)
    # ピーク位置をピークの大きい順にインデックス取得
    st          = np.argsort(-CSP_ave[pk_csp])
    # CSPの大きい順にD位のピーク位置をピークの大きい順に取得
    pk_csp      = pk_csp[st[:D]]
    # 第１スピーカの遅延推定 (CSPの最大ピーク位置)
    delay1      = pk_csp[0]
    # # 重み (CSPから、CSPの最大ピークを除いた、D-1位のピークのみ抽出したもの)
    # weight      = np.zeros(CSP_ave.size)
    # weight[pk_csp[1:]] = CSP_ave[pk_csp[1:]]
    
    # 重み
    weight      = np.copy(CSP_ave)
    weight[delay1-3:delay1+3] = 0    # 推定した第１スピーカのピークを除去
    weight[weight<TH]   = 0
    
    
    '''---------------
        5. 重み付け差分CSPによる遅延推定
    ---------------'''
    # CSPの差分
    CSP_sub     = CSP_ave - CSP_emb     # CSPの差分
    #CSP_sub[CSP_sub>CSP_ave] = CSP_ave[CSP_sub>CSP_ave]

    # 重み付け差分CSP
    CSP_wt = weight*CSP_sub            # 重み付け埋め込み差分CSP
    
    # 重み付き差分CSPの図示
    # plt.figure()
    # plt.plot(CSP_ave, 'lightgray')
    # plt.plot(CSP_wt)
    # plt.show()
    
    # ログに溜め込む
    CSP_log.append(CSP_ave)
    CSP_emb_log.append(CSP_emb)
    CSP_sub_log.append(CSP_sub)
    CSP_wtd_log.append(CSP_wt)

# numpyに変更
CSP_log     = np.array(CSP_log)         # CSP1
CSP_emb_log = np.array(CSP_emb_log)     # CSP2
CSP_sub_log = np.array(CSP_sub_log)     # 差分CSP
CSP_wtd_log = np.array(CSP_wtd_log)     # 重み付き差分CSP

'''---------------
    遅延量推定精度
---------------'''
Delay = []
for csp1, csp2 in zip(CSP_log, CSP_sub_log):
    
    # 遅延量の推定
    delay1 = np.argmax(csp1)
    delay2 = np.argmax(csp2)
    # 遅延量をバッファリング
    delay = np.array([delay1, delay2])
    Delay.append(delay)
    
Delay = np.array(Delay)

# 遅延推定誤差 (平均絶対誤差)
error = []
for delay in Delay:
    tmp1 = np.sum(np.abs(delay - pos_imp ))
    tmp2 = np.sum(np.abs(np.flip(delay) - pos_imp ))
    error.append(np.min([tmp1, tmp2]))
error = np.mean(np.array(error))
delay_time_error = error/fs
print('[ 推定誤差 ]')
print('平均到来時間推定誤差: {0:.2f} [ms]'.format(100*delay_time_error))
print('平均距離推定誤差: {0:.2f} [m]'.format(delay_time_error*c)) # 音速掛ける
print('')

'''---------------
    Peak Ratio (真のピークと第２ピークの比) の計算
---------------'''
PR_log = []
for csp2, delay in zip(CSP_wtd_log, Delay):
    # まずcsp1が第１スピーカと第２スピーカどちらの遅延を検知したか判定
    if np.abs(delay[0]-pos_imp[0]) < np.abs(delay[0]-pos_imp[1]):
        pos_truth = pos_imp[1]         # csp2はpos_imp[1]を推定したと判定
    else:
        pos_truth = pos_imp[0]         # csp2はpos_imp[0]を推定したと判定
        
    # 真の遅延 pos_truth におけるピークの大きさ
    csp2_peak   = csp2[pos_truth]
    
    # それ以外での最大ピーク
    tmp         = np.copy(csp2)
    tmp[pos_truth] = 0 
    peak_2nd    = np.max(tmp)

    PR_log.append(csp2_peak/(np.max([peak_2nd,10**(-8)])))
    
PR_log = np.array(PR_log)

print('[ ピーク比(PR) ]')
print('  PRが1を超えると正しく検知できる．大きいほどノイズ耐性に頑健になる．')
print('  PR < 1 のとき、遅延推定に誤りが生じる')
print(' - 平均PR: {0:.2f}'.format(np.mean(PR_log)))
print(' - 最小PR: {0:.2f}'.format(np.min(PR_log)))
print(' - 正しく検知できる確率: {0:.3f}'.format(PR_log[PR_log>=1].size/PR_log.size))


'''---------------
    プロット
---------------'''

#-------------------------
# インパルス応答プロット
#-------------------------
fig = plt.figure(num='Impulse')
for p, c in zip(pos_imp, ['r', 'g']):
    plt.axvline(p/fs, color=c, linestyle='--')
plt.plot(t, h[:N])

plt.savefig('impulse.svg')

#-------------------------
# 平均CSPのプロット
#-------------------------
fig = plt.figure(num='CSP',figsize=(6,6))
plt.subplots_adjust(wspace=0.4, hspace=0.8)
ax1 = fig.add_subplot(3, 1, 1)
ax2 = fig.add_subplot(3, 1, 2)
ax3 = fig.add_subplot(3, 1, 3)

# 各計算結果をプロット
CSPs_ave    = np.mean(CSP_log[:,:N],axis=0)
CSPs_emb    = np.mean(CSP_emb_log[:,:N],axis=0)
CSPs_sub    = np.mean(CSP_sub_log[:,:N],axis=0)
CSPs_wtd    = np.mean(CSP_wtd_log[:,:N],axis=0)
for p, c in zip(pos_imp, ['r', 'g']):
    for ax in [ax1, ax2, ax3]:
        ax.axvline(p/fs, color=c, linestyle='--')   
    for ax in [ax1, ax2]:
        ax.axhline(TH, color='k', linestyle=':')   
ax1.plot(t, CSPs_ave)
ax2.plot(t, CSPs_emb)
ax3.plot(t, CSPs_sub, 'lightgray')
ax3.plot(t, CSPs_wtd, 'r')

# 各subplotにラベルを追加
ax1.set_xlabel("Time [s]", fontname="Arial")
ax1.set_ylabel("Amplitude")
ax1.set_title('CSP1 (without embedding)')
ax1.set_xlim([t[0], t[-1]])

ax2.set_xlabel("Time [s]", fontname="Arial")
ax2.set_ylabel("Amplitude")
ax2.set_title("CSP2 (with embedding)")
ax2.set_xlim([t[0], t[-1]])

ax3.set_xlabel("Time [s]", fontname="Arial")
ax3.set_ylabel("Amplitude")
ax3.set_title("Weighted Difference-CSP")
ax3.set_xlim([t[0], t[-1]])
_, y_max = ax3.get_ylim()
ax3.set_ylim([0, y_max])

plt.savefig('CSP.svg')

plt.show()
