import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from pesq import pesq
from librosa import stft, istft, magphase, resample
from scipy.signal import find_peaks
from scipy.fft import rfft, irfft, fftshift

'''---------------
    パラメータ
---------------'''
c   = 340.29    # 音速 [m/s]
L   = 16000*18
N   = 1024
S   = 512
#st  = 1000      # スタートポイント
#ed  = st + L    # エンドポイント


emb_bin = np.arange(0.05, 1.0, 0.05)
dte_log = []
pesq_log = []

for b in emb_bin:

    st = 1000  # スタートポイント
    ed = st + L  # エンドポイント
    # 埋め込み関連
    Tn  = 100       # トライアル回数
    K   = 40        # 連続して埋め込むフレーム数
    D   = np.floor(N*b).astype(int)  # 埋め込み周波数のビン数
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
    file_name_origin    = 'music1_mono.wav'
    file_name_received1 = 'music1_room_seed1.wav'
    file_name_received2 = 'music1_room_seed1234.wav'
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
    x       = x[st-d:ed-d]

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
        Y1emb[embedded_freq, :] = 0.2 * Y1emb[embedded_freq, :]        # embedded_freqの周波数ビンのみ０に
        Yspec   = Y1emb + Y2spec

        # 音質検査用
        Y1zero[embedded_freq, k:k+3] = 0.2*Y1zero[embedded_freq, k:k+3]

        '''---------------
            5. 埋め込み信号を利用したCSPの計算
        ---------------'''
        # 白色化相互相関(CSP)
        eps         = 1e-20
        XY          = Yspec[:, k+K:k+2*K] * np.conj(Xspec[:, k+K:k+2*K])  # 相互相関(周波数領域)
        XYamp       = np.abs(XY)
        XYamp[XYamp < eps] = eps                # 分母がほぼ0になるのを防止
        CSP_sp      = XY/XYamp                  # 白色化相互相関(周波数領域)
        CSP2         = np.mean(CSP_sp, axis=1)   # 時間方向で平均
        CSP2_ave     = irfft(CSP2, axis=0)        # 逆STFT

        # CSPの整形
        CSP2_ave     = CSP2_ave[:N]               # いらない後半部を除去
        CSP2_ave     = CSP2_ave/np.max(CSP2_ave)   # 最大で割り算

        '''---------------
            5.2 埋め込み信号を利用したCSP(埋込周波数のみ)の計算
        ---------------'''
        CSP2_emb = np.zeros_like(CSP2)
        CSP2_emb[embedded_freq]     = CSP2[embedded_freq]
        CSP2_emb_ave = irfft(CSP2_emb, axis=0)  # 逆STFT

        # CSPの整形
        CSP2_emb_ave = CSP2_emb_ave[:N]  # いらない後半部を除去
        CSP2_emb_ave = CSP2_emb_ave / np.max(CSP2_emb_ave)  # 最大で割り算

        '''---------------
            6. 重み付け差分CSP用の重み計算
        ---------------'''
        # CSPのピーク位置を計算
        pk_csp, _   = find_peaks(CSP1_ave, threshold=0.01)
        # ピーク位置をピークの大きい順にインデックス取得
        st          = np.argsort(-CSP1_ave[pk_csp])
        # CSPの大きい順にD位のピーク位置をピークの大きい順に取得
        pk_csp      = pk_csp[st[:D]]
        # 第１スピーカの遅延推定 (CSPの最大ピーク位置)
        delay1      = pk_csp[0]
        # # 重み (CSPから、CSPの最大ピークを除いた、D-1位のピークのみ抽出したもの)
        # weight      = np.zeros(CSP_ave.size)
        # weight[pk_csp[1:]] = CSP_ave[pk_csp[1:]]

        # 重み
        weight      = np.copy(CSP1_ave)
        weight[delay1-3:delay1+3] = 0    # 推定した第１スピーカのピークを除去
        weight[weight < TH]   = 0
        weight      = weight/np.max(np.abs(weight))  # 正規化


        '''---------------
            7. 重み付け差分CSPによる遅延推定
        ---------------'''
        # CSPの差分
        CSP_sub     = CSP1_ave - CSP2_ave     # CSPの差分
        CSP_sub     = CSP_sub / np.max(CSP_sub) # 正規化

        # 重み付け差分CSP
        CSP_wt      = weight*CSP_sub            # 重み付け埋め込み差分CSP

        '''---------------
            6.2 重み付け差分CSP(埋込周波数のみ)用の重み計算
        ---------------'''
        # CSPのピーク位置を計算
        pk_csp, _ = find_peaks(CSP1_emb_ave, threshold=0.01)
        # ピーク位置をピークの大きい順にインデックス取得
        st = np.argsort(-CSP1_emb_ave[pk_csp])
        # CSPの大きい順にD位のピーク位置をピークの大きい順に取得
        pk_csp = pk_csp[st[:D]]
        # 第１スピーカの遅延推定 (CSPの最大ピーク位置)
        delay1 = pk_csp[0]
        # # 重み (CSPから、CSPの最大ピークを除いた、D-1位のピークのみ抽出したもの)
        # weight      = np.zeros(CSP_ave.size)
        # weight[pk_csp[1:]] = CSP_ave[pk_csp[1:]]

        # 重み
        weight = np.copy(CSP1_emb_ave)
        weight[delay1 - 3:delay1 + 3] = 0  # 推定した第１スピーカのピークを除去
        weight[weight < TH] = 0
        weight = weight / np.max(np.abs(weight))  # 正規化

        '''---------------
            7. 重み付け差分CSP(埋込周波数のみ)による遅延推定
        ---------------'''
        # CSPの差分
        CSP_emb_sub     = CSP1_emb_ave - CSP2_emb_ave     # CSPの差分
        CSP_emb_sub     = CSP_emb_sub / np.max(CSP_emb_sub) # 正規化

        # 重み付け差分CSP
        CSP_emb_wt      = weight*CSP_emb_sub            # 重み付け埋め込み差分CSP

        # 重み付き差分CSPの図示
        # plt.figure()
        # plt.plot(CSP_ave, 'lightgray')
        # plt.plot(CSP_wt)
        # plt.show()

        # ログに溜め込む
        CSP_log.append(CSP1_ave)
        CSP_emb_log.append(CSP2_ave)
        CSP_sub_log.append(CSP_sub)
        CSP_wtd_log.append(CSP_wt)
        CSP1_log.append(CSP1_emb_ave)
        CSP2_log.append(CSP2_emb_ave)
        CSP_emb_sub_log.append(CSP_emb_sub)
        CSP_emb_wtd_log.append(CSP_emb_wt)

    # numpyに変更
    CSP_log     = np.array(CSP_log)         # CSP1
    CSP_emb_log = np.array(CSP_emb_log)     # CSP2
    CSP_sub_log = np.array(CSP_sub_log)     # 差分CSP
    CSP_wtd_log = np.array(CSP_wtd_log)     # 重み付き差分CSP
    CSP1_log     = np.array(CSP1_log)       # CSP1(埋込周波数のみ)
    CSP2_log     = np.array(CSP2_log)       # CSP2(埋込周波数のみ)
    CSP_emb_sub_log = np.array(CSP_emb_sub_log)     # 差分CSP
    CSP_emb_wtd_log = np.array(CSP_emb_wtd_log)     # 重み付き差分CSP

    print('[ マイク・スピーカ距離・到来時間 ]')
    print(' - スピーカ１：{0:.2f}[m] , {1:.2f}[ms]'.format(pos_imp[0]/fs*c, 1000*pos_imp[0]/fs))
    print(' - スピーカ２：{0:.2f}[m] , {1:.2f}[ms]'.format(pos_imp[1]/fs*c, 1000*pos_imp[1]/fs))
    print('')

    '''---------------
        遅延量推定精度
    ---------------'''
    Delay = []
    for csp1, csp2 in zip(CSP1_log, CSP_emb_wtd_log):
        # 遅延量の推定
        csp1_imp = []
        csp1_peaks, _ = find_peaks(csp1, height=0.7)
        csp1_imp.append(csp1_peaks[0])
        #delay1 = np.argmax(csp1)
        delay1 = csp1_imp[0]
        delay2 = np.argmax(csp2)
        # 遅延量をバッファリング
        delay = np.array([delay1, delay2])
        Delay.append(delay)

    Delay = np.array(Delay)

    # 遅延推定誤差 (平均絶対誤差)
    error = []
    for delay in Delay:
        tmp1 = np.sum(np.abs(delay - pos_imp2))
        tmp2 = np.sum(np.abs(np.flip(delay) - pos_imp2))
        error.append(np.min([tmp1, tmp2]))
    error = np.mean(np.array(error))
    delay_time_error = error / fs
    delay_time_error = 1000 * delay_time_error #[ms]に変換
    #print('[ 推定誤差 ]')
    #print('平均到来時間推定誤差: {0:.2f} [ms]'.format(delay_time_error))
    #print('平均距離推定誤差: {0:.2f} [m]'.format(delay_time_error * c))  # 音速掛ける
    #print('')

    dte_log.append(delay_time_error)

    '''---------------
        Peak Ratio (真のピークと第２ピークの比) の計算
    ---------------'''

    PR_log = []
    for csp2, delay in zip(CSP_emb_wtd_log, Delay):
        # まずcsp1が第１スピーカと第２スピーカどちらの遅延を検知したか判定
        if np.abs(delay[0] - pos_imp2[0]) < np.abs(delay[0] - pos_imp2[1]):
            pos_truth = pos_imp2[1]  # csp2はpos_imp[1]を推定したと判定
        else:
            pos_truth = pos_imp2[0]  # csp2はpos_imp[0]を推定したと判定

        # 真の遅延 pos_truth におけるピークの大きさ
        csp2_peak = csp2[pos_truth]

        # それ以外での最大ピーク
        tmp = np.copy(csp2)
        tmp[pos_truth] = 0
        peak_2nd = np.max(tmp)

        PR_log.append(csp2_peak / (np.max([peak_2nd, 10 ** (-8)])))

    PR_log = np.array(PR_log)

    print('[ ピーク比(PR) ]')
    print('  PRが1を超えると正しく検知できる．大きいほどノイズ耐性に頑健になる．')
    print('  PR < 1 のとき、遅延推定に誤りが生じる')
    print(' - 平均PR: {0:.2f}'.format(np.mean(PR_log)))
    print(' - 最小PR: {0:.2f}'.format(np.min(PR_log)))
    print(' - 正しく検知できる確率: {0:.3f}'.format(PR_log[PR_log >= 1].size / PR_log.size))
    print('')

    '''---------------
        音質評価
    ---------------'''
    # 時間波形
    y1_org = istft(Y1spec[:, :Tn * 3], hop_length=S)
    y1_emb = istft(Y1zero[:, :Tn * 3], hop_length=S)
    # PESQ
    y1_org = resample(y1_org, orig_sr=fs, target_sr=16000)
    y1_emb = resample(y1_emb, orig_sr=fs, target_sr=16000)
    score = pesq(16000, y1_org, y1_emb)
    # SNR
    snr = 20 * np.log10(sum(y1_org ** 2) / sum((y1_org - y1_emb) ** 2))
    print('[ 音質 ]')
    print(' - PESQ :  {0:.2f}'.format(score))
    print(' - SNR  :  {0:.2f} [dB]'.format(snr))

    pesq_log.append(score)

dte_log = np.array(dte_log)
print(dte_log)
pesq_log = np.array(pesq_log)
print(pesq_log)

emb_bin = np.arange(5, 100, 5)

'''---------------
    プロット
---------------'''

'''--------------------------------------
 埋込ビン数の変化に伴う推定誤差結果の変化をプロット
--------------------------------------'''
fig = plt.figure(num='埋込ビン数変化', figsize=(6, 3))
plt.subplots_adjust(bottom=0.15)
ax1 = fig.add_subplot(1, 1, 1)

ax1.plot(emb_bin, dte_log,label='Distance Estimation Error')
ax1.set_ylim([-0.05,0.7])
ax1.set_xlabel("Embedding rate of frequency bin [%]", fontname="MS Gothic")
ax1.set_ylabel("距離推定誤差[m]", fontname="MS Gothic")

ax2 = ax1.twinx()
ax2.plot(emb_bin, pesq_log, 'r', label='PESQ')
ax2.set_ylim([-0.05, 5.5])
ax2.set_xlabel("Embedding amplitude", fontname="MS Gothic")
ax2.set_ylabel("PESQ", fontname="MS Gothic")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1+lines2, labels1+labels2, loc='upper left')


plt.savefig('bin.svg')

plt.show()