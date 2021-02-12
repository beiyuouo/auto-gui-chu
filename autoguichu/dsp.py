# Author: BeiYu
# Github: https://github.com/beiyuouo
# Date  : 2021/2/11 19:07
# Description:
import os

import cv2
import librosa
import numpy as np
import logging
import wave
import matplotlib.pyplot as plt
import scipy
from scipy.io import wavfile
from .arrop import *
from . import imgop
from .utils import *

piano = np.array([0, 27.5, 29.1, 30.9, 32.7, 34.6, 36.7, 38.9, 41.2,
                  43.7, 46.2, 49.0, 51.9, 55.0, 58.3, 61.7, 65.4, 69.3,
                  73.4, 77.8, 82.4, 87.3, 92.5, 98.0, 103.8, 110.0, 116.5,
                  123.5, 130.8, 138.6, 146.8, 155.6, 164.8, 174.6, 185.0,
                  196.0, 207.7, 220.0, 233.1, 246.9, 261.6, 277.2, 293.7,
                  311.1, 329.6, 349.2, 370.0, 392.0, 415.3, 440.0, 466.2,
                  493.9, 523.3, 554.4, 587.3, 622.3, 659.3, 698.5, 740.0,
                  784.0, 830.6, 880.0, 932.3, 987.8, 1047, 1109, 1175, 1245,
                  1319, 1397, 1480, 1568, 1661, 1760, 1865, 1976, 2093, 2217,
                  2349, 2489, 2637, 2794, 2960, 3136, 3322, 3520, 3729, 3951, 4186, 4400])

piano_10 = np.array([0, 73.4, 207.7, 349.2, 587.3, 987.8, 1245, 1661, 2093, 2794, 4400])


def filter(audio, fc=[], fs=44100, win_length=4096):
    for i in range(len(audio) // win_length):
        audio[i * win_length:(i + 1) * win_length] = fft_filter(audio[i * win_length:(i + 1) * win_length], fs=fs,
                                                                fc=fc)

    return audio


def freq_correct(src, dst=0, fs=44100, fc=3000, mode='normal', win_length=1024, alpha=1, srcfreq=0, dstfreq=0):
    out = np.zeros_like(src)
    try:
        if mode == 'normal':
            if srcfreq == 0 and dstfreq == 0:
                src_oct = librosa.hz_to_octs(base_freq(src, fs, fc))
                dst_oct = librosa.hz_to_octs(base_freq(dst, fs, fc))
            else:
                src_oct = librosa.hz_to_octs(srcfreq)
                dst_oct = librosa.hz_to_octs(dstfreq)
            offset_step = (dst_oct - src_oct) * 12 * alpha
            out = librosa.effects.pitch_shift(src, fs, n_steps=offset_step)
        elif mode == 'track':
            length = min([len(src), len(dst)])
            for i in range(length // win_length):
                src_oct = librosa.hz_to_octs(base_freq(src[i * win_length:(i + 1) * win_length], fs, fc))
                dst_oct = librosa.hz_to_octs(base_freq(dst[i * win_length:(i + 1) * win_length], fs, fc))

                offset_step = (dst_oct - src_oct) * 12 * alpha
                out[i * win_length:(i + 1) * win_length] = librosa.effects.pitch_shift(
                    src[i * win_length:(i + 1) * win_length], 44100, n_steps=offset_step)
        return out
    except Exception as e:
        return src

    # print('freqloss:',round((basefreq(out, 44100,3000)-basefreq(src, 44100,3000))/basefreq(src, 44100,3000),3))


def energy_correct(src, dst, mode='normal', win_length=512, alpha=1):
    """
    mode: normal | track
    """
    out = np.zeros_like(src)
    if mode == 'normal':
        src_rms = rms(src)
        dst_rms = rms(dst)
        out = src * (dst_rms / src_rms) * alpha
    elif mode == 'track':
        length = min([len(src), len(dst)])
        tracks = []
        for i in range(length // win_length):
            src_rms = np.clip(rms(src[i * win_length:(i + 1) * win_length]), 1e-6, np.inf)
            dst_rms = rms(dst[i * win_length:(i + 1) * win_length])
            tracks.append((dst_rms / src_rms) * alpha)
        tracks = np.clip(np.array(tracks), 0.1, 10)
        tracks = interp(tracks, length)
        out = src * tracks

    return np.clip(out, -32760, 32760)


def time_correct(src, dst=0, _min=0.25, out_time=0, fs=44100):
    src_length = len(src)
    if out_time == 0:
        dst_length = len(dst)
    else:
        dst_length = int(out_time * fs)
    rate = np.clip(src_length / dst_length, _min, 100)
    out = librosa.effects.time_stretch(src, rate)
    return out


def highlight_bass(src, srcfreq, contrastfreq):
    if srcfreq < contrastfreq:
        src = src / srcfreq * contrastfreq
    return src


def freq_features(signal, fs):
    signal = normliaze(signal, mode='5_95', truncated=100)
    signal_fft = np.abs(scipy.fftpack.fft(signal))
    length = len(signal)
    features = []
    for i in range(len(piano_10) - 1):
        k1 = int(length / fs * piano_10[i])
        k2 = int(length / fs * piano_10[i + 1])
        features.append(np.mean(signal_fft[k1:k2]))
    return np.array(features)


def get_info(path):
    logging.debug(f'reading file {path}')
    f = wave.open(path, "rb")
    # getparams() 一次性返回所有的WAV文件的格式信息
    params = f.getparams()
    # nframes 采样点数目
    nchannels, sampwidth, framerate, nframes = params[:4]
    # readframes() 按照采样点读取数据
    # str_data = f.readframes(nframes)  # str_data 是二进制字符串
    info = {'nchannels': nchannels, 'sampwidth': sampwidth,
            'framerate': framerate, 'nframes': nframes}

    f.close()
    return info


def read_wave(path):
    sampling_freq, audio = wavfile.read(path)
    return audio[:, 0]


def rms(signal):
    signal = signal.astype('float64')
    return np.mean((signal * signal)) ** 0.5


def get_energy(signal, kernel_size: int, stride: int, padding: int = 0):
    signal = np.array(signal)

    signal = pad(signal, padding)
    out_len = int((len(signal) + 1 - kernel_size) / stride)
    energy = np.zeros(out_len)
    for i in range(out_len):
        energy[i] = rms(signal[i * stride:i * stride + kernel_size])
    return energy


def find_peak(indata, ismax=False, interval=10, threshold=0.1, reverse=False):
    indexs = []
    if reverse:
        indata = -1 * indata
        indexs = [0, len(indata) - 1]
    else:
        indata = np.clip(indata, np.max(indata) * threshold, np.max(indata))
    diff = diff1d(indata)
    if ismax:
        return np.array([np.argmax(indata)])

    rise = True
    if diff[0] <= 0:
        rise = False
    for i in range(len(diff)):
        if rise is True and diff[i] <= 0:
            index = i
            ok_flag = True
            for x in range(interval):
                if indata[np.clip(index - x, 0, len(indata) - 1)] > indata[index] \
                        or indata[np.clip(index + x, 0, len(indata) - 1)] > indata[index]:
                    ok_flag = False
            if ok_flag:
                indexs.append(index)

        if diff[i] <= 0:
            rise = False
        else:
            rise = True

    return np.sort(np.array(indexs))


def show_peak(indexs, energy):
    y = get_y(indexs, energy)
    plt.plot(energy)
    plt.scatter(indexs, y, c='orange')
    plt.show()


def fft_filter(signal, fs: int, fc: list = None, type: str = 'bandpass'):
    """
    signal: Signal
    fs: Sampling frequency
    fc: [fc1,fc2...] Cut-off frequency
    type: bandpass | bandstop
    """
    if fc is None:
        fc = []
    k = []
    N = len(signal)  # get N

    for i in range(len(fc)):
        k.append(int(fc[i] * N / fs))

    # FFT
    signal_fft = scipy.fftpack.fft(signal)

    # Frequency truncation
    if type == 'bandpass':
        a = np.zeros(N)
        for i in range(int(len(fc) / 2)):
            a[k[2 * i]:k[2 * i + 1]] = 1
            a[N - k[2 * i + 1]:N - k[2 * i]] = 1
    elif type == 'bandstop':
        a = np.ones(N)
        for i in range(int(len(fc) / 2)):
            a[k[2 * i]:k[2 * i + 1]] = 0
            a[N - k[2 * i + 1]:N - k[2 * i]] = 0
    signal_fft = a * signal_fft
    signal_ifft = scipy.fftpack.ifft(signal_fft)
    result = signal_ifft.real
    return result


def sin(f, fs, time, theta=0):
    x = np.linspace(0, 2 * np.pi * f * time, int(fs * time))
    return np.sin(x + theta)


def wave(f, fs, time, mode='sin'):
    f, fs, time = float(f), float(fs), float(time)
    if mode == 'sin':
        return sin(f, fs, time, theta=0)
    elif mode == 'square':
        half_T_num = int(time * f) * 2 + 1
        half_T_point = int(fs / f / 2)
        x = np.zeros(int(fs * time) + 2 * half_T_point)
        for i in range(half_T_num):
            if i % 2 == 0:
                x[i * half_T_point:(i + 1) * half_T_point] = -1
            else:
                x[i * half_T_point:(i + 1) * half_T_point] = 1
        return x[:int(fs * time)]
    elif mode == 'triangle':
        half_T_num = int(time * f) * 2 + 1
        half_T_point = int(fs / f / 2)
        up = np.linspace(-1, 1, half_T_point)
        down = np.linspace(1, -1, half_T_point)
        x = np.zeros(int(fs * time) + 2 * half_T_point)
        for i in range(half_T_num):
            if i % 2 == 0:
                x[i * half_T_point:(i + 1) * half_T_point] = up.copy()
            else:
                x[i * half_T_point:(i + 1) * half_T_point] = down.copy()
        return x[:int(fs * time)]


def down_sample(signal, fs1=0, fs2=0, alpha=0, mod='just_down'):
    if alpha == 0:
        alpha = int(fs1 / fs2)
    if mod == 'just_down':
        return signal[::alpha]
    elif mod == 'avg':
        result = np.zeros(int(len(signal) / alpha))
        for i in range(int(len(signal) / alpha)):
            result[i] = np.mean(signal[i * alpha:(i + 1) * alpha])
        return result


def base_freq(signal, fs, fc=0, mode='centroid'):
    if fc == 0:
        kc = int(len(signal) / 2)
    else:
        kc = int(len(signal) / fs * fc)
    length = len(signal)
    signal_fft = np.abs(scipy.fftpack.fft(signal))[:kc]
    # centroid
    _sum = np.sum(signal_fft) / 2
    tmp_sum = 0
    for i in range(kc):
        tmp_sum += signal_fft[i]
        if tmp_sum > _sum:
            centroid_freq = i / (length / fs)
            break
    # print(centroid_freq)
    # max
    max_freq = np.argwhere(signal_fft == np.max(signal_fft))[0][0] / (length / fs)
    # print(max_freq)
    if mode == 'centroid':
        return centroid_freq
    elif mode == 'max':
        return max_freq
    elif mode == 'mean':
        return (centroid_freq + max_freq) / 2


def show_freq(signal, fs, fc=0):
    """
    return f,fft
    """
    if fc == 0:
        kc = int(len(signal) / 2)
    else:
        kc = int(len(signal) / fs * fc)
    signal_fft = np.abs(scipy.fftpack.fft(signal))
    f = np.linspace(0, fs / 2, num=int(len(signal_fft) / 2))
    return f[:kc], signal_fft[0:int(len(signal_fft) / 2)][:kc]


def medfilt(signal, x):
    return scipy.signal.medfilt(signal, x)


def cleanoffset(signal):
    return signal - np.mean(signal)


def bpf(signal, fs, fc1, fc2, numtaps=3, mode='iir'):
    if mode == 'iir':
        b, a = scipy.signal.iirfilter(numtaps, [fc1, fc2], fs=fs)
    elif mode == 'fir':
        b = scipy.signal.firwin(numtaps, [fc1, fc2], pass_zero=False, fs=fs)
        a = 1
    return scipy.signal.lfilter(b, a, signal)


def peak_match(audios, music, args):
    dst_features = []
    dst_syllables = []
    dst_indexs = []
    for audio in audios:
        energy = get_energy(audio, 4410, 441, 4410)
        indexs = find_peak(energy, ismax=True, interval=args.viv)
        dst_indexs.append(indexs[0])
        syllable = crop(audio, indexs * 441, int(44100 * 0.2))[0]
        dst_syllables.append(syllable)
        dst_features.append(freq_features(syllable, 44100))
    logging.debug('dst init ok')

    if args.debug:
        endtime = 20 * args.sf
    else:
        endtime = int(len(music) / args.sf)

    music = music[args.sf * 0: args.sf * endtime]
    musicH = fft_filter(music, args.sf, [args.cut_off_low_freq, args.cut_off_high_freq])
    energy = get_energy(musicH, 4410, 441, 4410)
    src_indexs = find_peak(energy, interval=int(args.aiv))
    src_syllables = crop(musicH, src_indexs * 441, int(44100 * 0.2))
    src_features = []
    for syllable in src_syllables:
        src_features.append(freq_features(syllable, 44100))
    logging.debug('src init ok')

    match_indexs = match(src_features, dst_features)
    logging.debug('match ok')

    new_music = np.zeros_like(music)
    for i in range(len(src_indexs)):
        dst_index = dst_indexs[match_indexs[i]] * 441
        src_index = src_indexs[i] * 441
        length = len(audios[match_indexs[i]])
        left = np.clip(int(src_index - dst_index), 0, len(new_music))
        right = np.clip(int(src_index + length - dst_index), 0, len(new_music))

        this_syllable = audios[match_indexs[i]][0:right - left]
        if args.cf:
            this_syllable = freq_correct(src_syllables[i], this_syllable)
        if args.ce:
            this_syllable = energy_correct(src_syllables[i], this_syllable, args.ea)
        new_music[left:right] = new_music[left:right] + this_syllable
    new_music = new_music + music * args.om + args.lfm * fft_filter(music, args.sf, [0, args.cut_off_low_freq])
    return new_music, dst_indexs, src_indexs, match_indexs


def generate_video(dst_indexs, src_indexs, match_indexs, args):
    dst_last_frames = []
    dst_adv_frames = np.round(np.array(dst_indexs) / 100 * args.fps).astype(np.int64)
    names = os.listdir(args.v2i)
    logging.info(names)
    for name in names:
        dst_last_frames.append(len(os.listdir(os.path.join(args.v2i, name))))

    fill_flags = -1 * np.ones(int(src_indexs[-1] / 100 * args.fps) + args.fps, dtype=np.int64)

    logging.debug('chkp 1')
    for i in range(len(src_indexs)):
        match_index = match_indexs[i]
        adv_frame = dst_adv_frames[match_indexs[i]]
        start_frame = int(np.round(src_indexs[i] / 100 * args.fps) - adv_frame)
        last_frame = int(dst_last_frames[match_index])

        logging.debug(f'chkp 1.1 {i}')

        for j in range(last_frame):
            fill_flags[start_frame + j] = match_index
            img = cv2.imread(os.path.join(args.v2i, '%02d' % match_index, '%05d' % (j + 1) + '.jpg'))
            imgop.imwrite(os.path.join(args.i2i, '%05d' % (start_frame + j) + '.jpg'), img)

        if i != (len(src_indexs) - 1) and start_frame + last_frame <= int(
                np.round(src_indexs[i + 1] / 100 * args.fps) - adv_frame):
            for j in range(
                    int(np.round(src_indexs[i + 1] / 100 * args.fps) - adv_frame) - start_frame - last_frame + 1):
                frame = start_frame + last_frame + j
                fill_flags[frame] = match_index
                img = cv2.imread(os.path.join(args.v2i, '%02d' % match_index, '%05d' % (last_frame) + '.jpg'))
                imgop.imwrite(os.path.join(args.i2i, '%05d' % (frame) + '.jpg'), img)
                # print(fill_flags)
    logging.debug('chkp 2')
    blackground = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(len(fill_flags)):
        if fill_flags[i] == -1:
            imgop.imwrite(os.path.join(args.i2i, '%05d' % (i) + '.jpg'), blackground)

    logging.debug('To generate')

    image2video(args.fps, os.path.join(args.i2i, '%5d.jpg'),
                os.path.join('.', 'log', 'output.wav'),
                os.path.join(args.output, 'output.mp4'))


def find_peaks():
    pass
