import os
from copy import deepcopy

from moviepy.editor import *
import numpy as np
import matplotlib.pyplot as plt


def get_video(video_path):
    list_tmp = os.listdir(video_path)
    video_list = []
    for i in range(0, len(list_tmp)):
        path = os.path.join(video_path, list_tmp[i])
        if os.path.isfile(path):
            if path.endswith('.mp4'):
                video_list.append(path)
    return video_list


def merge_video(args, video_list):
    clip_list = []
    for path in video_list:
        clip = VideoFileClip(path)
        clip_list.append(clip)
    video = concatenate_videoclips(clip_list)
    video.write_videofile(os.path.join(args.output, 'output.mp4'))


def show_wave(args, music_path):
    if isinstance(music_path, str):
        music = AudioFileClip(music_path)
    else:
        music = music_path
    rate = 10
    # music.max_volume(0.5)
    # print(music)
    music_array = music.to_soundarray()
    music_array = np.array(music_array)
    music_array = music_array[::rate]
    # print(music.fps)
    # print(len(music_array))
    # print(len(music_array) / music.fps * rate)
    music_subarray = music_array[:int(10 * (music.fps / rate))]
    time = np.arange(0, len(music_subarray)) * (1.0 / music.fps * rate)
    plt.subplot(211)
    plt.plot(time, music_subarray[:, 0])
    plt.subplot(212)
    plt.plot(time, music_subarray[:, 1], c="g")
    plt.xlabel("time (seconds)")
    plt.show()


def get_music_bang(args, music_path):
    if isinstance(music_path, str):
        music = AudioFileClip(music_path)
    else:
        music = music_path
    rate = 10
    # music.max_volume(0.5)
    # print(music)
    music_array = music.to_soundarray()
    music_array = np.array(music_array)
    music_array = music_array[::rate]
    # print(music.fps)
    # print(len(music_array))
    # print(len(music_array) / music.fps * rate)
    music_len = len(music_array) / music.fps * rate
    len_unit = int(len(music_array) / (music_len / args.mini_seg))
    # print(music_len, len_unit)
    return np.where(music_array == np.max(music_array))[0][0] / (music.fps / rate)


def get_bang_list(args, music_path):
    if isinstance(music_path, str):
        music = AudioFileClip(music_path)
    else:
        music = music_path
    rate = 10
    # music.max_volume(0.5)
    # print(music)
    music_array = music.to_soundarray()
    music_array = np.array(music_array)
    music_array = music_array[::rate]
    # print(music.fps)
    # print(len(music_array))
    # print(len(music_array) / music.fps * rate)
    music_len = len(music_array) / music.fps * rate
    len_unit = int(len(music_array) / (music_len / args.mini_seg))
    # print(music_len, len_unit)
    max_array = []
    for i in range(0, len(music_array), len_unit):
        st = i
        ed = min(len(music_array), i + len_unit)
        max_array.append(np.max(music_array[st:ed, 0]))
    # print(max_array)
    result = []
    for i in range(1, len(max_array) - 1):
        if max_array[i] - args.mini_eps > max(max_array[i - 1], max_array[i + 1]):
            result.append(i * args.mini_seg)
    return result


def get_video_info(args, video_path):
    if isinstance(video_path, str):
        video = VideoFileClip(video_path)
    else:
        video = video_path

    video_length = video.duration / video.fps
    audio_bang = get_music_bang(args, video.audio)
    video.close()
    return video_length, audio_bang
