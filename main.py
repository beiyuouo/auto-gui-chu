import os
from copy import copy

from utils.options import *
from moviepy.editor import *
import numpy as np
import matplotlib.pyplot as plt


def get_video(args):
    list_tmp = os.listdir(args.video)
    video_list = []
    for i in range(0, len(list_tmp)):
        path = os.path.join(args.video, list_tmp[i])
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
        max_array.append(np.max(music_array[st:ed]))
    # print(max_array)
    result = []
    for i in range(1, len(max_array) - 1):
        if max_array[i] - args.mini_eps > max(max_array[i - 1], max_array[i + 1]):
            result.append(i * args.mini_seg)
    return result


def make_video_list(args, bang_list, video_list):
    list = []
    cur = 0
    for i in range(bang_list):
        if args.opt == 'normal':
            list.append(video_list[cur])
            cur = (cur + 1) % (len(video_list))
        elif args.opt == 'random':
            list.append(np.random.randint(0, len(video_list)))
        else:
            raise Exception("can't understand args.opt={}".format(args.opt))
    return list


def get_video_info(video_path):
    video = VideoFileClip(video_path)
    video_length = video.duration / video.fps
    audio_bang = get_music_bang(args, video.audio)
    video.close()
    return video_length, audio_bang


def get_play_list(args, video_list, bang_list):
    clip_list = []
    cur = 0
    for i in range(len(bang_list)):
        video_length, video_bang = get_video_info(video_list[cur])
        bang = bang_list[cur]
        clip_list.append(copy(VideoFileClip(video_list[cur]).set_start(bang - video_bang - args.bias)))
        print(cur, bang - video_bang)
        cur += 1
        if cur > 80:
            break
    final_Clips = CompositeVideoClip(clip_list, size=(1920, 1080))
    bgm = AudioFileClip(args.bgm)
    bgm = bgm.fx(afx.volumex, args.bgm_coefficient)
    clip_audio = copy(final_Clips.audio).fx(afx.volumex, args.video_coefficient)
    final_Clips.audio = CompositeAudioClip([clip_audio, bgm])
    final_Clips.write_videofile(os.path.join(args.output, 'output.mp4'))


def test(args):
    video_list = get_video(args)
    bgm = args.bgm
    print(video_list)
    # merge_video(args, video_list)
    # get_music_bang(args, bgm)
    print(get_music_bang(args, bgm))
    bang_list = get_bang_list(args, bgm)
    print(bang_list)
    # show_wave(args, bgm)
    video_list = make_video_list(args, len(bang_list), video_list)
    print(video_list)
    video = VideoFileClip(video_list[0])
    print(get_music_bang(args, video.audio))
    show_wave(args, video.audio)
    get_video_info(video_list[0])


def main(args):
    video_list = get_video(args)
    bgm = args.bgm
    bang_list = get_bang_list(args, bgm)
    video_list = make_video_list(args, len(bang_list), video_list)
    print(bang_list)
    print(video_list)
    print(len(bang_list))
    get_play_list(args, video_list, bang_list)


if __name__ == '__main__':
    args = args_parser()
    main(args)
    # test(args)
