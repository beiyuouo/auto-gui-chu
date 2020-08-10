import os
from copy import deepcopy, copy

from utils.utils import *


class bgmclip:
    def __init__(self, args):
        self.args = args
        self.bgm_path = args.bgm
        self.bang_list = get_bang_list(args, args.bgm)
        self.bgm = AudioFileClip(args.bgm)
        self.length = self.bgm.duration

    def get_bang_list(self):
        return self.bang_list

    def get_length(self):
        return self.length

    def get_audio(self):
        return self.audio

    def show_wave(self):
        show_wave(self.args, self.bgm)


class avclip:
    def __init__(self, args):
        self.args = args
        self.av_path = args.video
        self.av_list = get_video(args.video)
        self.bang_list = [get_music_bang(args, VideoFileClip(i).audio) for i in self.av_list]
        self.av_length = [VideoFileClip(i).duration for i in self.av_list]

    def get_av(self, idx):
        return self.av_list[idx]

    def show_wave(self, idx=0):
        show_wave(self.args, VideoFileClip(self.av_list[idx]).audio)

    def get_av_length(self):
        return self.av_length

    def get_av_list(self):
        return self.av_list

    def get_bang_list(self):
        return self.bang_list


class guichu:
    def __init__(self, args):
        self.args = args
        self.av = avclip(args)
        print('-'*10, 'av done', '-'*10)
        self.bgm = bgmclip(args)
        print('-' * 10, 'bgm done', '-' * 10)
        self.play_list = self.get_video_list(self.bgm.get_bang_list(), self.av.get_av_list())
        print('-' * 10, 'play list done', '-' * 10)
        print(self.play_list)

    def get_video_list(self, bang_list, video_list):
        res_list = []
        cur = 0
        for i in range(len(bang_list)):
            if self.args.opt == 'normal':
                res_list.append(cur)
                cur = (cur + 1) % (len(video_list))
            elif self.args.opt == 'random':
                res_list.append(np.random.randint(0, len(video_list)))
            else:
                raise Exception("can't understand args.opt={}".format(self.args.opt))
        return res_list

    def make(self):
        video_list = self.play_list
        bang_list = self.bgm.get_bang_list()
        av_length_list = self.av.get_av_length()
        av_bang_list = self.av.get_bang_list()
        clip_list = []
        cur = 0
        bgm = self.bgm.bgm
        bgm = bgm.fx(afx.volumex, self.args.bgm_coefficient)
        print('total: ', len(self.play_list))
        try:
            os.remove(os.path.join(self.args.output, 'output.mp4'))
        except Exception:
            pass
        for i in range(len(bang_list)):
            av_length, av_bang = av_length_list[video_list[i]], av_bang_list[video_list[i]]
            bang = bang_list[cur]
            clip_list.append(copy(VideoFileClip(self.av.get_av(video_list[cur])))
                             .set_start(bang - av_bang - self.args.bias))
            print(cur, bang - av_bang)
            cur += 1
            if (cur+1) % self.args.seq == 0:
                final_clips = CompositeVideoClip(clip_list, size=(1920, 1080))
                final_clips.write_videofile(os.path.join(self.args.output, 'output_temp.mp4'))
                try:
                    os.remove(os.path.join(self.args.output, 'output.mp4'))
                except Exception:
                    pass
                try:
                    os.rename(os.path.join(self.args.output, 'output_temp.mp4'),
                              os.path.join(self.args.output, 'output.mp4'))
                except Exception:
                    pass
                clip_list.clear()
                clip_list.append(copy(VideoFileClip(os.path.join(self.args.output, 'output.mp4'))).set_start(0))

            if cur > 200:
                break

        final_clips = CompositeVideoClip(clip_list, size=(1920, 1080))
        clip_audio = copy(final_clips.audio).fx(afx.volumex, self.args.video_coefficient)
        final_clips.audio = CompositeAudioClip([clip_audio, bgm])
        final_clips.write_videofile(os.path.join(self.args.output, 'output_temp.mp4'))
        try:
            os.remove(os.path.join(self.args.output, 'output.mp4'))
        except Exception:
            pass
        try:
            os.rename(os.path.join(self.args.output, 'output_temp.mp4'),
                      os.path.join(self.args.output, 'output.mp4'))
        except Exception:
            pass
        print('-' * 10, 'finished', '-' * 10)
