# Author: BeiYu
# Github: https://github.com/beiyuouo
# Date  : 2021/2/11 10:33
# Description:

from .utils import *
from .dsp import *
import wave
import numpy as np
import matplotlib.pyplot as plt
import logging


class guichu(object):
    def __init__(self, args):
        self.args = args
        create_dir(args.log)
        create_dir(args.output)
        self.videos = [video(i, args) for i in str(args.video).split(',')]
        self.audio = audio(audio2wav(args.audio, args), args)
        self.gaudio = None
        self.gvideo = None
        logging.info(f'init guichu ok')

    def generate(self):
        logging.info('Reading files')
        audios = []
        for video in self.videos:
            audios.append(read_wave(video.audio.path))
        music = read_wave(self.audio.path)

        logging.info('Getting peak match')
        new_music, dst_indexs, src_indexs, match_indexs = peak_match(audios, music, self.args)

        logging.info('Writing file')
        self.gaudio = write(new_music, os.path.join('.', 'log', 'output.wav'), self.args.sf)

        logging.debug('Video to images')
        for idx, video in enumerate(self.videos):
            img_path = os.path.join(self.args.log, 'video_image', '{:02d}'.format(idx))
            create_dir(img_path)
            video2image(video.path, os.path.join(img_path, '%05d.jpg'), fps=self.args.fps)

        logging.info('Generating final video')
        self.args.v2i = os.path.join(self.args.log, 'video_image')
        self.args.i2i = os.path.join(self.args.log, 'output_image')
        create_dir(self.args.i2i)
        generate_video(dst_indexs, src_indexs, match_indexs, self.args)


class audio(object):
    def __init__(self, path: str, args):
        self.path = path  # TODO: REFORMAT
        self.args = args
        self.info = None
        self.energy = None
        self.peaks_idx = None

    def get_energy(self):
        if self.energy is None:
            self.energy = get_energy(self.path, kernel_size=4410, stride=441, padding=4410)
        return self.energy

    def get_peak(self):
        if self.peaks_idx is None:
            self.peaks_idx = find_peak(self.get_energy(), interval=self.args.interval,
                                       threshold=self.args.threshold)
        return self.peaks_idx

    def show_wave(self):
        show_peak(self.get_peak(), self.get_energy())


class video(object):
    def __init__(self, path: str, args):
        self.path = path
        self.args = args
        self.audio = audio(video2audio(self.path, self.args), self.args)
