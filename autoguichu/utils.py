# Author: BeiYu
# Github: https://github.com/beiyuouo
# Date  : 2021/2/11 10:20
# Description:

import os
import logging

from scipy.io import wavfile

from .ffmpeg import ffmpeg
import numpy as np

args = None


def create_dir(path: str):
    try:
        os.makedirs(path)
    except Exception as e:
        pass


def video2audio(input_path: str, args, output_path: str = None) -> str:
    inputs = input_path
    if output_path is not None:
        outputs = output_path
    else:
        outputs = os.path.join(os.path.dirname(args.log),
                               os.path.basename(input_path).split('.')[0] + '.wav')

    logging.debug(f'Coverting media [{inputs}] to audio [{outputs}]')

    fargs = []
    if args is not None:  # TODO: remove this
        fargs += ['-ar', str(args.sf)]
    fargs += ['-y']

    ff = ffmpeg(inputs, outputs, fargs)
    ff.run()

    return outputs


def video2video(input_path: str, args, output_path: str = None) -> str:
    inputs = input_path
    if output_path is not None:
        outputs = output_path
    else:
        outputs = os.path.join(os.path.dirname(args.log),
                               os.path.basename(input_path).split('.')[0] + '.mp4')

    logging.debug(f'Coverting media [{inputs}] to video [{outputs}]')

    fargs = []
    if args is not None:
        fargs += ['-fps', str(args.fps)]
    fargs += ['-c:v', 'copy', '-an', '-y']
    ff = ffmpeg(inputs, outputs, fargs)
    ff.run()

    return outputs


def audio2wav(input_path: str, args, output_path: str = None) -> str:
    inputs = input_path
    if output_path is not None:
        outputs = output_path
    else:
        outputs = os.path.join(os.path.dirname(args.log),
                               os.path.basename(input_path).split('.')[0] + '.wav')

    logging.debug(f'Coverting audio [{inputs}] to wav [{outputs}]')

    fargs = []
    if args is not None:
        fargs += ['-ar', str(args.sf)]
    fargs += ['-y']
    ff = ffmpeg(inputs, outputs, fargs)
    ff.run()

    return outputs


def video2image(videopath: str, imagepath: str, fps: int = 0, start_time: int = 0, last_time: int = 0):
    inputs = videopath
    outputs = imagepath

    logging.debug(f'Coverting audio [{inputs}] to wav [{outputs}]')

    fargs = []
    if last_time != 0:
        fargs += ['-ss', str(start_time)]
        fargs += ['-t', str(last_time)]
    if fps != 0:
        fargs += ['-r', str(fps)]
    fargs += ['-f', 'image2', '-q:v', '-0']
    fargs += ['-y']
    ff = ffmpeg(inputs, outputs, fargs)
    ff.run()

    return outputs


def image2video(fps, imagepath, voicepath, videopath):
    inputs = imagepath
    tmpoutputs = os.path.join('.', 'log', 'temp.mp4')  # TODO
    outputs = videopath

    logging.debug(f'Coverting audio [{inputs}] to wav [{outputs}]')

    fargs = []
    fargs += ['-r', str(fps)]
    fargs += ['-y']
    fargs += ['-vcodec libx264']
    ff = ffmpeg(inputs, tmpoutputs, fargs)
    ff.run()
    fargs = []
    fargs += ['-i', voicepath]
    fargs += ['-y']
    fargs += ['-vcodec copy']
    fargs += ['-acodec aac']
    ff = ffmpeg(tmpoutputs, outputs, fargs)
    ff.run()


def merge_video_audio(video_path: str, audio_path: str, args, output_path: str = None) -> str:
    pass


def numpy2voice(npdata):
    voice = np.zeros((len(npdata), 2))
    voice[:, 0] = npdata
    voice[:, 1] = npdata
    return voice


def write(npdata, path='./tmp/test_output.wav', freq=44100):
    voice = numpy2voice(npdata)
    wavfile.write(path, freq, voice.astype(np.int16))
    return path
