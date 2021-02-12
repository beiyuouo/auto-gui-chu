#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import os
import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='video.mp4', help='Video path')
    parser.add_argument('--audio', type=str, default='audio.mp3', help='Audio path')
    parser.add_argument('--log', type=str, default=os.path.join('.', 'log'), help='Log path')
    parser.add_argument('--output', type=str, default=os.path.join('.', 'result'), help='Output path')
    # parser.add_argument('--v2i', type=str, default=os.path.join('.', 'log', 'video_image'), help='Video to image path')
    parser.add_argument('--viv', type=float, default=20, help='Video Interval')
    parser.add_argument('--aiv', type=float, default=100, help='Audio Interval()')
    parser.add_argument('--threshold', type=float, default=0.1, help='Threshold()')
    parser.add_argument('--fps', type=int, default=25, help='Fps for video')
    parser.add_argument('--sf', type=int, default=44100, help='Sample frequency for audio')
    parser.add_argument('--cut_off_low_freq', type=int, default=20, help='Cut off lowest frequency')
    parser.add_argument('--cut_off_high_freq', type=int, default=10000, help='Cut off highest frequency')
    parser.add_argument('--ea', type=float, default=0.3, help='EnergyAlpha')
    parser.add_argument('--cf', type=bool, default=False, help='CorrectFreq')
    parser.add_argument('--ce', type=bool, default=False, help='CorrectEnergy')
    parser.add_argument('--om', type=float, default=0.1, help='OriMix')
    parser.add_argument('--lfm', type=float, default=0.1, help='LfreqMix')
    parser.add_argument('--debug', type=bool, default=True, help='debug')
    args = parser.parse_args()
    return args
