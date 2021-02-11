#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import os
import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='video.mp4', help='video path')
    parser.add_argument('--audio', type=str, default='audio.mp3', help='audio path')
    parser.add_argument('--log', type=str, default='./log', help='log path')
    parser.add_argument('--output', type=str, default='./result', help='output path')
    parser.add_argument('--mini_seg', type=float, default=0.01, help='mini seg')
    parser.add_argument('--fps', type=int, default=25, help='fps for video')
    parser.add_argument('--sr', type=int, default=44100, help='sample rate for audio')
    args = parser.parse_args()
    return args
