#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import os
import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='./data')
    parser.add_argument('--bgm', type=str, default='bgm.mp3')
    parser.add_argument('--log', type=str, default='./log')
    parser.add_argument('--output', type=str, default='./result')
    parser.add_argument('--mini_seg', type=float, default=0.01)
    parser.add_argument('--mini_eps', type=float, default=0.1)
    parser.add_argument('--bgm_coefficient', type=float, default=2)
    parser.add_argument('--video_coefficient', type=float, default=3)
    parser.add_argument('--bias', type=float, default=0.1)
    parser.add_argument('--seq', type=int, default=10)
    parser.add_argument('--opt', type=str, default='normal')
    args = parser.parse_args()
    return args
