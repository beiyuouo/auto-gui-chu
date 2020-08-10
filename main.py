import os
from copy import copy

from utils.options import *
from utils.GuichuClip import *
from moviepy.editor import *
import numpy as np
import matplotlib.pyplot as plt


def test(args):
    my = guichu(args)


def main(args):
    my = guichu(args)
    my.make()


if __name__ == '__main__':
    args = args_parser()
    main(args)
    # test(args)
