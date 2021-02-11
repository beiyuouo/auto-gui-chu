# Author: BeiYu
# Github: https://github.com/beiyuouo
# Date  : 2021/2/11 10:33
# Description:

from autoguichu.utils import *
from autoguichu import *


class guichu(object):
    def __init__(self, args):
        self.config = args
        create_dir(args.log)
        create_dir(args.output)
        self.videos = [str(args.video).split('')]
        self.audio = args.audio

    def generate(self):
        pass


class audio(object):
    def __init__(self, path: str):
        pass


class video(object):
    def __init__(self, path: str):
        pass
