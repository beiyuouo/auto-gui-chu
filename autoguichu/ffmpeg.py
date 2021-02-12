# Author: BeiYu
# Github: https://github.com/beiyuouo
# Date  : 2021/2/11 15:41
# Description:

import os
import logging


class ffmpeg(object):
    def __init__(self, input: str, output: str, args: list or str):
        self.input = input
        self.output = output
        self.cmd = self.__args2cmd__(args)

    def __args2cmd__(self, args: list or str) -> str:
        logging.debug(args)
        if isinstance(args, str):
            return args

        cmd = f'ffmpeg -i {self.input} '
        cmd += ' '.join(args)
        cmd += f' {self.output}'
        return cmd

    def run(self):
        logging.debug(self.cmd)
        os.system(self.cmd)
