# Author: BeiYu
# Github: https://github.com/beiyuouo
# Date  : 2021/2/11 10:20
# Description:

import os


def create_dir(path: str):
    try:
        os.makedirs(path)
    except Exception as e:
        pass


