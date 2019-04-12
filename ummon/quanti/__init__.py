# -*- coding: utf-8 -*-
# @Author: daniel
# @Date:   2018-12-27 17:02:36
# @Last Modified by:   Daniel
# @Last Modified time: 2019-04-11 15:11:20
# from .quanti_net import QuantiNetWrapper
import logging
import os

__log = logging.getLogger('quanti_log')
os.makedirs("./logfile/", exist_ok=True)
# create file handler which logs even debug messages
fh = logging.FileHandler("./logfile/" + 'quanti.log')
fh.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
# add the handlers to the logger
__log.addHandler(fh)
__log.propagate = True
