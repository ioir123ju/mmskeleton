#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   build_label.py    
@Contact :   JZ
@License :   (C)Copyright 2018-2019, Liugroup-NLPR-CASIA

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/10/29 15:08   juzheng      1.0         None
"""

import os
import json


label_id = dict(walk=0, run=1, fall_down=2, jump=3, squart=4, clean=5)
label_name = ['walk', 'run', 'fall_down', 'jump', 'squart', 'clean']


def build(video_dir, out_dir):
    category_annotattion = dict()
    path_dir = video_dir
    categories = []
    annotations = {}
    for dir_name in next(os.walk(path_dir))[1]:
        print(dir_name)
        categories.append(dir_name)
        action_path = os.path.join(path_dir, dir_name)
        for file in next(os.walk(action_path))[2]:
            video_dict = dict(category_id=label_id[dir_name])
            annotations[file] = video_dict
    category_annotattion.update(categories=label_name, annotations=annotations)
    with open(os.path.join(out_dir, 'category_annotation_granary' + '.json'), 'w') as f:
        json.dump(category_annotattion, f)
