#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   recognition_granary.py.py    
@Contact :   juzheng@hxdi.com
@License :   (C)Copyright 2018-2019, Liugroup-NLPR-CASIA

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/10/23 15:07   juzheng      1.0         None
"""

import os
import cv2
import torch
import logging
import json
import numpy as np
from mmskeleton.utils import call_obj, import_obj, load_checkpoint, cache_checkpoint
from mmcv.runner import Runner
from mmcv import Config, ProgressBar
from mmcv.parallel import MMDataParallel
import mmcv
from mmskeleton.apis.estimation import init_pose_estimator, inference_pose_estimator
from multiprocessing import current_process, Process, Manager


# process a batch of data
def batch_processor(model, datas, train_mode, loss):

    data, label = datas
    data = data.cuda()
    label = label.cuda()

    # forward
    output = model(data)
    losses = loss(output, label)

    # output
    log_vars = dict(loss=losses.item())
    if not train_mode:
        log_vars['top1'] = topk_accuracy(output, label)
        log_vars['top5'] = topk_accuracy(output, label, 5)

    outputs = dict(loss=losses, log_vars=log_vars, num_samples=len(data.data))
    return outputs


def topk_accuracy(score, label, k=1):
    rank = score.argsort()
    hit_top_k = [l in rank[i, -k:] for i, l in enumerate(label)]
    accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
    return accuracy


def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv1d') != -1:
        model.weight.data.normal_(0.0, 0.02)
        if model.bias is not None:
            model.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        model.weight.data.normal_(0.0, 0.02)
        if model.bias is not None:
            model.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)


def render(image, pred, label, person_bbox, bbox_thre=0):
    if pred is None:
        return image

    mmcv.imshow_det_bboxes(image,
                           person_bbox,
                           np.zeros(len(person_bbox)).astype(int),
                           class_names=['person'],
                           score_thr=bbox_thre,
                           show=False,
                           wait_time=0)

    for person_pred in pred:
        for i, joint_pred in enumerate(person_pred):
            cv2.circle(image, (int(joint_pred[0]), int(joint_pred[1])), 2,
                       [255, 0, 0], 2)
            cv2.putText(image, '{}'.format(i), (int(joint_pred[0]), int(joint_pred[1])),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, [255, 255, 255])

        cv2.putText(image, '{}'.format(label), (int(person_pred[0][0]), int(person_pred[0][1] - 20)),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, [255, 255, 255])
        image = draw_body_pose(image, person_pred)
    return np.uint8(image)


# draw the body keypoint and lims
def draw_body_pose(image, person_pred):
    line_seq = [[0, 2], [2, 4], [0, 1], [1, 3], [0, 6], [6, 8], [8, 10], [0, 5],
                [5, 7], [7, 9], [0, 12], [12, 14], [14, 16], [0, 11], [11, 13], [13, 15]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    for i, line in enumerate(line_seq):
        first_index, second_index = line
        first_point = person_pred[first_index]
        second_point = person_pred[second_index]
        cv2.line(image, (int(first_point[0]), int(first_point[1])), (int(second_point[0]), int(second_point[1])),
                 colors[i], 2)
    return image


def build(inputs,
          detection_cfg,
          estimation_cfg,
          tracker_cfg,
          video_dir,
          gpus=1,
          video_max_length=10000,
          category_annotation=None):
    print('data build start')
    cache_checkpoint(detection_cfg.checkpoint_file)
    cache_checkpoint(estimation_cfg.checkpoint_file)

    if category_annotation is None:
        video_categories = dict()
    else:
        with open(category_annotation) as f:
            video_categories = json.load(f)['annotations']

    if tracker_cfg is not None:
        raise NotImplementedError

    pose_estimators = init_pose_estimator(
        detection_cfg, estimation_cfg, device=0)

    video_file_list = os.listdir(video_dir)
    prog_bar = ProgressBar(len(video_file_list))
    for video_file in video_file_list:
        reader = mmcv.VideoReader(os.path.join(video_dir, video_file))
        video_frames = reader[:video_max_length]

        annotations = []
        num_keypoints = -1
        for i, image in enumerate(video_frames):
            res = inference_pose_estimator(pose_estimators, image)
            res['frame_index'] = i
            if not res['has_return']:
                continue
            num_person = len(res['joint_preds'])
            assert len(res['person_bbox']) == num_person

            for j in range(num_person):
                keypoints = [[p[0], p[1], round(s[0], 2)] for p, s in zip(
                    res['joint_preds'][j].round().astype(int).tolist(), res[
                        'joint_scores'][j].tolist())]
                num_keypoints = len(keypoints)
                person_info = dict(
                    person_bbox=res['person_bbox'][j].round().astype(int).tolist(),
                    frame_index=res['frame_index'],
                    id=j,
                    person_id=None,
                    keypoints=keypoints)
                annotations.append(person_info)
        annotations = sorted(annotations, key=lambda x: x['frame_index'])
        category_id = video_categories[video_file][
            'category_id'] if video_file in video_categories else -1
        info = dict(
            video_name=video_file,
            resolution=reader.resolution,
            num_frame=len(video_frames),
            num_keypoints=num_keypoints,
            keypoint_channels=['x', 'y', 'score'],
            version='1.0')
        video_info = dict(
            info=info, category_id=category_id, annotations=annotations)
        inputs.put(video_info)
        prog_bar.update()


def data_parse(data, pipeline, num_track=1):
    info = data['info']
    annotations = data['annotations']
    num_frame = info['num_frame']
    num_keypoints = info['num_keypoints']
    channel = info['keypoint_channels']
    num_channel = len(channel)

    # get data
    data['data'] = np.zeros(
        (num_channel, num_keypoints, num_frame, num_track),
        dtype=np.float32)

    for a in annotations:
        person_id = a['id'] if a['person_id'] is None else a['person_id']
        frame_index = a['frame_index']
        if person_id < num_track and frame_index < num_frame:
            data['data'][:, :, frame_index, person_id] = np.array(
                a['keypoints']).transpose()
    # 数据预处理
    for stage_args in pipeline:
        data = call_obj(data=data, **stage_args)
    return data


def detect(inputs, results, model_cfg, dataset_cfg, checkpoint, video_dir,
           batch_size=64, gpus=1, workers=4):
    print('detect start')
    # put model on gpus
    if isinstance(model_cfg, list):
        model = [call_obj(**c) for c in model_cfg]
        model = torch.nn.Sequential(*model)
    else:
        model = call_obj(**model_cfg)
    load_checkpoint(model, checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=range(gpus)).cuda()
    model.eval()

    results = []
    labels = []
    video_file_list = os.listdir(video_dir)
    prog_bar = ProgressBar(len(video_file_list))
    for video_file in video_file_list:
        data = inputs.get()
        data_loader = data_parse(data, dataset_cfg.pipeline, dataset_cfg.data_source.num_track)
        data, label = data_loader
        with torch.no_grad():
            data = torch.from_numpy(data)
            # 增加一维，表示batch_size
            data = data.unsqueeze(0)
            data = data.float().to("cuda:0").detach()
            output = model(data).data.cpu().numpy()
        results.append(output)
        labels.append(torch.tensor([label]))
        for i in range(len(data)):
            prog_bar.update()
    print('--------', results, labels, '--------------')
    results = np.concatenate(results)
    labels = np.concatenate(labels)

    print('Top 1: {:.2f}%'.format(100 * topk_accuracy(results, labels, 1)))
    print('Top 5: {:.2f}%'.format(100 * topk_accuracy(results, labels, 5)))


def realtime_detect(detection_cfg, estimation_cfg, model_cfg, dataset_cfg, tracker_cfg, video_dir,
                    category_annotation, checkpoint, batch_size=64, gpus=1, workers=4):
    # 初始化模型
    pose_estimators = init_pose_estimator(
        detection_cfg, estimation_cfg, device=0)
    if isinstance(model_cfg, list):
        model = [call_obj(**c) for c in model_cfg]
        model = torch.nn.Sequential(*model)
    else:
        model = call_obj(**model_cfg)
    load_checkpoint(model, checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=range(gpus)).cuda()
    model.eval()

    # 获取图像
    video_file = 'walk3.avi'
    reader = mmcv.VideoReader(os.path.join(video_dir, video_file))
    video_frames = reader[:10000]

    if category_annotation is None:
        video_categories = dict()
    else:
        with open(category_annotation) as f:
            json_file = json.load(f)
            video_categories = json_file['annotations']
            action_class = json_file['categories']
    annotations = []
    num_keypoints = -1
    for i, image in enumerate(video_frames):
        res = inference_pose_estimator(pose_estimators, image)
        res['frame_index'] = i
        if not res['has_return']:
            continue
        num_person = len(res['joint_preds'])
        assert len(res['person_bbox']) == num_person

        for j in range(num_person):
            keypoints = [[p[0], p[1], round(s[0], 2)] for p, s in zip(
                res['joint_preds'][j].round().astype(int).tolist(), res[
                    'joint_scores'][j].tolist())]
            num_keypoints = len(keypoints)
            person_info = dict(
                person_bbox=res['person_bbox'][j].round().astype(int).tolist(),
                frame_index=res['frame_index'],
                id=j,
                person_id=None,
                keypoints=keypoints)
            annotations.append(person_info)
        category_id = video_categories[video_file][
            'category_id'] if video_file in video_categories else -1
        info = dict(
            video_name=video_file,
            resolution=reader.resolution,
            num_frame=len(video_frames),
            num_keypoints=num_keypoints,
            keypoint_channels=['x', 'y', 'score'],
            version='1.0')
        video_info = dict(info=info, category_id=category_id, annotations=annotations)

        data_loader = data_parse(video_info, dataset_cfg.pipeline, dataset_cfg.data_source.num_track)
        data, label = data_loader
        with torch.no_grad():
            data = torch.from_numpy(data)
            # 增加一维，表示batch_size
            data = data.unsqueeze(0)
            data = data.float().to("cuda:0").detach()
            output = model(data).data.cpu().numpy()
        top1 = output.argmax()
        print("reslt:", output)
        res['render_image'] = render(image, res['joint_preds'],
                                     action_class[top1],
                                     res['person_bbox'],
                                     detection_cfg.bbox_thre)
        cv2.imshow('image', image)
        cv2.waitKey(10)


def recognition(detection_cfg, estimation_cfg, model_cfg, dataset_cfg, tracker_cfg, video_dir,
                category_annotation, checkpoint, batch_size=64, gpus=1, workers=4):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    inputs = Manager().Queue(10000)
    results = Manager().Queue(10000)
    procs = []
    p1 = Process(
        target=build,
        args=(inputs, detection_cfg, estimation_cfg, tracker_cfg, video_dir,
              gpus, 10000, category_annotation))
    p1.start()
    procs.append(p1)
    p2 = Process(
        target=detect,
        args=(inputs, results,  model_cfg, dataset_cfg, checkpoint, video_dir,
              batch_size, gpus, workers))
    p2.start()
    procs.append(p2)
    for p in procs:
        p.join()

