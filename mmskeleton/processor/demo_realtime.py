#!/usr/bin/env python
import os
import sys

import time
import cv2
import torch
import logging
import numpy as np
from mmskeleton.utils import call_obj, import_obj, load_checkpoint
import deprecated.origin_stgcn_repo.tools.utils as utils
from mmcv.parallel import MMDataParallel


def test(model_cfg, dataset_cfg, checkpoint, batch_size=64, gpus=1, workers=4):

    model = call_obj(**model_cfg)
    edge = model.graph.edge
    load_checkpoint(model, checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=range(gpus)).cuda()
    model.eval()

    sys.path.append('{}/{}/build/python'.format(os.getcwd(), "openpose"))

    try:
        from openpose import pyopenpose as op
    except:
        print('Can not find Openpose Python API.')
        return
    opWrapper = op.WrapperPython()

    params = dict(model_folder='openpose/models', model_pose='COCO')
    params["hand"] = True

    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # video_capture = cv2.VideoCapture("mmskeleton/deprecated/st_gcn/resource/media/clean_and_jerk.mp4")
    video_capture = cv2.VideoCapture("fall01.mp4")
    pose_tracker = naive_pose_tracker()
    # start recognition
    start_time = time.time()
    frame_index = 0
    gt_labels = []
    with open('mmskeleton/deprecated/st_gcn/resource/kinetics_skeleton/label_name.txt', 'r') as f:
        for line in f:
            gt_labels.append(line.strip('\n'))

    while (True):
        tic = time.time()

        # get image
        ret, orig_image = video_capture.read()
        # orig_image = cv2.imread("3.jpg")
        if orig_image is None:
            break
        source_H, source_W, _ = orig_image.shape
        # orig_image = cv2.resize(
        #     orig_image, (256 * source_W // source_H, 256))
        H, W, _ = orig_image.shape

        # pose estimation

        datum = op.Datum()
        datum.cvInputData = orig_image
        opWrapper.emplaceAndPop([datum])

        multi_pose = datum.poseKeypoints  # (num_person, num_joint, 3)

        # orig_image = cv2.resize(orig_image, (768, 1024))
        # cv2.imshow("orig_image-GCN", orig_image)
        # cv2.waitKey(0)

        if len(multi_pose.shape) != 3:
            continue

        # normalization
        multi_pose[:, :, 0] = multi_pose[:, :, 0] / W
        multi_pose[:, :, 1] = multi_pose[:, :, 1] / H
        multi_pose[:, :, 0:2] = multi_pose[:, :, 0:2] - 0.5
        multi_pose[:, :, 0][multi_pose[:, :, 2] == 0] = 0
        multi_pose[:, :, 1][multi_pose[:, :, 2] == 0] = 0

        # pose tracking
        # if self.arg.video == 'camera_source':
        #     frame_index = int((time.time() - start_time) * self.arg.fps)
        # else:
        #     frame_index += 1
        frame_index += 1
        pose_tracker.update(multi_pose, frame_index)
        data_numpy = pose_tracker.get_skeleton_sequence()

        data = torch.from_numpy(data_numpy)

        data = data.unsqueeze(0)
        data = data.float().to("cuda:0").detach()
        with open("de.txt", 'w+') as f:
            for i in data[0][0]:
                f.write(str(i) + '\n\n')
        # break
        with torch.no_grad():
            output = model(data).data.cpu().numpy()
        voting_label = int(output.argmax(axis=1))

        print('voting_label_index:{}'.format(voting_label))
        print(len(gt_labels))
        print(gt_labels[voting_label])
        print(output[0][voting_label])
        app_fps = 1 / (time.time() - tic)
        image = render(edge, data_numpy, gt_labels[voting_label],
                            [[gt_labels[voting_label]]], None, orig_image, app_fps)
        cv2.imshow("ST-GCN", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def render(edge, data_numpy, voting_label_name, video_label_name, intensity, orig_image, fps=0):
    images = utils.visualization.stgcn_visualize(
        data_numpy[:, [-1]],
        edge,
        intensity, [orig_image],
        voting_label_name,
        [video_label_name[-1]],
        fps=fps)
    image = next(images)
    image = image.astype(np.uint8)
    return image


class naive_pose_tracker():
    """ A simple tracker for recording person poses and generating skeleton sequences.
    For actual occasion, I recommend you to implement a robuster tracker.
    Pull-requests are welcomed.
    """

    def __init__(self, data_frame=128, num_joint=18, max_frame_dis=np.inf):
        self.data_frame = data_frame
        self.num_joint = num_joint
        self.max_frame_dis = max_frame_dis
        self.latest_frame = 0
        self.trace_info = list()

    def update(self, multi_pose, current_frame):
        # multi_pose.shape: (num_person, num_joint, 3)

        if current_frame <= self.latest_frame:
            return

        if len(multi_pose.shape) != 3:
            return

        score_order = (-multi_pose[:, :, 2].sum(axis=1)).argsort(axis=0)

        for p in multi_pose[score_order]:

            # match existing traces
            matching_trace = None
            matching_dis = None
            for trace_index, (trace, latest_frame) in enumerate(self.trace_info):
                # trace.shape: (num_frame, num_joint, 3)
                if current_frame <= latest_frame:
                    continue
                mean_dis, is_close = self.get_dis(trace, p)
                if is_close:
                    if matching_trace is None:
                        matching_trace = trace_index
                        matching_dis = mean_dis
                    elif matching_dis > mean_dis:
                        matching_trace = trace_index
                        matching_dis = mean_dis

            # update trace information
            if matching_trace is not None:
                trace, latest_frame = self.trace_info[matching_trace]

                # padding zero if the trace is fractured
                pad_mode = 'interp' if latest_frame == self.latest_frame else 'zero'
                pad = current_frame-latest_frame-1
                new_trace = self.cat_pose(trace, p, pad, pad_mode)
                self.trace_info[matching_trace] = (new_trace, current_frame)

            else:
                new_trace = np.array([p])
                self.trace_info.append((new_trace, current_frame))

        self.latest_frame = current_frame

    def get_skeleton_sequence(self):

        # remove old traces
        valid_trace_index = []
        for trace_index, (trace, latest_frame) in enumerate(self.trace_info):
            if self.latest_frame - latest_frame < self.data_frame:
                valid_trace_index.append(trace_index)
        self.trace_info = [self.trace_info[v] for v in valid_trace_index]

        num_trace = len(self.trace_info)
        if num_trace == 0:
            return None
        print("num_trace:", num_trace)
        data = np.zeros((3, self.data_frame, self.num_joint, num_trace))
        for trace_index, (trace, latest_frame) in enumerate(self.trace_info):
            end = self.data_frame - (self.latest_frame - latest_frame)
            d = trace[-end:]
            beg = end - len(d)
            data[:, beg:end, :, trace_index] = d.transpose((2, 0, 1))

        return data

    # concatenate pose to a trace
    def cat_pose(self, trace, pose, pad, pad_mode):
        # trace.shape: (num_frame, num_joint, 3)
        num_joint = pose.shape[0]
        num_channel = pose.shape[1]
        if pad != 0:
            if pad_mode == 'zero':
                trace = np.concatenate(
                    (trace, np.zeros((pad, num_joint, 3))), 0)
            elif pad_mode == 'interp':
                last_pose = trace[-1]
                coeff = [(p+1)/(pad+1) for p in range(pad)]
                interp_pose = [(1-c)*last_pose + c*pose for c in coeff]
                trace = np.concatenate((trace, interp_pose), 0)
        new_trace = np.concatenate((trace, [pose]), 0)
        return new_trace

    # calculate the distance between a existing trace and the input pose

    def get_dis(self, trace, pose):
        last_pose_xy = trace[-1, :, 0:2]
        curr_pose_xy = pose[:, 0:2]

        mean_dis = ((((last_pose_xy - curr_pose_xy)**2).sum(1))**0.5).mean()
        wh = last_pose_xy.max(0) - last_pose_xy.min(0)
        scale = (wh[0] * wh[1]) ** 0.5 + 0.0001
        is_close = mean_dis < scale * self.max_frame_dis
        return mean_dis, is_close


