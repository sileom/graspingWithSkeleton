import argparse

import cv2
import numpy as np
import torch
import timeit

from models.with_mobilenet import PoseEstimationWithMobileNet, PoseEstimationWithMobileNetV3
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses
from val import normalize, pad_width


class ImageReader(object):
    def __init__(self, file_names):
        self.file_names = file_names
        self.max_idx = len(file_names)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx == self.max_idx:
            raise StopIteration
        #img = cv2.imread(self.file_names[self.idx], cv2.IMREAD_COLOR)
        img = cv2.imread('/home/monica/ros_catkin_ws_mine/src/skeleton/data/rgb_01.png', cv2.IMREAD_COLOR)
        if img.size == 0:
            raise IOError('Image {} cannot be read'.format(self.file_names[self.idx]))
        self.idx = self.idx + 1
        return img


class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img


def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1/256)):
    height, width, _ = img.shape
    scale = net_input_height_size / height
    # print(scale)

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cuda()

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    # print(stage2_heatmaps.shape)

    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad


def run_demo(net, image_provider, height_size, cpu, track, smooth):
    net = net.eval()
    if not cpu:
        net = net.cuda()

    stride = 16
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    previous_poses = []
    delay = 1
    for img in image_provider:
        orig_img = img.copy()
        start = timeit.default_timer()
        heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, cpu)
        stop = timeit.default_timer()
        print('Time: ', stop - start)  

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        ###################################################################
        # for i, keys in enumerate(all_keypoints_by_type):
        #   for key in keys:
        #     new_key_x = round((key[0] * stride / upsample_ratio - pad[1]) / scale)
        #     new_key_y = round((key[1] * stride / upsample_ratio - pad[0]) / scale)
        #     print(new_key_x, new_key_y)
        #     cv2.circle(img, (new_key_x, new_key_y), 3, (0, 0, 255), 3)
        #     cv2.putText(img, 'id: {}'.format(key[3]), (new_key_x, new_key_y - 16),
        #                       cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))

        # print("all_keypoints_by_type")   
        # print(all_keypoints_by_type)
        # pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        # print(pose_entries)
        # print(heatmaps.shape)
        # bests = []
        # for i, keys in enumerate(all_keypoints_by_type):
        #   if len(keys) > 1:
        #     highest = []
        #     for key in keys:
        #       print(i)
        #       value = heatmaps[key[1], key[0], i]
        #       highest.append(value)
        #       # new_key_x = round((key[0] * stride / upsample_ratio - pad[1]) / scale)
        #       # new_key_y = round((key[1] * stride / upsample_ratio - pad[0]) / scale)
        #       # print(new_key_x, new_key_y)
        #       # cv2.circle(img, (new_key_x, new_key_y), 3, (0, 0, 255), 3)
        #     print(highest)
        #     max_value = max(highest)
        #     max_index = highest.index(max_value)
        #     bests.append(keys[max_index])
        #   elif len(keys) == 1:
        #     bests.append(keys[0])
        #   else:
        #     bests.append([])

        # print(bests)

        # # draw connections
        # if bests[2]:
        #   new_key_x_c = round((bests[2][0] * stride / upsample_ratio - pad[1]) / scale)
        #   new_key_y_c = round((bests[2][1] * stride / upsample_ratio - pad[0]) / scale)
        #   for i, best in enumerate(bests):
        #     if i == 2:
        #       continue
        #     if best:
        #       new_key_x = round((best[0] * stride / upsample_ratio - pad[1]) / scale)
        #       new_key_y = round((best[1] * stride / upsample_ratio - pad[0]) / scale)
        #       cv2.line(img, (new_key_x_c, new_key_y_c), (new_key_x, new_key_y), (100, 255, 0), 2)

        # for i, best in enumerate(bests):
        #   if best:
        #     new_key_x = round((best[0] * stride / upsample_ratio - pad[1]) / scale)
        #     new_key_y = round((best[1] * stride / upsample_ratio - pad[0]) / scale)
        #     print(best, new_key_x, new_key_y)
        #     cv2.circle(img, (new_key_x, new_key_y), 3, (0, 0, 255), 3)
        #     cv2.putText(img, 'id: {}'.format(i), (new_key_x, new_key_y - 16),
        #                       cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
        ###########################################################################

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)

        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
        # print(all_keypoints)
        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                # print("kpt_id and n", kpt_id, n)
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    # print(int(pose_entries[n][kpt_id]))
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            # print(pose_keypoints)
            pose = Pose(pose_keypoints, pose_entries[n][5])
            current_poses.append(pose)
        
        if len(pose_entries) == 0:
            print()
            print('** NO KEYPOINTS DETECTED **')
            print()
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
        np.savetxt('/home/monica/ros_catkin_ws_mine/src/skeleton/data/keypoints.txt', pose_keypoints, fmt='%i', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None) # fmt='%10.5f' for float


        for pose in current_poses:
            pose.draw(img)
        img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
        for pose in current_poses:
            cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                          (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
            if track:
                cv2.putText(img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
                            
        cv2.imwrite('/home/monica/ros_catkin_ws_mine/src/skeleton/data/test2.png', img)
        #cv2.imwrite("test.png", img)
        # cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)
        # key = cv2.waitKey(delay)
        # if key == 27:  # esc
        #     return
        # elif key == 112:  # 'p'
        #     if delay == 1:
        #         delay = 0
        #     else:
        #         delay = 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Lightweight human pose estimation python demo.
                       This is just for quick results preview.
                       Please, consider c++ demo for the best performance.''')
    parser.add_argument('--checkpoint-path', type=str, required=True, help='path to the checkpoint')
    parser.add_argument('--height-size', type=int, default=384, help='network input layer height size')
    parser.add_argument('--video', type=str, default='', help='path to video file or camera id')
    parser.add_argument('--images', nargs='+', default='', help='path to input image(s)')
    parser.add_argument('--cpu', action='store_true', help='run network inference on cpu')
    parser.add_argument('--track', type=int, default=0, help='track pose id in video')
    parser.add_argument('--smooth', type=int, default=0, help='smooth pose keypoints')
    args = parser.parse_args()

    if args.video == '' and args.images == '':
        raise ValueError('Either --video or --image has to be provided')

    # net = PoseEstimationWithMobileNet()
    net = PoseEstimationWithMobileNetV3(pretrained='')
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    load_state(net, checkpoint)

    frame_provider = ImageReader(args.images)
    if args.video != '':
        frame_provider = VideoReader(args.video)
    else:
        args.track = 0

    run_demo(net, frame_provider, args.height_size, args.cpu, args.track, args.smooth)
