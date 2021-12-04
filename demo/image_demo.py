# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import cv2
import torch
import warnings
from mmocr.apis import init_detector, model_inference
from mmocr.models import build_detector  # noqa: F401
import os


def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection webcam demo.')
    parser.add_argument('img_file', help='Image file')
    parser.add_argument('out_file', help='output file')
    parser.add_argument('config', help='Test config file path.')
    parser.add_argument('checkpoint', help='Checkpoint file.')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option.')
    parser.add_argument(
        '--camera-id', type=int, default=0, help='Camera device id.')
    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='Bbox score threshold.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    device = torch.device(args.device)

    model = init_detector(args.config, args.checkpoint, device=device)
    if model.cfg.data.test['type'] == 'ConcatDataset':
        model.cfg.data.test.pipeline = model.cfg.data.test['datasets'][
            0].pipeline

    if not os.path.isdir(args.out_file):
        res = input(f"{args.out_file} doesn't exist! Do you want to create it? yes/no")
        if res.startswith('y'):
            os.mkdir(args.out_file)
        else:
            return FileNotFoundError

    assert os.path.isdir(args.out_file)
    img_file_list = os.listdir(args.img_file)
    for img_name in img_file_list:
        if not img_name[-4:] not in ["jpg", "png", "bmp"]:
            warnings.warn(f"{img_name} is not a image name")
            continue
        img_path = os.path.join(args.img_file, img_name)
        out_path = os.path.join(args.out_file, img_name)
        img = cv2.imread(img_path)
        result = model_inference(model, img)
        model.show_result(img, result, score_thr=args.score_thr, wait_time=1, show=True)


if __name__ == '__main__':
    main()
