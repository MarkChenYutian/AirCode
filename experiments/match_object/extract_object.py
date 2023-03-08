#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import yaml
import argparse

import matplotlib.pyplot as plt

from match_object import \
    mkdirIfNotExist, \
    loadModels, \
    loadImage, \
    network_output, \
    filterNetworkOutput

def main(config):
    sequence_dir = config["data_root"]
    visual_save_dir = os.path.join(config["save_dir"], "visualize")
    label_save_dir  = os.path.join(config["save_dir"], "label")
    obj_save_dir    = os.path.join(config["save_dir"], "object")
    mkdirIfNotExist(visual_save_dir, label_save_dir, obj_save_dir)
    
    input_image_names = os.listdir(sequence_dir)
    input_image_names.sort()

    point_model, maskrcnn_model, gcn_model = loadModels(config)

    for input_image_name in input_image_names:
        print("=> ", input_image_name)

        input_image = loadImage(config, input_image_name)
        target_out  = network_output(input_image, point_model, maskrcnn_model, gcn_model, config)
        target_out = filterNetworkOutput(target_out)

        _, detections, _, _ = target_out
        target_boxes  = detections["boxes"].cpu().numpy()
        target_labels = detections["labels"].cpu().numpy()
        target_scores = detections["scores"].cpu().numpy()
        target_masks  = detections["masks"].cpu().numpy()

        print(target_masks.shape)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="match objects in two images")
    parser.add_argument(
        "-c", "--config_file",
        dest="config_file",
        type=str,
        default=""
    )
    parser.add_argument(
        "-g", "--gpu",
        dest = "gpu",
        type = int, 
        default = 0 
    )
    parser.add_argument(
        "-s", "--save_dir",
        dest = "save_dir",
        type = str, 
        default = "" 
    )
    parser.add_argument(
        "-d", "--data_root",
        dest = "data_root",
        type = str, 
        default = "" 
    )
    parser.add_argument(
        "-m", "--model_dir",
        dest = "model_dir",
        type = str, 
        default = "" 
    )
    args = parser.parse_args()

    print(os.getcwd())
    print(os.path.abspath(args.config_file));
    
    with open(args.config_file, "r", encoding="utf-8") as f:
        config = yaml.load(f.read(), Loader=yaml.UnsafeLoader)
    
    config['use_gpu']   = args.gpu
    config['data_root'] = args.data_root
    config['model_dir'] = args.model_dir
    config['save_dir']  = args.save_dir

    main(config)

