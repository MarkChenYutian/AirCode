#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
sys.path.append('.')

from visualize import showFeaturePts, showDetectionBoxes, showPointClusters, overlay_mask
# from draw_object import overlay_mask
from datasets.utils import preprocess
from model.inference import detection_inference
from model.build_model import build_maskrcnn, build_gcn, build_superpoint_model

from kornia.feature import match_nn
from torchvision import transforms

import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import yaml
import cv2
import os


# Preprocess Image
def preProcessImage(image):
    transform = transforms.ToTensor()
    return transform(image).unsqueeze(0)


# Read image in gray scale and cvt into 3 channels
def readImage(image_path):
    print(image_path)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if len(image.shape) == 2:
        image = cv2.merge([image, image, image])
    return image


# Load template file
def loadImage(config, image_name):
    image_path = os.path.join(config['data_root'], image_name)
    return readImage(image_path)


# Load models and set them to inference mode
def loadModels(configs):
    # read configs
    model_dir = configs['model_dir']
    configs['num_gpu'] = [0] if configs['use_gpu'] != 0 else []
    configs['public_model'] = 0

    superpoint_model_path = os.path.join(model_dir, "points_model.pth")
    maskrcnn_model_path   = os.path.join(model_dir, "maskrcnn_model.pth")
    gcn_model_path        = os.path.join(model_dir, "gcn_model.pth")

    configs["maskrcnn_model_path"]   = maskrcnn_model_path
    configs["superpoint_model_path"] = superpoint_model_path
    configs["graph_model_path"]      = gcn_model_path

    # model 
    superpoint_model = build_superpoint_model(configs, requires_grad=False)
    maskrcnn_model   = build_maskrcnn(configs)
    gcn_model        = build_gcn(configs)

    superpoint_model.eval()
    maskrcnn_model.eval()
    gcn_model.eval()

    return superpoint_model, maskrcnn_model, gcn_model


# Given an image, return the output from models
# images - a list of image that will be calculated with
def network_output(raw_images, superpoint_model, maskrcnn_model, gcn_model, config):
    with torch.no_grad():
        point_config  = config["model"]["superpoint"]
        detect_thresh = point_config["detection_threshold"]
        use_gpu       = config["use_gpu"]

        images = preProcessImage(raw_images)
        sizes  = [list(image.shape[-2:]) for image in images]
        batch  = {'image': images }
        
        points_output, detections, _ = detection_inference(
            maskrcnn_model, 
            superpoint_model,
            batch,
            use_gpu=use_gpu,
            gaussian_radius=1,
            detection_threshold=detect_thresh,
            data_config=config["data"]
        )

        # pts_output : [{
        #   points: Tensor[Nx2], 
        #   point_descs: Tensor[Nx256]
        # }]
        # pts_output[0]["points"] - the position of feature points in image
        # pts_output[0]["point_descs"] - the descriptor of feature points
        # 
        # Visualize:
        # showFeaturePts(points_output, raw_images)

        # detections : [{
        #   boxes : Tensor[Mx4],
        #   labels: Tensor[M], 1 - people, 2 - bike, 3 - car, 10 - traffic light
        #   scores: Tensor[M],
        #   masks : Tensor[Mx1xHxW], accurate mask for the detected object
        # }]
        # 
        # Visualize:
        # showDetectionBoxes(detections, raw_images)

        batch_points, batch_descriptors = preprocess.extract_points_clusters(points_output, list([detections[0]["masks"]]))
        # batch_points = [List[Tensor(Nx2), (len=M)] ]
        # batch_descriptors = [List[Tensor(Nx256), (len=M)]]
        #
        # Visualiuze:
        # showPointClusters(batch_points, raw_images)

        # Normalize points into [0, 1] range
        norm_points       = preprocess.normalize_points(batch_points, sizes)
        
        # Merge all object (point clusters) in the batch into one list of clusters
        merge_points      = preprocess.batch_merge(norm_points)
        merge_descriptors = preprocess.batch_merge(batch_descriptors)

        is_good_cluster   = preprocess.select_good_clusters(merge_points)
        # is_good_cluster: bool Tensor[M]
        
        good_points, good_descriptors = [], []
        for objcnt in range(len(is_good_cluster)):
            if is_good_cluster[objcnt].item():
                good_points.append(merge_points[objcnt])
                good_descriptors.append(merge_descriptors[objcnt])
        
        object_descriptors, _ = gcn_model(good_points, good_descriptors)

    # good_points - good point clusters for objects in images
    #       Shape: List[M'] with elems = Tensor(Nx2)
    # detections[0] - {boxes, labels, scores, masks}
    #       Note: detections includes both good and bad objects
    # object_descriptors
    #       Shape: Tensor[M' x 2048]
    # is_good_cluster
    #       Shape: bool Tensor[M']
    return good_points, detections[0], object_descriptors, is_good_cluster


# Given all the object descriptors (allDescs), find the matching ones
# in imgDescs.
def findMatching(imgDescs, allDescs):
    distance, matching = match_nn(imgDescs, allDescs)
    distance, matching = distance.cpu(), matching.cpu()
    distance, matching = distance.squeeze(1).numpy(), matching.numpy()
    return distance, matching


# Given distance of matching, return a bool array to show
# which descriptors are actually matched with current object
# descriptors and which are not
def filterMatching(distance, threshold=0.95):
    return [dist < threshold for dist in distance]


# Update the tensor of all known object descriptors summarized by AirCode
def updateObjectLibrary(imgDescs, allDescs, filterMatching):
    new_library = allDescs
    counter     = 0
    for idx, is_good in enumerate(filterMatching):
        if not is_good:
            counter += 1
            new_library = torch.vstack((new_library, imgDescs[idx]))
    print(f"updateObjectLibrary: {counter} new objects are added")
    return new_library


def generateMappingImage(masks, good_obj, matching):
    mapping = torch.zeros_like(masks[0])

    good_masks = masks[good_obj] 
    obj_match = {matching[i, 0] : matching[i, 1] for i in range(matching.shape[0])}

    for objcnt in range(good_masks.shape[0]):
        mapping += good_masks[objcnt] * obj_match[objcnt]
    
    return mapping

def generateVisualLabels(image, masks, boxes, good_obj, matching):
    masked_image = np.copy(image)

    good_boxes = []
    for i, box in enumerate(boxes):
        if good_obj[i].item(): good_boxes.append(box)
    
    good_masks = masks[good_obj]
    good_masks:np.ndarray = good_masks.cpu().numpy().astype(np.uint8)

    obj_match = {matching[i, 0] : matching[i, 1] for i in range(matching.shape[0])}

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0)]

    for key in obj_match:
        color = colors[obj_match[key] % len(colors)]
        box = good_boxes[key]
        mask = good_masks[key]

        overlay_mask(masked_image, mask, [color])
        cv2.putText(masked_image, str(obj_match[key]), (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color)
        cv2.rectangle(masked_image, (box[0], box[1]), (box[2], box[3]), color, 1)
    
    return masked_image


def main(config):
    sequence_dir = os.path.join(config["data_root"], "sequence")

    point_model, maskrcnn_model, gcn_model = loadModels(config)
    template = loadImage(config, "000000.png")
    template_enc = network_output(template, point_model, maskrcnn_model, gcn_model, config)
    all_object_encodings = template_enc[2]

    image_names = os.listdir(sequence_dir)
    image_names.sort()
    
    for image_name in image_names:
        targetIm   = loadImage(config, os.path.join("sequence", image_name))
        target_enc = network_output(targetIm, point_model, maskrcnn_model, gcn_model, config)

        # Now, we are use all known objects to match objects in target 
        new_object_encodings = target_enc[2]
        distance, matching = findMatching(new_object_encodings, all_object_encodings)

        # If an object in new image is not seen before, add its
        # descriptor to all_object_encodings
        is_good_match = filterMatching(distance, threshold=1.125)
        all_object_encodings = updateObjectLibrary(new_object_encodings, all_object_encodings, is_good_match)

        # re-match the objects, should be all good matches now
        distance, matching = findMatching(new_object_encodings, all_object_encodings)
        
        masks           = target_enc[1]["masks"]
        boxes           = target_enc[1]["boxes"]
        is_good_cluster = target_enc[3]
        
        # encode the matching result into a label image
        # map_result = generateMappingImage(masks, is_good_cluster, matching)
        # save result (an array with 0 as background, n as the object id)
        # np.save(os.path.join(config["save_dir"], image_name[:-4]), map_result)
        
        # visualize the matching result
        target_src = loadImage(config, os.path.join("sequence", image_name))
        vis_result = generateVisualLabels(target_src, masks, boxes, is_good_cluster, matching)
        cv2.imwrite(os.path.join(config["save_dir"], image_name[:-4]) + ".jpg", vis_result)

    
    print("All object descriptors: ", all_object_encodings.shape)
    np.save(os.path.join(config["save_dir"], "object_library"), all_object_encodings.cpu());

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
    
    with open(args.config_file, "r", encoding="utf-8") as f:
        config = yaml.load(f.read())
    
    config['use_gpu']   = args.gpu
    config['data_root'] = args.data_root
    config['model_dir'] = args.model_dir
    config['save_dir']  = args.save_dir

    main(config)

