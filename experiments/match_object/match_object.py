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

from typing import Dict, List
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
    # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # if len(image.shape) == 2:
    #     image = cv2.merge([image, image, image])
    image = cv2.imread(image_path, cv2.IMREAD_COLOR);
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

        # print(len(merge_points))
        # print(detections[0]["masks"].shape)

        is_good_cluster   = preprocess.filter_good_object(merge_points, detections[0]["masks"], size_thr=.001)
        # is_good_cluster: bool Tensor[M]
        
        good_points, good_descriptors = [], []
        for objcnt in range(len(is_good_cluster)):
            if is_good_cluster[objcnt].item():
                good_points.append(merge_points[objcnt])
                good_descriptors.append(merge_descriptors[objcnt])
        
        # If no good object, we return some empty result
        if len(good_points) == 0 and len(good_descriptors) == 0:
            empty_points = []
            empty_detection = {
                "boxes":  torch.zeros((0, 4)),
                "labels": torch.zeros((0,)),
                "scores": torch.zeros((0,)),
                "masks":  torch.zeros((0, 1, sizes[0][0], sizes[0][1]))
            }
            empty_descriptors = torch.zeros((0, 2048))
            empty_cluster     = torch.zeros((0,)).to(torch.bool)
            return empty_points, empty_detection, empty_descriptors, empty_cluster
        
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
def findMatching(new_output, obj_library):
    _, detections, object_descriptors, _ = new_output
    labels = detections["labels"]
    
    distances = []
    matching  = []

    for objcnt in range(object_descriptors.shape[0]):
        obj_descs = object_descriptors[objcnt][np.newaxis, :]
        obj_label = labels[objcnt]
        obj_labval= obj_label.item()

        if obj_labval in obj_library:
            lib_descs = obj_library[obj_labval]

            dist, match = match_nn(obj_descs, lib_descs)
            dist, match = dist.cpu().squeeze(1).numpy(), match.cpu().numpy()
            distances.append(dist[0])
            # match_result = (obj_labval, match[0])
            matching.append({ "label" : obj_labval, "id" : match[0, 1] })
        else:
            distances.append(float("Inf"))
            matching.append({ "label" : -1, "id" : -1 })

    # distance, matching = match_nn(imgDescs, allDescs)
    # distance, matching = distance.cpu(), matching.cpu()
    # distance, matching = distance.squeeze(1).numpy(), matching.numpy()
    return distances, matching


# Given distance of matching, return a bool array to show
# which descriptors are actually matched with current object
# descriptors and which are not
def filterMatching(distance, threshold=0.95):
    return [dist < threshold for dist in distance]


# Given a network output, filter the rcnn detection results so that only 
# "good" objects are preserved
def filterNetworkOutput(network_out):
    clusters, detections, object_descriptors, is_good_cluster = network_out

    new_masks = (detections["masks"])[is_good_cluster].cpu()
    new_boxes = (detections["boxes"])[is_good_cluster].cpu()
    new_scores= (detections["scores"])[is_good_cluster].cpu()
    new_labels= (detections["labels"])[is_good_cluster].cpu()

    good_detections = {
        "masks" : new_masks,
        "boxes" : new_boxes,
        "scores": new_scores,
        "labels": new_labels
    }

    object_descriptors = object_descriptors.cpu()

    return (clusters, good_detections, object_descriptors, is_good_cluster)


# Add an entry into the object library
def addNewObject(library, descriptor, label):
    label_val = label.item()
    if label_val in library:
        library[label_val] = torch.vstack((library[label_val], descriptor))
    else:
        library[label_val] = descriptor[np.newaxis, :]
    return library


# Initialize the object library
def initObjectLibrary(init_output):
    allObjects: Dict[int, torch.Tensor] = dict()
    cluster, detection, descriptor, _ = init_output
    labels = detection["labels"]

    for objcnt in range(len(cluster)):
        obj_label = labels[objcnt]
        obj_descs = descriptor[objcnt]
        
        allObjects = addNewObject(allObjects, obj_descs, obj_label)
    return allObjects


# Update the tensor of all known object descriptors summarized by AirCode
def updateObjectLibrary(new_output, object_library, is_good_matching):
    counter = 0
    _, detection, descriptor, _ = new_output

    for idx, is_good in enumerate(is_good_matching):
        if not is_good:
            counter += 1
            obj_label = detection["labels"][idx]
            obj_descs = descriptor[idx]
            object_library = addNewObject(object_library, obj_descs, obj_label)
    
    if (counter != 0):
        print(f"updateObjectLibrary: {counter} new objects are added")
    
    return object_library


def generateMappingImage(masks, matching):
    mapping = torch.zeros_like(masks[0])
    obj_match = {i : matching[i] for i in range(len(matching))}
    for objcnt in range(masks.shape[0]):
        mapping += masks[objcnt] * obj_match[objcnt]
    return mapping

def generateVisualLabels(clusters, image, good_masks, good_boxes, matching, distance: List[float]):
    masked_image = np.copy(image)
    
    good_masks = good_masks.cpu().numpy().astype(np.uint8)

    obj_match = {i : matching[i] for i in range(len(matching))}

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0)]

    for objcnt in obj_match:
        color = colors[obj_match[objcnt]["id"] % len(colors)]
        box = good_boxes[objcnt]
        mask = good_masks[objcnt]

        overlay_mask(masked_image, mask, [color])
        # mask_size = np.sum(mask)
        match_dist = str(round(distance[objcnt], 3))
        cv2.putText( masked_image, 
                        f"{obj_match[objcnt]['label']}-{obj_match[objcnt]['id']}|{match_dist}",
                        # f"{obj_match[objcnt]['label']}-{obj_match[objcnt]['id']}|{mask_size}", 
                        (box[0], box[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, color)
        
    # showPointClusters(clusters, masked_image)

    return masked_image

def mkdirIfNotExist(*directories: List[str]):
    for directory in directories:
        if (os.path.exists(directory)): return
        os.mkdir(directory)

def main(config):
    sequence_dir = config["data_root"]
    visual_save_dir = os.path.join(config["save_dir"], "visualize")
    label_save_dir  = os.path.join(config["save_dir"], "label")
    obj_save_dir    = os.path.join(config["save_dir"], "object")
    mkdirIfNotExist(visual_save_dir, label_save_dir, obj_save_dir)
    

    point_model, maskrcnn_model, gcn_model = loadModels(config)
    initial = loadImage(config, "000000.png")
    initial_out = network_output(initial, point_model, maskrcnn_model, gcn_model, config)
    initial_out = filterNetworkOutput(initial_out)

    all_object_encodings = initObjectLibrary(initial_out)

    # print(all_object_encodings)

    image_names = os.listdir(sequence_dir)
    image_names.sort()

    for image_name in image_names:
        print("\r" + image_name, end=" ", flush=True)
        target_img = loadImage(config, image_name)
        target_out = network_output(target_img, point_model, maskrcnn_model, gcn_model, config)
        target_out = filterNetworkOutput(target_out)

        # Now, we are use all known objects to match objects in target 
        # new_object_encodings = target_out[2]
        distance, matching = findMatching(target_out, all_object_encodings)

        # If an object in new image is not seen before, add its
        # descriptor to all_object_encodings
        is_good_match        = filterMatching(distance, threshold=.7)
        all_object_encodings = updateObjectLibrary(target_out, all_object_encodings, is_good_match)

        # re-match the objects, should be all good matches now
        distance, matching = findMatching(target_out, all_object_encodings)

        masks           = target_out[1]["masks"]
        boxes           = target_out[1]["boxes"]
        clusters        = target_out[0]
        
        # encode the matching result into a label image
        # map_result = generateMappingImage(masks, matching)
        # save result (an array with 0 as background, n as the object id)
        # np.save(os.path.join(label_save_dir, image_name[:-4]), map_result)
        
        # visualize the matching result
        target_src = loadImage(config, image_name)
        vis_result = generateVisualLabels(clusters, target_src, masks, boxes, matching, distance)
        cv2.imwrite(os.path.join(visual_save_dir, image_name[:-4]) + ".jpg", vis_result)

    
    # print("All object descriptors: ", all_object_encodings.shape)
    # np.save(os.path.join(config["save_dir"], "object_library"), all_object_encodings.cpu());

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

