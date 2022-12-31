import cv2
import matplotlib.pyplot as plt

from utils import cv2_util

def showFeaturePts(pts_output, template):
    points = pts_output[0]["points"]

    for ptcnt in range(points.shape[0]):
        cv2.circle(template, (points[ptcnt, 1], points[ptcnt, 0]), radius=1, color=(
            0, 255, 255), thickness=-1)
    
    cv2.imwrite("feature_points.jpg", template)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(template)
    plt.show()


def showDetectionBoxes(detections, template):
    detection = detections[0]
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    for boxcnt in range(detection["labels"].shape[0]):
        color = colors[boxcnt % len(colors)]
        box = detection["boxes"][boxcnt]
        label = detection["labels"][boxcnt]
        cv2.rectangle(template, (box[0], box[1]),
                      (box[2], box[3]), color, 1)
        cv2.putText(template, str(
            int(label)), (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
    
    cv2.imwrite("detection_boxes.jpg", template)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(template)
    plt.show()


def showPointClusters(batch_points, template):
    img_points = batch_points[0]
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    for objcnt in range(len(img_points)):
        obj_points = img_points[objcnt]
        for ptcnt in range(obj_points.shape[0]):
            cv2.circle(template, (obj_points[ptcnt, 1], obj_points[ptcnt, 0]),
                       radius=3, color=colors[objcnt % len(colors)], thickness=-1)
    
    cv2.imwrite("point_clusters.jpg", template)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(template)
    plt.show()


def overlay_mask(image, masks, colors):
    """
    Adds the instances contours for each predicted object.
    Each label has a different color.

    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `mask` and `labels`.
    """
    for mask, color in zip(masks, colors):
        contours, hierarchy = cv2_util.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        image = cv2.drawContours(image, contours, -1, color, 2)

    composite = image

    return composite
