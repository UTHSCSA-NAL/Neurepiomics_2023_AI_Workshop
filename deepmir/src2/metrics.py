import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K

def dice_coef(y_true, y_pred):
    smooth = 1e-7
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * K.round(y_pred_f))
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_multilabel(y_true, y_pred):
    numLabels = K.int_shape(y_pred)[-1]

    dice = 0
    for index in range(numLabels):
        dice += dice_coef(y_true[:, :, :, index], y_pred[:, :, :, index])
    return dice / numLabels

def dice_loss(y_true, y_pred):
    return 1 - dice_coef_multilabel(y_true, y_pred)

def iou(y_true, y_pred, label: int):
    """
    Return the Intersection over Union (IoU) for a given label.
    Args:
        y_true: the expected y values as a one-hot
        y_pred: the predicted y values as a one-hot or softmax output
        label: the label to return the IoU for
    Returns:
        the IoU for the given label
    """
    # Extract the label values using the argmax operator, then
    # calculate equality of the predictions and truths to the label
    y_true = K.cast(K.equal(K.argmax(y_true), label), K.floatx())
    y_pred = K.cast(K.equal(K.argmax(y_pred), label), K.floatx())

    # Calculate the |intersection| (AND) of the labels
    intersection = K.sum(y_true * y_pred)

    # Calculate the |union| (OR) of the labels
    union = K.sum(y_true) + K.sum(y_pred) - intersection

    # Avoid divide by zero - if the union is zero, return 1
    # Otherwise, return the intersection over union
    ret = K.switch(K.equal(union, 0), 1.0, intersection / union)
    return ret

def mean_iou(y_true, y_pred):
    """
    Return the Intersection over Union (IoU) score.
    Args:
        y_true: the expected y values as a one-hot
        y_pred: the predicted y values as a one-hot or softmax output
    Returns:
        the scalar IoU value (mean over all labels)
    """
    # Get the number of labels to calculate IoU for
    num_labels = K.int_shape(y_pred)[-1]
    total_iou = 0

    # Iterate over labels to calculate IoU for
    for label in range(1, num_labels):
        total_iou += iou(y_true, y_pred, label)
    
    # Divide total IoU by the number of labels to get mean IoU
    return total_iou / num_labels
