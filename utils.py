import numpy as np
import cv2 
import tensorflow as tf

def get_clean_name(string):
    if "depth" in string.lower() and "kernel" in string.lower():
        return string.split('/')[0] + '/' + 'Kernel'
    elif "depth" in string.lower() and "bias" in string.lower():
        return string.split('/')[0] + '/' + 'Bias'
    elif "conv2d" in string.lower() and "kernel" in string.lower():
        return string.split('/')[0] + '/' + 'Kernel'
    elif "conv2d" in string.lower() and "bias" in string.lower():
        return string.split('/')[0] + '/' + 'Bias'
    elif "lu" in string.lower():
        return string.split('/')[0] + '/' + "Alpha"
    elif "kernel" in string.lower():
        return string.split('/')[0] + '/' + "Kernel"
    elif "bias" in string.lower():
        return string.split('/')[0] + '/' + "Bias"
    else:
        raise ValueError("Input string not understood")
        
exception_mapping = {
    "depthwise_conv2d_18/depthwise_kernel" : "depthwise_conv2d_22/Kernel",
    "depthwise_conv2d_18/bias": "depthwise_conv2d_22/Bias",
    "conv2d_21/kernel" : "conv2d_27/Kernel",
    "conv2d_21/bias": "conv2d_27/Bias",
    "p_re_lu_20/alpha": "p_re_lu_25/Alpha"
}

def restore_variables(model,tf_lite_mapping, data_format):
    channels_first = True if data_format == "channels_first" else False
    restored = 0
    total_params = 0
    for var in model.variables:
        try:
            name = get_clean_name(var.name)
            weight = tf_lite_mapping[name]
        except KeyError:
            map_string = exception_mapping[var.name[:-2]]
            name = get_clean_name(map_string)
            weight = tf_lite_mapping[name]
        if weight.ndim == 4:
            weight = np.transpose(weight, (1,2,3,0)) # conv transpose
        elif weight.ndim ==3:
            if channels_first: weight = np.transpose(weight, (2, 0, 1)) #prelu_transpose
        total_params += np.product(weight.shape)
        var.assign(weight)
        print("{} assinged with {}".format(var.name, name))
        restored += 1
    print("Restored {} variables from tflite file".format(restored))
    print("Restore {} float values".format(total_params))
    
def xywh_to_tlbr(boxes, y_first=False):
    """
    boxes - (N, 4)
    """
    final_boxes = boxes.copy()
    if not y_first:
        final_boxes[:, 0:2] = np.clip(boxes[:, 0:2] - (boxes[:, 2:4]/2), 0, None) #clipping at 0 since image dim starts at 0
        final_boxes[:, 2:4] = boxes[:, 0:2] + (boxes[:, 2:4]/2)
    else:
        final_boxes[:, 0:2] = np.clip(boxes[:, [1,0]] - (boxes[:, [3,2]]/2), 0, None)
        final_boxes[:, 2:4] = boxes[:, [1,0]] + (boxes[:, [3,2]]/2)
    return final_boxes
    
def create_letterbox_image(frame, dim):
    h, w = frame.shape[0:2]
    scale = min(dim/h, dim/w)
    nh, nw = int(scale*h), int(scale*w)
    resized = cv2.resize(frame, (nw, nh))
    new_image = np.zeros((dim, dim, 3), np.uint8) 
    new_image.fill(256)
    dx = (dim-nw)//2
    dy = (dim-nh)//2
    new_image[dy:dy+nh, dx:dx+nw,:] = resized
    return new_image

#takes the letterbox dimensions and the original dimensions to map the results in letterbox image coordinates
#to original image coordinates
def convert_to_orig_points(results, orig_dim, letter_dim):
    if results.ndim == 1: np.expand_dims(results, 0)
    inter_scale = min(letter_dim/orig_dim[0], letter_dim/orig_dim[1])
    inter_h, inter_w = int(inter_scale*orig_dim[0]), int(inter_scale*orig_dim[1])
    offset_x, offset_y = (letter_dim - inter_w)/2.0/letter_dim, (letter_dim - inter_h)/2.0/letter_dim
    scale_x, scale_y = letter_dim/inter_w, letter_dim/inter_h
    results[:, 0:2] = (results[:, 0:2] - [offset_x, offset_y]) * [scale_x, scale_y]
    results[:, 2:4] =  results[:, 2:4] * [scale_x, scale_y]
    results[:, 4:16:2] = (results[:, 4:16:2] - offset_x) * scale_x
    results[:, 5:17:2] = (results[:, 5:17:2] - offset_y) * scale_y
    #converting from 0-1 range to (orign_dim) range
    results[:, 0:16:2] *= orig_dim[1]
    results[:, 1:17:2] *= orig_dim[0]
    
    return results.astype(np.int32)

def process_detections(results, orig_dim, max_boxes=5, score_threshold=0.75, iou_threshold=0.5, pad_ratio=0.5):
    box_tlbr = xywh_to_tlbr(results[:, 0:4], y_first=True)
    out_boxes = tf.image.non_max_suppression(box_tlbr, results[:, -1], max_boxes,
                                             score_threshold=score_threshold, iou_threshold=iou_threshold)
    filter_boxes = results[out_boxes.numpy(), :-1]
    orig_points = convert_to_orig_points(filter_boxes, orig_dim, 128)
    landmarks_xywh = orig_points.copy()
    landmarks_xywh[:, 2:4] += (landmarks_xywh[:, 2:4] * pad_ratio).astype(np.int32) #adding some padding around detection for landmark detection step.
    landmarks_xywh[:, 1:2] -= (landmarks_xywh[:, 3:4]*0.08).astype(np.int32) #adjusting center_y since the detector outputs boxes from forehead and to account for that bias
    final_boxes = xywh_to_tlbr(orig_points).astype(np.int32)
    return final_boxes, landmarks_xywh

def get_landmarks_crop(orig_frame, landmarks_proposals, input_dim):
    landmarks_proposals = xywh_to_tlbr(landmarks_proposals).astype(np.int32)
    proposals = []
    for prop in landmarks_proposals:
        proposals.append(cv2.cvtColor(cv2.resize(orig_frame[prop[1]:prop[3], prop[0]:prop[2], :], (input_dim[1], input_dim[0])), cv2.COLOR_BGR2RGB))
    proposals = np.array(proposals).astype(np.float32)/127.5 - 1
    return proposals

def process_landmarks(landmarks_result, landmarks_proposals, orig_dim, land_dim):
    landmarks_result = np.reshape(landmarks_result, (-1, 468, 3))[:, :, :2]
    proposal_orig_scale = landmarks_proposals[:, 2:4] / land_dim
    landmarks_result[:, :, :2] *= proposal_orig_scale[:, np.newaxis, :]
    landmarks_prop_tlwh = xywh_to_tlbr(landmarks_proposals)
    landmarks_result += landmarks_prop_tlwh[:,np.newaxis,  0:2]
    landmarks_result = landmarks_result.astype(np.int32)
    return landmarks_result
        