
import numpy as np
from PIL import Image
import torch
import torchvision
import cv2
import os
import json
from utils.basic_utils import overlay_masks_detectron_style
from segment_anything import sam_model_registry, SamPredictor

def get_points(binary_mask):
    y_indices, x_indices = np.where(binary_mask > 0)

    # Calculate x_min and x_max
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()
    x_center = (x_min + x_max) // 2  # Use integer division for center line
    y_center = (y_min + y_max) // 2 
    center_point = [x_center, y_center]

    width = x_max - x_min
    height = y_max - y_min
    if width > height:
        # Filter points where x equals x_center
        center_x_mask = x_indices == x_center
        y_values_at_center = y_indices[center_x_mask]

        # Find the lowest and highest y values at the center line
        y_lowest = y_values_at_center.min()
        y_highest = y_values_at_center.max()

        # Output the coordinates
        lowest_point = [x_center, y_lowest]
        highest_point = [x_center, y_highest]
    else:
        # Filter points where y equals y_center
        center_y_mask = y_indices == y_center
        x_values_at_center = x_indices[center_y_mask]
        
        # Find the lowest and highest x values at the center line
        x_lowest = x_values_at_center.min()
        x_highest = x_values_at_center.max()
        # Output the coordinates
        lowest_point = [x_lowest, y_center]
        highest_point = [x_highest, y_center]

    points = np.array([lowest_point, center_point, highest_point])
    return points

def segment(sam_predictor, image, xyxy):
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, _, _ = sam_predictor.predict(
            box=box,
            multimask_output=False
        )
        refined_mask = masks[0]
        points = get_points(refined_mask)
        refined_mask2, _, _ = sam_predictor.predict(
            point_coords=points,
            point_labels=np.ones(len(points)),
            mask_input=None,
            multimask_output=False,
        )
        refined_mask = np.logical_or(refined_mask, refined_mask2[0])
        result_masks.append(refined_mask)
    return np.array(result_masks)

def ground_segment(grounding_dino_model, sam_predictor, input_image, objects_to_seg, box_threshold, text_threshold):
    NMS_THRESHOLD = 0.8
    
    # detect objects
    detections = grounding_dino_model.predict_with_classes(
        image=input_image,
        classes=objects_to_seg,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )
    
    # NMS post process
    nms_idx = torchvision.ops.nms(
        torch.from_numpy(detections.xyxy), 
        torch.from_numpy(detections.confidence), 
        NMS_THRESHOLD
    ).numpy().tolist()

    detections.xyxy = detections.xyxy[nms_idx]
    detections.confidence = detections.confidence[nms_idx]
    detections.class_id = detections.class_id[nms_idx]
    
    # Prompting SAM with detected boxes
    def sam_segment(sam_predictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = sam_predictor.predict(
                box=box,
                multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)

    # convert detections to masks
    detections.mask = sam_segment(
        sam_predictor=sam_predictor,
        image=cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB),
        xyxy=detections.xyxy
    )
    
    # collect the mask for each label
    labels = {}
    for i, (_, _, confidence, class_id, *_) in enumerate(detections):
        label_name = objects_to_seg[class_id]
        if labels.get(label_name) is None:
            labels[label_name] = []
        labels[label_name].append(i)
    
    # get the masks and bboxes
    masks = []
    bboxes = []
    for name in objects_to_seg:
        if name in labels:
            mask = np.any(detections.mask[labels[name]],axis=0)
            bbox = mask_to_normalized_bbox(mask)
            masks.append(mask)
            bboxes.append(bbox)
        else:
            masks.append(np.zeros((512,512),dtype=bool))
            bboxes.append([0,0,0,0])
    return masks, bboxes

def mask_to_normalized_bbox(mask):
    # Find the rows and columns where the mask is non-zero
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    # Find the minimum and maximum row and column indices
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Normalize the coordinates by the size of the mask
    height, width = mask.shape
    x1, x2 = cmin / width, cmax / width
    y1, y2 = rmin / height, rmax / height
    return (x1, y1, x2, y2)

def initialize_sam_model(device):
    # Segment-Anything checkpoint
    SAM_ENCODER_VERSION = "vit_h"
    SAM_CHECKPOINT_PATH = "checkpoints/sam_vit_h_4b8939.pth"

    # Building SAM Model and SAM Predictor
    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
    sam.to(device)
    sam_predictor = SamPredictor(sam)
    return sam_predictor

def initialize_ground_sam_models(device):
    from groundingdino.util.inference import Model
    from segment_anything import sam_model_registry, SamPredictor
    
    # GroundingDINO config and checkpoint
    GROUNDING_DINO_CONFIG_PATH = "utils/Segment/GroundingDINO_SwinT_OGC.py"
    GROUNDING_DINO_CHECKPOINT_PATH = "./checkpoints/groundingdino_swint_ogc.pth"

    # Segment-Anything checkpoint
    SAM_ENCODER_VERSION = "vit_h"
    SAM_CHECKPOINT_PATH = "./checkpoints/sam_vit_h_4b8939.pth"

    # Building GroundingDINO inference model
    grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

    # Building SAM Model and SAM Predictor
    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)
    return grounding_dino_model, sam_predictor

def construct_node_masks(image_path, basic_SG_path, out_dir = "outputs"):
    input_image = Image.open(image_path)
    with open(basic_SG_path) as f:
        basic_SG = json.load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    grounding_dino_model, sam_predictor = initialize_ground_sam_models(device)
    
    # objects to detect
    objects_to_seg = basic_SG['objects']
    input_image = np.array(input_image)
    
    # initialize model
    with torch.no_grad():
        masks, bboxes = ground_segment(grounding_dino_model, sam_predictor, input_image, objects_to_seg, box_threshold=0.25, text_threshold=0.25)

    # save the input image
    save_name = os.path.join(out_dir, f"input.png")
    Image.fromarray(input_image).save(save_name)

    # save the masks
    for name, mask in zip(objects_to_seg, masks):
        save_name = os.path.join(out_dir, f"{name}_mask.png")
        Image.fromarray(mask).save(save_name)

    # visualize the mask
    seg_vis_image = overlay_masks_detectron_style(input_image, masks)
    save_name = os.path.join(out_dir, "initial_seg.png")
    seg_vis_image.save(save_name)
    
    # save the scene_graph_dict
    basic_SG["bboxes"] = bboxes
    with open(basic_SG_path, 'w') as f:
        json.dump(basic_SG, f, indent=4)
    