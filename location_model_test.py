"""
run inference on a local model
"""
import torch
import torchvision.transforms as T # noqa
from PIL import Image
from loguru import logger
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor # noqa
from torchvision.utils import draw_bounding_boxes
import json
import os

local_model_path = "trained_models/ddr_location_detect.pth"
coco_classes_path = "trained_models/CoCoClasses.json"


def load_model(model_path):
    model = fasterrcnn_resnet50_fpn(weight=None,
                                    weights_backbone=None)
    num_classes = 6
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model


def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img.resize(size=(640, 640))
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    return img.unsqueeze(0)


def load_coco_classes(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return {category['id']: category['name'] for category in data['categories']}


def get_label_name(label_id, coco_classes):
    return coco_classes.get(label_id, f"Unknown label: {label_id}")


def run_inference_max_score(model, image_tensor):
    with torch.no_grad():
        predictions = model(image_tensor)

    print(predictions)

    try:
        max_score_index = torch.argmax(predictions[0]["scores"]).item()
        max_score = predictions[0]["scores"][max_score_index].item()
    except IndexError as e:
        logger.error(f"Error during inference: {e}")
        return None

    return [{
        "label": predictions[0]["labels"][max_score_index].item(),
        "bbox": predictions[0]["boxes"][max_score_index].tolist(),
        "score": max_score
    }]


def crop_and_save_image(image_tensor, bbox, input_filename):
    """
    Crop the image based on the bounding box and save it.
    Add 10% to the height of the cropped area.
    """
    # Convert tensor to PIL Image
    image = T.ToPILImage()(image_tensor.squeeze(0))
    
    # Unpack bbox coordinates
    x1, y1, x2, y2 = bbox
    
    # Calculate 10% of the height
    height_increase = (y2 - y1) * 0.15
    
    # Adjust y coordinates
    y1 = max(0, y1 - height_increase / 2)
    y2 = min(image.height, y2 + height_increase / 2)
    
    # Crop the image
    cropped_image = image.crop((x1, y1, x2, y2))
    
    # Save the cropped image
    output_path = os.path.join('saved_images', f'{input_filename}_cropped.jpg')
    cropped_image.save(output_path)
    logger.info(f"Cropped image saved as: {output_path}")


def run_inference_threshold(model, image_tensor, confidence_threshold=0.4):
    with torch.no_grad():
        predictions = model(image_tensor)

    try:
        # Get indices of all predictions above threshold
        scores = predictions[0]["scores"]
        above_threshold = scores >= confidence_threshold
        
        # If no predictions above threshold, return empty list
        if not torch.any(above_threshold):
            return []

        return [{
            "label": predictions[0]["labels"][i].item(),
            "bbox": predictions[0]["boxes"][i].tolist(),
            "score": predictions[0]["scores"][i].item()
        } for i in range(len(scores)) if scores[i] >= confidence_threshold]
        
    except IndexError as e:
        logger.error(f"Error during inference: {e}")
        return []


def main_inference(test_image_path):
    """
    Main inference loop function
    """
    # Load model and COCO classes
    local_model = load_model(model_path=local_model_path)
    coco_classes = load_coco_classes(coco_classes_path)

    # Preprocess image
    local_image_tensor = preprocess_image(image_path=test_image_path)

    # Get the prediction with the highest score
    model_predictions = run_inference_threshold(model=local_model,
                                                image_tensor=local_image_tensor,
                                                confidence_threshold=0.4)

    if model_predictions is None:
        logger.error("No predictions made")
        return None, None

    # Map numeric label to name
    for pred in model_predictions:
        pred["label_name"] = get_label_name(pred["label"], coco_classes)

    box_tensors = [torch.tensor(pred["bbox"]) for pred in model_predictions]

    # plot bounding boxes
    if len(model_predictions) > 0:
        boxes = torch.stack(box_tensors, dim=0)

        result_image = draw_bounding_boxes(local_image_tensor.squeeze(0).mul(255).byte(),
                                           boxes=boxes,
                                           labels=[pred["label_name"] for pred in model_predictions],
                                           colors="red",
                                           width=4)

        # Get the input image filename without extension
        input_filename = os.path.splitext(os.path.basename(test_image_path))[0]
        
        # Save image with the input filename
        result_image_pil = T.ToPILImage()(result_image)
        output_path = os.path.join('saved_images', f'{input_filename}_annotated.jpg')
        result_image_pil.save(output_path)
        logger.info(f"Annotated image saved as: {output_path}")

        # Crop and save the image based on the bounding box
        crop_and_save_image(local_image_tensor, model_predictions[0]["bbox"], input_filename)
        
        # Return label name and coordinates
        return model_predictions

    else:
        logger.error("No predictions made")
        return None
