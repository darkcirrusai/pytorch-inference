"""
run inference on a local model
"""
import torch
import torchvision.transforms as T # noqa
from PIL import Image
from loguru import logger
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor # noqa

local_model_path = "trained_models/location_detect.pth"


def load_model(model_path):
    model = fasterrcnn_resnet50_fpn(weight=False)
    num_classes = 3
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img.resize(size=(640, 640))
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    return img.unsqueeze(0)


def run_inference(model, image_tensor, threshold=0.4):
    with torch.no_grad():
        predictions = model(image_tensor)
    filtered_predictions = []
    for i in range(len(predictions[0]["labels"])):
        if predictions[0]["scores"][i] > threshold:
            filtered_predictions.append({
                "label": predictions[0]["labels"][i].item(),
                "bbox": predictions[0]["boxes"][i].tolist(),
                "score": predictions[0]["scores"][i].item()
            })
    return filtered_predictions


def run_inference_max_score(model,image_tensor):
    with torch.no_grad():
        predictions = model(image_tensor)

    max_score_index = torch.argmax(predictions[0]["scores"]).item()
    max_score = predictions[0]["scores"][max_score_index].item()

    return [{
        "label": predictions[0]["labels"][max_score_index].item(),
        "bbox": predictions[0]["boxes"][max_score_index].tolist(),
        "score": max_score
    }]


def main_inference(test_image_path):
    """
    Main inference loop function
    """
    # Load model
    local_model = load_model(model_path=local_model_path)

    local_image_tensor = preprocess_image(image_path=test_image_path)

    # Get the prediction with the highest score
    model_predictions = run_inference_max_score(model=local_model,
                                                image_tensor=local_image_tensor)

    box_coordinates = []

    if len(model_predictions) == 0:
        logger.error("No predictions made")
    else:
        for pred in model_predictions:
            box_tensor = torch.tensor(pred["bbox"])
            # Convert tensor to list
            box_list = box_tensor.tolist()
            box_coordinates.append(box_list)

    # Return list of coordinates
    return box_coordinates[0]
