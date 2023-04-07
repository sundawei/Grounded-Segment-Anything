import argparse
import os
import copy

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np
import matplotlib.pyplot as plt

sam_checkpoint = "sam_vit_h_4b8939.pth"
predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint))

def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def show_mask(image, mask, random_color=False):
    if random_color:
        color = np.random.random(3)
    else:
        color = np.array([30/255, 144/255, 255/255])

    h, w = mask.shape[-2:]
    mask_image = np.repeat(mask.reshape(h, w, 1), 3, axis=2) * color.reshape(1, 1, -1)
    mask_image = (mask_image * 255).astype(np.uint8)
    return cv2.addWeighted(image, 1, mask_image, 0.6, 0)


def show_box(image, box, label):
    x0, y0, x1, y1 = box.astype(int)
    image = cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    (label_width, label_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
    image = cv2.rectangle(image, (x0, y0 - label_height - baseline), (x0 + label_width, y0), (0, 255, 0), -1)
    image = cv2.putText(image, label, (x0, y0 - baseline), font, font_scale, (0, 0, 0), font_thickness)
    return image

from flask import Flask, request, redirect, url_for, render_template, send_from_directory,send_file
import uuid
import shutil

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"

@app.route('/', methods=['GET', 'POST'])
def upload_predict(model=None):
    if request.method == "POST":
        image_file = request.files.get("file")
        text_prompt = request.form.get("text_prompt")
        if not image_file or not text_prompt:
            return "Missing file or text prompt", 400

        image_pil = Image.open(image_file).convert("RGB")
        image_pil.save("input_image.jpg")
        image_pil, image = load_image("input_image.jpg")
        boxes_filt, pred_phrases = get_grounding_output(
            model, image, text_prompt, box_threshold, text_threshold, device=device
        )

        predictor.set_image(np.array(image_pil))
        H, W = image_pil.size[1], image_pil.size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, np.array(image_pil).shape[:2])

        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        output_image = np.array(image_pil)
        for mask in masks:
            output_image = show_mask(output_image, mask.cpu().numpy(), random_color=True)
        for box, label in zip(boxes_filt, pred_phrases):
            output_image = show_box(output_image, box.numpy(), label)

        output_image_filename = "output_image.jpg"
        cv2.imwrite(output_image_filename, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
        return send_file(output_image_filename, mimetype='image/jpeg')

    return '''
    <!doctype html>
    <title>Grounded-Segment-Anything Demo</title>
    <h1>Upload an image and enter a text prompt</h1>
    <form method=post enctype=multipart/form-data>
    <label for="text_prompt">Text Prompt:</label>
    <input type=text id="text_prompt" name="text_prompt"><br><br>
    <input type=file id="file" name="file">
    <input type=submit value=Upload>
    </form>
    '''


@app.route("/result/<image_id>")
def result(image_id):
    return send_from_directory(os.path.join("outputs", image_id), "grounded_sam_output.jpg")

def main(
    config_file,
    grounded_checkpoint,
    sam_checkpoint,
    input_image,
    text_prompt,
    output_dir,
    box_threshold,
    text_threshold,
    device,
):
    # Load image
    image_pil, image = load_image(input_image)

    # Load model
    model = load_model(config_file, grounded_checkpoint, device=device)

    # Run grounding dino model
    boxes_filt, pred_phrases = get_grounding_output(
        model, image, text_prompt, box_threshold, text_threshold, device=device
    )

    # Initialize SAM
    predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint))
    image = cv2.imread(input_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2])

    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )

    # Draw output image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    for box, label in zip(boxes_filt, pred_phrases):
        show_box(box.numpy(), plt.gca(), label)
    plt.axis("off")
    plt.savefig(os.path.join(output_dir, "grounded_sam_output.jpg"), bbox_inches="tight")

if __name__ == "__main__":
    # Set your configuration and checkpoint paths
    config_file = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    grounded_checkpoint = "groundingdino_swint_ogc.pth"
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    box_threshold = 0.3
    text_threshold = 0.25
    device = "cpu"
    # Load the model before starting the app
    model = load_model(config_file, grounded_checkpoint, device=device)

    # Pass the model as an argument to upload_predict
    app.view_functions['upload_predict'] = lambda: upload_predict(model=model)

    app.run(host="0.0.0.0", port=5001 ,debug=True, threaded=False)