# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import argparse
import csv
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix

import util.misc as utils
from datasets.spa import make_spa_transforms
from models import build_model


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def get_images(in_path):
    img_files = []
    for dirpath, dirnames, filenames in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == ".jpg" or ext == ".jpeg" or ext == ".gif" or ext == ".png" or ext == ".pgm":
                img_files.append(os.path.join(dirpath, file))

    return img_files


1


def plot_confusion_matrix(y_true, y_pred, classes, path, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = "Normalized confusion matrix"
        else:
            title = "Confusion matrix, without normalization"

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    cm = np.nan_to_num(cm)

    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix")

    cm = np.round(cm, 2).T

    print(cm)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="Predicted",
        xlabel="True",
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_yticklabels(), rotation=90, ha="center", rotation_mode="anchor")
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    # Loop over data dimensions and create text annotations.
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = format(cm[i, j], fmt) if cm[i, j] != 0.00 else ""
            ax.text(j, i, value, ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.xlim(-0.5, len(np.unique(y_true)) - 0.5)
    plt.ylim(len(np.unique(y_true)) - 0.5, -0.5)
    plt.savefig(path)


def get_args_parser():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lr_backbone", default=1e-5, type=float)
    parser.add_argument("--batch_size", default=6, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--lr_drop", default=200, type=int)
    parser.add_argument("--clip_max_norm", default=0.1, type=float, help="gradient clipping max norm")

    # Model parameters
    parser.add_argument(
        "--frozen_weights",
        type=str,
        default=None,
        help="Path to the pretrained model. If set, only the mask head will be trained",
    )
    # * Backbone
    parser.add_argument("--backbone", default="resnet50", type=str, help="Name of the convolutional backbone to use")
    parser.add_argument(
        "--dilation",
        action="store_true",
        help="If true, we replace stride with dilation in the last convolutional block (DC5)",
    )
    parser.add_argument(
        "--position_embedding",
        default="sine",
        type=str,
        choices=("sine", "learned"),
        help="Type of positional embedding to use on top of the image features",
    )

    # * Transformer
    parser.add_argument("--enc_layers", default=6, type=int, help="Number of encoding layers in the transformer")
    parser.add_argument("--dec_layers", default=6, type=int, help="Number of decoding layers in the transformer")
    parser.add_argument(
        "--dim_feedforward",
        default=2048,
        type=int,
        help="Intermediate size of the feedforward layers in the transformer blocks",
    )
    parser.add_argument(
        "--hidden_dim", default=256, type=int, help="Size of the embeddings (dimension of the transformer)"
    )
    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout applied in the transformer")
    parser.add_argument(
        "--nheads", default=8, type=int, help="Number of attention heads inside the transformer's attentions"
    )
    parser.add_argument("--num_queries", default=100, type=int, help="Number of query slots")
    parser.add_argument("--pre_norm", action="store_true")

    # * Segmentation
    parser.add_argument("--masks", action="store_true", help="Train segmentation head if the flag is provided")

    # # Loss
    parser.add_argument(
        "--no_aux_loss",
        dest="aux_loss",
        action="store_false",
        help="Disables auxiliary decoding losses (loss at each layer)",
    )
    # * Matcher
    parser.add_argument("--set_cost_class", default=1, type=float, help="Class coefficient in the matching cost")
    parser.add_argument("--set_cost_bbox", default=5, type=float, help="L1 box coefficient in the matching cost")
    parser.add_argument("--set_cost_giou", default=2, type=float, help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument("--mask_loss_coef", default=1, type=float)
    parser.add_argument("--dice_loss_coef", default=1, type=float)
    parser.add_argument("--bbox_loss_coef", default=5, type=float)
    parser.add_argument("--giou_loss_coef", default=2, type=float)
    parser.add_argument(
        "--eos_coef", default=0.1, type=float, help="Relative classification weight of the no-object class"
    )

    # dataset parameters
    parser.add_argument("--dataset_file", default="spa")
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--data_panoptic_path", type=str)
    parser.add_argument("--remove_difficult", action="store_true")

    parser.add_argument("--output_dir", default="", help="path where to save the results, empty for no saving")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--resume", default="", help="resume from checkpoint")

    parser.add_argument("--thresh", default=0.5, type=float)
    parser.add_argument("--nms", default=1, type=int, help="delete duplicate boxes")
    parser.add_argument("--nms_thresh", default=0.65, type=float)
    parser.add_argument("--save_images", default=0, type=int)
    parser.add_argument("--show_imgs", default=0, type=int)
    parser.add_argument("--json_path", default="")
    parser.add_argument("--iou_thresh", default=0.3, type=float)

    return parser


@torch.no_grad()
def infer(
    images_path,
    model,
    postprocessors,
    device,
    output_path,
    nms,
    nms_thresh,
    save_images,
    show_imgs,
    json_path,
    iou_thresh,
):
    detecciones = []
    clases_reales = []
    clases_predecidas = []

    if json_path != "":
        f = open(json_path)
        labeled_data = json.load(f)
    model.eval()
    duration = 0
    for img_sample in images_path:
        filename = os.path.basename(img_sample)
        print("Processing...{}".format(filename))
        orig_image = Image.open(img_sample)
        w, h = orig_image.size

        if json_path != "":
            labeled_img = next(item for item in labeled_data if item["data_row"]["external_id"] == filename)
            proj_key = list(labeled_img["projects"].keys())[0]
            aux_labels = labeled_img["projects"][proj_key]["labels"][0]["annotations"]["objects"]
            labels = []
            nombres = []
            for label in aux_labels:
                if label["name"] not in ["seg_lin", "seg_veg", "Cap", "Aux"]:
                    if len(label["classifications"]) > 0:
                        nombre = label["name"] + "_" + label["classifications"][0]["radio_answer"]["name"]
                    else:
                        nombre = label["name"]
                    labels.append(label)
                    nombres.append(nombre)

            xmin = [x["bounding_box"]["left"] for x in labels]
            xmax = [x["bounding_box"]["left"] + x["bounding_box"]["width"] for x in labels]
            ymin = [x["bounding_box"]["top"] for x in labels]
            ymax = [x["bounding_box"]["top"] + x["bounding_box"]["height"] for x in labels]
            # ymax = [h - x["bbox"]["top"] for x in labels]
            # ymin = [h - x["bbox"]["top"] - x["bbox"]["height"] for x in labels]
            bboxes_labeled = [[xmin[i], ymin[i], xmax[i], ymax[i]] for i in range(len(xmin))]
            # categories_labeled = [x["title"] for x in labels]
            categories_labeled = nombres

        transform = make_spa_transforms("val")
        dummy_target = {"size": torch.as_tensor([int(h), int(w)]), "orig_size": torch.as_tensor([int(h), int(w)])}
        image, targets = transform(orig_image, dummy_target)
        image = image.unsqueeze(0)
        image = image.to(device)

        conv_features, enc_attn_weights, dec_attn_weights = [], [], []
        hooks = [
            model.backbone[-2].register_forward_hook(lambda self, input, output: conv_features.append(output)),
            model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
                lambda self, input, output: enc_attn_weights.append(output[1])
            ),
            model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
                lambda self, input, output: dec_attn_weights.append(output[1])
            ),
        ]

        start_t = time.perf_counter()
        outputs = model(image)
        end_t = time.perf_counter()

        outputs["pred_logits"] = outputs["pred_logits"].cpu()
        outputs["pred_boxes"] = outputs["pred_boxes"].cpu()

        probas = outputs["pred_logits"].softmax(-1)[0, :, :-1]
        # keep = probas.max(-1).values > 0.85
        keep_index = [x > args.thresh for x in list(probas.max(-1).values > args.thresh)]
        keep = probas.max(-1).values > args.thresh
        clases_aux = probas.max(-1).indices[keep]
        probas_aux = probas.max(-1).values[keep]

        if nms == 1:
            marca = 0
            bboxes = []
            confidence = []
            clases = []
            indices = []
            bboxes_pre = outputs["pred_boxes"][0, keep]
            bboxes_aux = box_cxcywh_to_xyxy(bboxes_pre)
            bboxes_aux_2 = bboxes_aux.tolist()

            probas_aux = list(probas_aux)
            clases_aux = list(clases_aux)
            bboxes_aux = list(bboxes_aux)

            while len(probas_aux) > 0:
                marca = 0
                i = probas_aux.index(max(probas_aux))
                proba_aux = probas_aux[i]
                del probas_aux[i]
                clase_aux = clases_aux[i]
                del clases_aux[i]
                box_aux = bboxes_aux[i]
                del bboxes_aux[i]

                for j, box in enumerate(bboxes):
                    if clases[j] == clase_aux:
                        xx1 = np.maximum(box[0], box_aux[0])
                        yy1 = np.maximum(box[1], box_aux[1])
                        xx2 = np.minimum(box[2], box_aux[2])
                        yy2 = np.minimum(box[3], box_aux[3])
                        w = np.maximum(0, xx2 - xx1 + 0.000001)
                        h = np.maximum(0, yy2 - yy1 + 0.000001)

                        area = (box_aux[2] - box_aux[0]) * (box_aux[3] - box_aux[1])
                        overlap = w * h / area
                        if overlap > nms_thresh:
                            marca = 1
                            break

                if marca == 0:
                    bboxes.append(box_aux)
                    clases.append(clase_aux)
                    confidence.append(proba_aux)
                    indices.append(bboxes_aux_2.index(box_aux.tolist()))

            if len(bboxes) == 0:
                continue

            bboxes = torch.stack(bboxes)
            bboxes_scaled = rescale_bboxes(bboxes_pre[indices], orig_image.size)

        else:
            # print(outputs["pred_boxes"][0, keep])
            bboxes_scaled = rescale_bboxes(outputs["pred_boxes"][0, keep], orig_image.size)
            clases = clases_aux
            confidence = probas_aux

        for hook in hooks:
            hook.remove()

        conv_features = conv_features[0]
        enc_attn_weights = enc_attn_weights[0]
        dec_attn_weights = dec_attn_weights[0].cpu()

        # get the feature map shape
        h, w = conv_features["0"].tensors.shape[-2:]

        if len(bboxes_scaled) == 0:
            continue

        # colores = [(0,0,0),(0,255,75),(145,255,0),(255,145,0),(255,0,72),(218,0,255)]
        # names = ['None', 'GrB', 'GrT', 'GrL', 'PCo', 'Bac']
        # numbers = [5, 2, 0, 1, 3, 4]
        colores = [
            (218, 0, 255),
            (0, 255, 75),
            (255, 145, 0),
            (145, 255, 0),
            (255, 0, 72),
            (255, 0, 72),
            (255, 0, 72),
            (255, 0, 72),
            (255, 0, 72),
            (0, 0, 0),
        ]
        names = [
            "Bac",
            "GrB",
            "GrL",
            "GrT",
            "PCo_Low-longitudinal",
            "PCo_Low-random",
            "PCo_Medium",
            "PCo_High",
            "Pel",
            "background",
        ]
        numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        img = np.array(orig_image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if json_path != "":
            indices_aux = list(np.arange(len(categories_labeled)))

        for idx, box in enumerate(bboxes_scaled):
            clase = clases[idx]
            bbox = box.cpu().data.numpy()
            bbox = bbox.astype(np.int32)
            dictio = {
                "file": filename,
                "model": "detr",
                "xmin": max(bbox[0], 0),
                "ymin": max(bbox[1], 0),
                "xmax": bbox[2],
                "ymax": bbox[3],
                "confidence": float(confidence[idx]),
                "class": numbers[clase],
                "name": names[clase],
            }
            detecciones.append(dictio)

            if json_path != "":
                xx1 = np.maximum(xmin, bbox[0])
                yy1 = np.maximum(ymin, bbox[1])
                xx2 = np.minimum(xmax, bbox[2])
                yy2 = np.minimum(ymax, bbox[3])
                w2 = np.maximum(0, xx2 - xx1 + 0.000001)
                h2 = np.maximum(0, yy2 - yy1 + 0.000001)
                intersections = np.array(w2) * np.array(h2)
                union1 = (np.array(xmax) - np.array(xmin)) * (np.array(ymax) - np.array(ymin))
                union2 = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                unions = union1 - intersections + union2
                iou = list(intersections / unions)
                if len(iou) == 0:
                    clases_reales.append(9)
                    clases_predecidas.append(numbers[clase])
                else:
                    indice_aux = iou.index(max(iou))

                    clases_predecidas.append(numbers[clase])
                    if iou[indice_aux] > iou_thresh:
                        clases_reales.append(numbers[names.index(categories_labeled[indice_aux])])
                        if indice_aux in indices_aux:
                            indices_aux.remove(indice_aux)

                    else:
                        clases_reales.append(9)

            bbox = np.array(
                [
                    [bbox[0], bbox[1]],
                    [bbox[2], bbox[1]],
                    [bbox[2], bbox[3]],
                    [bbox[0], bbox[3]],
                ]
            )
            bbox = bbox.reshape((4, 2))
            cv2.polylines(img, [bbox], True, colores[clase], 2)

        if json_path != "":
            for indice_aux in indices_aux:
                clases_predecidas.append(9)
                clases_reales.append(numbers[names.index(categories_labeled[indice_aux])])

        if save_images == 1:
            cv2.imwrite(os.path.join(output_path, filename), img)

        if show_imgs == 1:
            cv2.imshow("img", img)
            cv2.waitKey()
        infer_time = end_t - start_t
        duration += infer_time
        print("Processing...{} ({:.3f}s)".format(filename, infer_time))

    if output_path != "":
        myFile = open(os.path.join(output_path, "anotations.csv"), "w", newline="")
        writer = csv.writer(myFile, delimiter=";")
        writer.writerow(list(detecciones[0].keys()))
        for dictionary in detecciones:
            writer.writerow(dictionary.values())
        myFile.close()

        if json_path != "":
            conf_out = os.path.join(output_path, "detr_conf.png")
            for k in clases_reales:
                if isinstance(k, str):
                    print(k, type(k))
            for l in clases_predecidas:
                if isinstance(l, str):
                    print(l, type(l))
            plot_confusion_matrix(
                clases_reales,
                clases_predecidas,
                path=conf_out,
                normalize=True,
                classes=[
                    "Bac",
                    "GrB",
                    "GrL",
                    "GrT",
                    "PCoLL",
                    "PCoLR",
                    "PCoM",
                    "PCoH",
                    "Pel",
                    "background",
                ],
                title="Normalized confusion matrix",
            )

    avg_duration = duration / len(images_path)
    print("Avg. Time: {:.3f}s".format(avg_duration))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("DETR training and evaluation script", parents=[get_args_parser()])
    args = parser.parse_args()

    # args.resume = "/home/aguia/train_models/output_24-06-14/detr/checkpoint.pth"
    # args.data_path = "/home/aguia/train_models/datasets/lbd_2406/images/val/"
    # args.json_path = "/mnt/drive/datos/labeled_datasets/2405/recortes/labels.json"
    # args.output_dir = "/home/aguia/train_models/output_24-06-14/detr/"
    # args.device = "cuda"
    # args.nms = 0
    # args.iou_thresh = 0.4

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    model, _, postprocessors = build_model(args)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    model.to(device)
    image_paths = get_images(args.data_path)

    infer(
        image_paths,
        model,
        postprocessors,
        device,
        args.output_dir,
        args.nms,
        args.nms_thresh,
        args.save_images,
        args.show_imgs,
        args.json_path,
        args.iou_thresh,
    )
