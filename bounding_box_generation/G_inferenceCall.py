##############################################################################
#   Ce fichier permet de setup biomedParse et de récupérer les masques       #
#   de tous les patients et les masques gt associés. Structure de dossier    #
#   identique à celle décrite dans le Notebook poumon.                       #
##############################################################################

from PIL import Image
import pydicom
import numpy as np
import cv2
import torch
import json
import argparse
import torch
import glob
from math import dist
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from modeling.BaseModel import BaseModel
from modeling import build_model
from utilities.distributed import init_distributed
from utilities.arguments import load_opt_from_config_files
from utilities.constants import BIOMED_CLASSES

from inference_utils.processing_utils import read_dicom
from inference_utils.processing_utils import read_png
from inference_utils.inference import interactive_infer_image
from inference_utils.inference import interactive_infer_image_all
from inference_utils.output_processing import check_mask_stats
import pydicom
import numpy as np
import cv2

from skimage.transform import resize

import os
os.environ['HF_HUB_DISABLE_SSL_VERIFICATION'] = '1'

import urllib3
urllib3.disable_warnings()

import ssl
import requests


def inference_dicom(file_path, text_prompts, model, is_CT, site=None):
    image = read_dicom(file_path, is_CT, site=site)
    pred_mask = interactive_infer_image(model, Image.fromarray(image), text_prompts)
    # pred_mask = interactive_infer_image_all(model, Image.fromarray(image), 'CT-Chest')
    return image, pred_mask

def get_target_segment_number(seg, keyword="neoplasm"):
    for s in seg.SegmentSequence:
        if keyword.lower() in s.SegmentLabel.lower():
            return s.SegmentNumber
    raise ValueError(f"Aucun segment contenant '{keyword}' trouvé dans le fichier de segmentation.")

text_prompt = ['tumor']

def getModel() :
    # Pour forcer le trust du certificat
    ssl._create_default_https_context = ssl._create_unverified_context

    # Charger les options depuis le fichier de configuration
    opt = load_opt_from_config_files(["configs/biomedparse_inference.yaml"])

    # Initialiser la distribution
    opt = init_distributed(opt)

    # Chemin vers les poids pré-entraînés
    pretrained_pth = 'pretrained/biomedparse_v1.pt'

    # Vérifier si CUDA est disponible et définir le périphérique
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # # Créer le modèle avec les options et la fonction de construction
    build = build_model(opt, device=device)
    model = BaseModel(opt, build)

    model = model.from_pretrained(pretrained_pth).eval()

    model.to(device)

    with torch.no_grad():
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(BIOMED_CLASSES + ["background"], is_eval=True)
    
    return model

def getMasks(model):
    predicted_masks = []
    ground_truth_masks = []

    patients = sorted(os.listdir("./test/dcm/"))

    for patient in patients :
        patPredMasks = []
        patGtMasks = []
        images_dir = f"./test/dcm/{patient}/0/"
        seg_path = f"./test/dcm/{patient}/1/1-1.dcm"

        dicom_files = sorted(os.listdir(images_dir))
        image_slices = [pydicom.dcmread(os.path.join(images_dir, f)) for f in dicom_files]

        seg = pydicom.dcmread(seg_path)
        target_segment_number = get_target_segment_number(seg)

        #  mapping UID / masque GT
        masks = {}
        for i, f in enumerate(seg.PerFrameFunctionalGroupsSequence):
            if int(f.SegmentIdentificationSequence[0].ReferencedSegmentNumber) == target_segment_number:
                uid = f.DerivationImageSequence[0].SourceImageSequence[0].ReferencedSOPInstanceUID
                masks[uid] = seg.pixel_array[i]

        for dicom_file, img_dcm in zip(dicom_files, image_slices):
            sop_uid = img_dcm.SOPInstanceUID
            img_path = os.path.join(images_dir, dicom_file)

            image, pred_mask = inference_dicom(img_path, text_prompt, model, is_CT=True, site='lung')
            pred_mask = np.squeeze(pred_mask)

            # Ajouter les masques seulement si le GT existe pour cette UID + resize
            if sop_uid in masks:
                gt_mask = masks[sop_uid]

                if pred_mask.shape != gt_mask.shape:
                    pred_mask = resize(pred_mask, gt_mask.shape, anti_aliasing=False)

                patPredMasks.append(pred_mask)
                patGtMasks.append(gt_mask)

        predicted_masks.append(patPredMasks)
        ground_truth_masks.append(patGtMasks)

    return predicted_masks, ground_truth_masks

def showInference(patient, model):

    images_dir = f"./test/dcm/{patient}/0/"       # Dossier contenant les fichiers DICOM image
    seg_path = f"./test/dcm/{patient}/1/1-1.dcm"  # Fichier de segmentation DICOM SEG

    dicom_files = sorted(os.listdir(images_dir))
    image_slices = [pydicom.dcmread(os.path.join(images_dir, f)) for f in dicom_files]

    seg = pydicom.dcmread(seg_path)
    n_frames = int(seg.NumberOfFrames)
    # print(n_frames)


    ref_uids = [ref.ReferencedSOPInstanceUID for ref in seg.ReferencedSeriesSequence[0].ReferencedInstanceSequence]

    # print(seg.SegmentSequence)
    # print(seg.PerFrameFunctionalGroupsSequence)

    segmentsLabels = {
        s.SegmentNumber: s.SegmentLabel
        for s in seg.SegmentSequence
    }

    # for num, label in segmentsLabels.items():
    #     print(f"{num} → {label}")

    # target_segment_number = 2  # on segmente la tumeur (neoplasm)

    target_segment_number = get_target_segment_number(seg)

    masks = []
    uids = []
    res = []

    for i, f in enumerate(seg.PerFrameFunctionalGroupsSequence):
        if int(f.SegmentIdentificationSequence[0].ReferencedSegmentNumber) == target_segment_number:
            masks.append(seg.pixel_array[i])
            uids.append(f.DerivationImageSequence[0].SourceImageSequence[0].ReferencedSOPInstanceUID)

    for dicom_file, img_dcm in zip(sorted(os.listdir(images_dir)), image_slices):
        sop_uid = img_dcm.SOPInstanceUID
        img_path = os.path.join(images_dir, dicom_file)

        image, pred_mask = inference_dicom(img_path, text_prompt, model, is_CT=True, site='lung')
        pred_mask = np.squeeze(pred_mask)
        pv=0

        if sop_uid in uids:
            seg_idx = uids.index(sop_uid)
            seg_mask = masks[seg_idx]
        else:
            seg_mask = np.zeros_like(img_dcm.pixel_array)

        img_array = img_dcm.pixel_array

        target_size = img_array.shape[::-1]  
        pred_mask_resized = cv2.resize(pred_mask, target_size, interpolation=cv2.INTER_NEAREST)

        # dice = diceScore(pred_mask_resized, seg_mask)

        print(sorted(os.listdir(images_dir)).index(dicom_file))

        plt.figure(figsize=(6, 6))
        plt.imshow(img_array, cmap='gray')
        plt.imshow(seg_mask, alpha=0.4, cmap='Blues')
        plt.imshow(pred_mask_resized, alpha=0.4, cmap='Reds')

        plt.axis('off')
        plt.title("SEG (Bleu) vs Prédiction (Rouge)")

        # Ajout de texte sous la figure
        # plt.figtext(0.5, -0.01, f"Dice Score: {dice:.4f} - P-value : {pv:.4f}", wrap=True, ha='center', fontsize=10)

        plt.tight_layout()
        plt.show()

        res.append([img_array, pred_mask_resized])

    return res

def showInferenceNP(patient, model):

    images_dir = f"./test/dcm/{patient}/0/"       # Dossier contenant les fichiers DICOM
    seg_path = f"./test/dcm/{patient}/1/1-1.dcm"  # Fichier de segmentation DICOM SEG

    dicom_files = sorted(os.listdir(images_dir))
    image_slices = [pydicom.dcmread(os.path.join(images_dir, f)) for f in dicom_files]

    seg = pydicom.dcmread(seg_path)
    n_frames = int(seg.NumberOfFrames)


    ref_uids = [ref.ReferencedSOPInstanceUID for ref in seg.ReferencedSeriesSequence[0].ReferencedInstanceSequence]


    segmentsLabels = {
        s.SegmentNumber: s.SegmentLabel
        for s in seg.SegmentSequence
    }

    target_segment_number = get_target_segment_number(seg)

    masks = []
    uids = []
    res = []

    for i, f in enumerate(seg.PerFrameFunctionalGroupsSequence):
        if int(f.SegmentIdentificationSequence[0].ReferencedSegmentNumber) == target_segment_number:
            masks.append(seg.pixel_array[i])
            uids.append(f.DerivationImageSequence[0].SourceImageSequence[0].ReferencedSOPInstanceUID)

    for dicom_file, img_dcm in zip(sorted(os.listdir(images_dir)), image_slices):
        sop_uid = img_dcm.SOPInstanceUID
        img_path = os.path.join(images_dir, dicom_file)

        image, pred_mask = inference_dicom(img_path, text_prompt, model, is_CT=True, site='lung')
        pred_mask = np.squeeze(pred_mask)
        pv=0

        if sop_uid in uids:
            seg_idx = uids.index(sop_uid)
            seg_mask = masks[seg_idx]
        else:
            seg_mask = np.zeros_like(img_dcm.pixel_array)

        img_array = img_dcm.pixel_array

        target_size = img_array.shape[::-1]  
        pred_mask_resized = cv2.resize(pred_mask, target_size, interpolation=cv2.INTER_NEAREST)

        res.append([img_array, pred_mask_resized])

    return res

def showBoundingBox(patient, model, vect):

    x1, x2, y1, y2 = vect

    images_dir = f"./test/dcm/{patient}/0/"
    seg_path = f"./test/dcm/{patient}/1/1-1.dcm"

    dicom_files = sorted(os.listdir(images_dir))
    image_slices = [pydicom.dcmread(os.path.join(images_dir, f)) for f in dicom_files]

    seg = pydicom.dcmread(seg_path)
    n_frames = int(seg.NumberOfFrames)

    ref_uids = [
        ref.ReferencedSOPInstanceUID
        for ref in seg.ReferencedSeriesSequence[0].ReferencedInstanceSequence
    ]

    segmentsLabels = {
        s.SegmentNumber: s.SegmentLabel
        for s in seg.SegmentSequence
    }

    target_segment_number = get_target_segment_number(seg)

    masks = []
    uids = []
    res = []

    for i, f in enumerate(seg.PerFrameFunctionalGroupsSequence):
        if int(f.SegmentIdentificationSequence[0].ReferencedSegmentNumber) == target_segment_number:
            masks.append(seg.pixel_array[i])
            uids.append(f.DerivationImageSequence[0].SourceImageSequence[0].ReferencedSOPInstanceUID)

    for dicom_file, img_dcm in zip(sorted(os.listdir(images_dir)), image_slices):
        sop_uid = img_dcm.SOPInstanceUID
        img_path = os.path.join(images_dir, dicom_file)

        image, pred_mask = inference_dicom(img_path, text_prompt, model, is_CT=True, site='lung')
        pred_mask = np.squeeze(pred_mask)

        if sop_uid in uids:
            seg_idx = uids.index(sop_uid)
            seg_mask = masks[seg_idx]
        else:
            seg_mask = np.zeros_like(img_dcm.pixel_array)

        img_array = img_dcm.pixel_array

        target_size = img_array.shape[::-1]
        pred_mask_resized = cv2.resize(pred_mask, target_size, interpolation=cv2.INTER_NEAREST)

        plt.figure(figsize=(6, 6))
        plt.imshow(img_array, cmap='gray')
        plt.imshow(seg_mask, alpha=0.4, cmap='Blues')
        plt.imshow(pred_mask_resized, alpha=0.4, cmap='Reds')

        # Tracé du rectangle rouge
        plt.plot([x1, x1], [y1, y2], 'r')  # gauche
        plt.plot([x2, x2], [y1, y2], 'r')  # droit
        plt.plot([x1, x2], [y1, y1], 'r')  # haut
        plt.plot([x1, x2], [y2, y2], 'r')  # bas

        plt.axis('off')
        plt.title("SEG (Bleu) vs Prédiction (Rouge) + Bounding Box")

        plt.tight_layout()
        plt.show()

        res.append([img_array, pred_mask_resized])

    return res

def showFilteredInference(patient, model, plot=True):

    def diceScore(mask1, mask2):
        mask1 = (mask1 > 0.5).astype(np.uint8)
        mask2 = (mask2 > 0.5).astype(np.uint8)
        intersection = np.sum(mask1 * mask2)
        total = np.sum(mask1) + np.sum(mask2)
        if total != 0:
            return (2.0 * intersection) / total
        return 0

    def diceFilter(masks):
        n=len(masks)
        if n > 1:
            lst = []
            for i in range(len(masks)):
                moy = 0
                for j in range(len(masks)):
                    if i != j:
                        moy += diceScore(masks[i], masks[j])
                lst.append((i, moy / (len(masks) - 1)))
            lst = sorted(lst, key=lambda x: x[1])
            res = [a for a, _ in lst]
            res = res[n // 2:]
        else:
            res = [i for i in range(n)]
        return res

    images_dir = f"./test/dcm/{patient}/0/"
    seg_path = f"./test/dcm/{patient}/1/1-1.dcm"

    dicom_files = sorted(os.listdir(images_dir))
    image_slices = [pydicom.dcmread(os.path.join(images_dir, f)) for f in dicom_files]
    seg = pydicom.dcmread(seg_path)

    target_segment_number = get_target_segment_number(seg)
    masks = []
    uids = []
    for i, f in enumerate(seg.PerFrameFunctionalGroupsSequence):
        if int(f.SegmentIdentificationSequence[0].ReferencedSegmentNumber) == target_segment_number:
            masks.append(seg.pixel_array[i])
            uids.append(f.DerivationImageSequence[0].SourceImageSequence[0].ReferencedSOPInstanceUID)
    uid_to_seg = {uid: mask for uid, mask in zip(uids, masks)}

    pred_masks = []
    seg_masks = []
    img_arrays = []
    sop_uids = []

    for dicom_file, img_dcm in zip(dicom_files, image_slices):
        sop_uid = img_dcm.SOPInstanceUID
        img_path = os.path.join(images_dir, dicom_file)
        image, pred_mask = inference_dicom(img_path, text_prompt, model, is_CT=True, site='lung')

        pred_mask = np.squeeze(pred_mask)
        img_array = img_dcm.pixel_array
        target_size = img_array.shape[::-1]
        pred_mask_resized = cv2.resize(pred_mask, target_size, interpolation=cv2.INTER_NEAREST)

        if sop_uid in uid_to_seg:
            seg_mask = uid_to_seg[sop_uid]
        else:
            seg_mask = np.zeros_like(img_array)

        pred_masks.append(pred_mask_resized)
        seg_masks.append(seg_mask)
        img_arrays.append(img_array)
        sop_uids.append(sop_uid)

    # Sortie du filtre
    selected_indices = diceFilter(pred_masks)

    # Afficher uniquement les images sélectionnées
    for idx in selected_indices:
        img = img_arrays[idx]
        pred = pred_masks[idx]
        seg = seg_masks[idx]

        if plot :
            plt.figure(figsize=(6, 6))
            plt.imshow(img, cmap='gray')
            plt.imshow(seg, alpha=0.4, cmap='Blues')
            plt.imshow(pred, alpha=0.4, cmap='Reds')
            plt.axis('off')
            plt.title("SEG (Bleu) vs Prédiction (Rouge)")
            plt.tight_layout()
            plt.show()

    return [  # retour des éléments gardés
        (pred_masks[i], seg_masks[i])
        for i in selected_indices
    ]


def maskFilter(patient_masks):
    def diceScore(mask1, mask2):
        mask1 = (mask1 > 0.5).astype(np.uint8)
        mask2 = (mask2 > 0.5).astype(np.uint8)
        intersection = np.sum(mask1 * mask2)
        total = np.sum(mask1) + np.sum(mask2)
        if total != 0:
            return (2.0 * intersection) / total
        return 0

    def diceFilter(masks):
        n = len(masks)
        if n > 1:
            lst = []
            for i in range(len(masks)):
                moy = 0
                for j in range(len(masks)):
                    if i != j:
                        moy += diceScore(masks[i], masks[j])
                lst.append((i, moy / (len(masks) - 1)))
            lst = sorted(lst, key=lambda x: x[1])
            res = [a for a, _ in lst]
            res = res[n // 2:]
        else:
            res = [i for i in range(n)]
        return res

    if isinstance(patient_masks, np.ndarray) and patient_masks.ndim == 3:
        masks_list = [patient_masks[i] for i in range(patient_masks.shape[0])]
        idx = diceFilter(masks_list)
        filtered = [masks_list[i] for i in idx]
        return np.stack(filtered, axis=0) if len(filtered) > 0 else np.empty((0,) + patient_masks.shape[1:], dtype=patient_masks.dtype)
    else:
        idx = diceFilter(patient_masks)
        return [patient_masks[i] for i in idx]
    

def maskFilterGT(patient_masks):
    #ici patient_masks = [preds, gts]
    def diceScore(mask1, mask2):
        mask1 = (mask1 > 0.5).astype(np.uint8)
        mask2 = (mask2 > 0.5).astype(np.uint8)
        intersection = np.sum(mask1 * mask2)
        total = np.sum(mask1) + np.sum(mask2)
        if total != 0:
            return (2.0 * intersection) / total
        return 0

    def diceFilter(masks):
        n = len(masks)
        if n > 1:
            lst = []
            for i in range(len(masks)):
                moy = 0
                for j in range(len(masks)):
                    if i != j:
                        moy += diceScore(masks[i], masks[j])
                lst.append((i, moy / (len(masks) - 1)))
            lst = sorted(lst, key=lambda x: x[1])
            res = [a for a, _ in lst]
            res = res[n // 2:]
        else:
            res = [i for i in range(n)]
        return res

    if isinstance(patient_masks[0], np.ndarray) and patient_masks[0].ndim == 3:
        preds, gts = patient_masks
        if not (isinstance(gts, np.ndarray) and gts.ndim == 3):
            raise TypeError("patient_masks[1] (gts) doit être un np.ndarray 3D.")
        if preds.shape[0] != gts.shape[0]:
            raise ValueError("preds et gts doivent avoir le même nombre de slices (axe 0).")

        preds_list = [preds[i] for i in range(preds.shape[0])]
        idx = diceFilter(preds_list)

        preds_f = [preds[i] for i in idx]
        gts_f   = [gts[i] for i in idx]

        preds_out = np.stack(preds_f, axis=0) if len(preds_f) > 0 else np.empty((0,) + preds.shape[1:], dtype=preds.dtype)
        gts_out   = np.stack(gts_f,   axis=0) if len(gts_f)   > 0 else np.empty((0,) + gts.shape[1:],   dtype=gts.dtype)
        return preds_out, gts_out
    else:
        idx = diceFilter(patient_masks[0])
        pred, gt = [patient_masks[0][i] for i in idx], [patient_masks[1][i] for i in idx]
        return (pred,gt)
    