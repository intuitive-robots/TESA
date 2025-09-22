from itertools import islice
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

from transformers import DetrFeatureExtractor, DetrForObjectDetection

from src.util.device import DEVICE

from collections import defaultdict

from train_sgg import SGGDecoder

def generate_scene_graphs(trainer):
    num_obj_classes = 151
    num_pred_classes = 51
    img_feat_dim = 512
    max_triplets = 100
    
    split = trainer.load_model_name.split('_')
    dataset_name = split[-1]
    model_name = split[-2]
    
    if dataset_name == 'PSG':
        num_obj_classes = 133
        num_pred_classes = 56
    
    if model_name == 'SigLIP':
        img_feat_dim = 768
    elif model_name == 'DINOv2':
        img_feat_dim = 384
    else:
        pass # TODO
    
    sgg_model = SGGDecoder(
        img_feature_dim=img_feat_dim,
        num_object_classes=num_obj_classes,
        num_predicate_classes=num_pred_classes
    ).to(DEVICE)
    
    sgg_model.load_state_dict(torch.load("/home/vquapil/TESA/out_backup/models_vqa_sgg/sgg_model_" + dataset_name + "_" + model_name + ".pth", weights_only=True))
    
    detection_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(DEVICE)
    feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
    
    recall_totals = {20: 0, 50: 0, 100: 0}
    mean_recall_totals = {20: 0, 50: 0, 100: 0}
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(trainer.test_loader):
            num_graphs = len(batch.ptr) - 1
            if glob == "image":
                img_feature_token = batch.y.to(DEVICE).view(num_graphs, img_feat_dim)
            elif glob == "graph":
                img_feature_token = trainer.model(batch.to(DEVICE))
            elif glob == "zero":
                img_feature_token = batch.y.to(DEVICE).view(num_graphs, img_feat_dim)
                #img_feature_token = torch.zeros_like(img_feature_token)
                img_feature_token = torch.rand_like(img_feature_token)

            # Prepare triplets and bounding boxes
            triplets = batch.triplets.to(DEVICE)
            boxes = batch.bbox.to(DEVICE)

            graph_triplets, bbox = [], []
            start_idx = 0
            for n_triplets in batch.num_triplet:
                end_idx = start_idx + n_triplets
                graph_triplets.append(triplets[start_idx:end_idx, :])
                bbox.append(boxes[start_idx:end_idx, :, :])
                start_idx = end_idx

            subj_labels = torch.full((num_graphs, max_triplets), -100, dtype=torch.long, device=DEVICE)
            pred_labels = torch.full_like(subj_labels, -100)
            obj_labels = torch.full_like(subj_labels, -100)

            subj_bboxes = torch.zeros((num_graphs, max_triplets, 4), device=DEVICE)
            obj_bboxes = torch.zeros((num_graphs, max_triplets, 4), device=DEVICE)

            for i, (t, b) in enumerate(zip(graph_triplets, bbox)):
                num = min(t.size(0), max_triplets)
                obj_labels[i, :num] = t[:num, 0]
                pred_labels[i, :num] = t[:num, 1]
                subj_labels[i, :num] = t[:num, 2]

                obj_bboxes[i, :num, :] = b[:num, 0, :]
                subj_bboxes[i, :num, :] = b[:num, 1, :]

            # DETR predictions for object features
            inputs = feature_extractor(images=batch.image, return_tensors="pt").to(DEVICE)
            outputs = detection_model(**inputs)

            pred_boxes = outputs.pred_boxes  # (B, num_queries, 4)
            backbone_features = outputs.last_hidden_state

            # Match GT boxes to predicted boxes
            matched_pred_boxes_obj, indices_obj = match_pred_to_gt_unique(obj_bboxes, pred_boxes, box_iou_cxcywh)
            matched_pred_boxes_subj, indices_subj = match_pred_to_gt_unique(subj_bboxes, pred_boxes, box_iou_cxcywh)

            # Gather object/subject features from matched predicted boxes
            B, Nq, feat_dim = backbone_features.shape
            obj_features = torch.zeros((B, max_triplets, feat_dim), device=DEVICE)
            subj_features = torch.zeros((B, max_triplets, feat_dim), device=DEVICE)

            for b in range(B):
                valid_mask_obj = indices_obj[b] >= 0
                valid_idx_obj = indices_obj[b][valid_mask_obj]
                if valid_idx_obj.numel() > 0:
                    obj_features[b, valid_mask_obj] = backbone_features[b, valid_idx_obj, :]

                valid_mask_subj = indices_subj[b] >= 0
                valid_idx_subj = indices_subj[b][valid_mask_subj]
                if valid_idx_subj.numel() > 0:
                    subj_features[b, valid_mask_subj] = backbone_features[b, valid_idx_subj, :]

            # Forward pass through SGG model
            out = sgg_model(img_feature_token, obj_features, subj_features,
                            matched_pred_boxes_obj, matched_pred_boxes_subj)
            recalls, mean_recalls = evaluate_recall_and_mean_recall(
                out, subj_labels, pred_labels, obj_labels,
                Ks=[20, 50, 100], topk_each=topn
            )

            for K in [20, 50, 100]:
                recall_totals[K] += recalls[K]
                mean_recall_totals[K] += mean_recalls[K]

            num_batches += 1
            
    print(f"Recall@20: {recall_totals[20] / num_batches:.4f}")
    print(f"Mean Recall@20: {mean_recall_totals[20] / num_batches:.4f}")
    print(f"Recall@50: {recall_totals[50] / num_batches:.4f}")
    print(f"Mean Recall@50: {mean_recall_totals[50] / num_batches:.4f}")
    print(f"Recall@100: {recall_totals[100] / num_batches:.4f}")
    print(f"Mean Recall@100: {mean_recall_totals[100] / num_batches:.4f}")