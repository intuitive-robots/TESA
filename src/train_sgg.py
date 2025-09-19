from itertools import islice
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

from transformers import DetrFeatureExtractor, DetrForObjectDetection

from src.util.device import DEVICE

from collections import defaultdict

def test_sgg_model(trainer, glob, topn):
    print("##############################################################")
    print("Global embedding " + glob + " top n " + str(topn))
    print("##############################################################")
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
    
    sgg_model.load_state_dict(torch.load("/home/vquapil/TESA/sgg_model_" + dataset_name + "_" + model_name + ".pth", weights_only=True))
    
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

def evaluate_recall_and_mean_recall(out, subj_labels, pred_labels, obj_labels, Ks=[20, 50, 100], topk_each=5):
    """
    Compute Recall@K and Mean Recall@K for scene graph generation.
    
    out: dict with subj_logits, obj_logits, pred_logits
    subj_labels, pred_labels, obj_labels: (B, max_triplets)
    Ks: list of cutoff values [20, 50, 100]
    topk_each: how many top predictions to consider for each subj/obj/pred (instead of just argmax)
    """
    B, T, _ = out['pred_logits'].shape
    results = {K: {"hits": 0, "total": 0, "per_class_hits": defaultdict(int), "per_class_total": defaultdict(int)} for K in Ks}

    for b in range(B):
        valid_mask = subj_labels[b] != -100
        n_gt = valid_mask.sum().item()
        if n_gt == 0:
            continue

        # Ground truth triplets
        gt_subj = subj_labels[b][valid_mask]
        gt_obj  = obj_labels[b][valid_mask]
        gt_pred = pred_labels[b][valid_mask]
        gt_triplets = torch.stack([gt_subj, gt_pred, gt_obj], dim=1)  # (n_gt, 3)

        # Predicted scores
        subj_scores = out['subj_logits'][b].softmax(-1)
        obj_scores  = out['obj_logits'][b].softmax(-1)
        pred_scores = out['pred_logits'][b].softmax(-1)

        all_triplets, all_scores = [], []

        for t in range(T):
            top_subj_scores, top_subj_idx = subj_scores[t].topk(min(topk_each, subj_scores.shape[1]))
            top_obj_scores, top_obj_idx   = obj_scores[t].topk(min(topk_each, obj_scores.shape[1]))
            top_pred_scores, top_pred_idx = pred_scores[t].topk(min(topk_each, pred_scores.shape[1]))

            # Cartesian product of top-k subj/obj/pred
            for si, sv in zip(top_subj_idx, top_subj_scores):
                for oi, ov in zip(top_obj_idx, top_obj_scores):
                    for pi, pv in zip(top_pred_idx, top_pred_scores):
                        score = sv * ov * pv
                        all_scores.append(score.item())
                        all_triplets.append((si.item(), pi.item(), oi.item()))

        # Rank all predicted triplets
        all_scores = torch.tensor(all_scores)
        sorted_idx = torch.argsort(all_scores, descending=True)
        ranked_triplets = [all_triplets[i] for i in sorted_idx]

        # Evaluate for each K
        for K in Ks:
            topk_triplets = set(ranked_triplets[:K])
            for gt in gt_triplets.cpu().numpy():
                gt_tuple = tuple(gt.tolist())
                results[K]["total"] += 1
                results[K]["per_class_total"][gt[1].item()] += 1
                if gt_tuple in topk_triplets:
                    results[K]["hits"] += 1
                    results[K]["per_class_hits"][gt[1].item()] += 1

    # Compute Recall@K and Mean Recall@K
    recalls, mean_recalls = {}, {}
    for K in Ks:
        R = results[K]["hits"] / max(1, results[K]["total"])
        recalls[K] = R
        per_class_recalls = []
        for c, total in results[K]["per_class_total"].items():
            if total > 0:
                per_class_recalls.append(results[K]["per_class_hits"][c] / total)
        mean_recalls[K] = sum(per_class_recalls) / len(per_class_recalls) if per_class_recalls else 0.0

    return recalls, mean_recalls

def train_sgg_model(trainer):
    num_obj_classes = 151
    num_pred_classes = 51
    img_feat_dim = 512
    max_triplets = 100
    max_epochs = 10
    
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

    detection_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(DEVICE)
    feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")

    # Optimizer + scheduler
    optimizer = AdamW(sgg_model.parameters(), lr=1e-2, weight_decay=0.05)
    total_steps = max_epochs * len(trainer.train_loader)
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    for i in tqdm(range(max_epochs)):
        sgg_model.train()
        train_loss = 0.0
        for batch in (pbar := tqdm(trainer.train_loader)):
            optimizer.zero_grad()
            
            triplets = batch.triplets.to(DEVICE)
            boxes = batch.bbox.to(DEVICE)
            num_graphs = len(batch.ptr) - 1
            
            graph_triplets = []
            bbox = []
            start_idx = 0
            for n_triplets in batch.num_triplet:  # num_triplets per graph
                end_idx = start_idx + n_triplets
                graph_triplets.append(triplets[start_idx:end_idx, :])    # shape (num_triplets_i, 3)
                bbox.append(boxes[start_idx:end_idx, :, :])      # shape (num_triplets_i, 2, 4)
                start_idx = end_idx
                
            subj_labels = torch.full((num_graphs, max_triplets), fill_value=-100, dtype=torch.long, device=DEVICE)
            pred_labels = torch.full_like(subj_labels, -100)
            obj_labels = torch.full_like(subj_labels, -100)

            subj_bboxes = torch.zeros((num_graphs, max_triplets, 4), dtype=torch.float32, device=DEVICE)
            obj_bboxes = torch.zeros((num_graphs, max_triplets, 4), dtype=torch.float32, device=DEVICE)

            for j, (t, b) in enumerate(zip(graph_triplets, bbox)):
                num = min(t.size(0), max_triplets)
                obj_labels[j, :num] = t[:num, 0]
                pred_labels[j, :num] = t[:num, 1]
                subj_labels[j, :num] = t[:num, 2]
                
                # b has shape (num_triplets_i, 2, 4) -> [:,0,:] subj, [:,1,:] obj
                obj_bboxes[j, :num, :] = b[:num, 0, :]
                subj_bboxes[j, :num, :] = b[:num, 1, :]
            
            img_feature_token = batch.y.to(DEVICE).view(len(batch.ptr) - 1, img_feat_dim)
            
            with torch.no_grad():
                inputs = feature_extractor(images=batch.image, return_tensors="pt").to(DEVICE)
                outputs = detection_model(**inputs, output_hidden_states=True)
            
                pred_boxes = outputs.pred_boxes  # (B, num_queries, 4) normalized cxcywh
                
                backbone_features = outputs.last_hidden_state
            
            matched_pred_boxes_obj, indices_obj = match_pred_to_gt_unique(obj_bboxes, pred_boxes, box_iou_cxcywh)
            matched_pred_boxes_subj, indices_subj = match_pred_to_gt_unique(subj_bboxes, pred_boxes, box_iou_cxcywh)
            
            B, Nq, _ = outputs.logits.shape
            
            obj_features = torch.zeros((B, obj_bboxes.size(1), backbone_features.size(-1)), device=DEVICE)
            subj_features = torch.zeros((B, subj_bboxes.size(1), backbone_features.size(-1)), device=DEVICE)
            
            for b in range(B):
                # Object features
                valid_mask_obj = indices_obj[b] >= 0
                valid_indices_obj = indices_obj[b][valid_mask_obj]  # indices of matched predicted boxes
                if valid_indices_obj.numel() > 0:
                    obj_features[b, valid_mask_obj] = backbone_features[b, valid_indices_obj, :]

                # Subject features
                valid_mask_subj = indices_subj[b] >= 0
                valid_indices_subj = indices_subj[b][valid_mask_subj]
                if valid_indices_subj.numel() > 0:
                    subj_features[b, valid_mask_subj] = backbone_features[b, valid_indices_subj, :]

            out = sgg_model(img_feature_token, obj_features, subj_features, matched_pred_boxes_obj, matched_pred_boxes_subj)

            loss_subj = criterion(out['subj_logits'].view(-1, num_obj_classes), subj_labels.view(-1))
            loss_obj = criterion(out['obj_logits'].view(-1, num_obj_classes), obj_labels.view(-1))
            loss_pred = criterion(out['pred_logits'].view(-1, num_pred_classes), pred_labels.view(-1))

            # Bounding box losses
            loss_subj_bbox = bbox_loss(out['subj_bboxes'], subj_bboxes)
            loss_obj_bbox  = bbox_loss(out['obj_bboxes'], obj_bboxes)

            loss = loss_subj + loss_obj + loss_pred + 5 * loss_subj_bbox + 5 * loss_obj_bbox
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            pbar.set_description(f"Loss: {loss.item():.4f}")
            
            train_loss += loss.item()
        
        print(f"Epoch {i+1} Train Loss: {train_loss / len(trainer.train_loader):.4f}")
        
        torch.save(sgg_model.state_dict(), "/home/vquapil/TESA/sgg_model_" + dataset_name + "_" + model_name + ".pth")
        
        sgg_model.eval()
        val_loss = 0.0
        correct_subj, correct_obj, correct_pred = 0, 0, 0
        total_subj, total_obj, total_pred = 0, 0, 0
        triplet_correct, triplet_total = 0, 0

        total_iou_obj, total_iou_subj = 0.0, 0.0
        num_iou_obj, num_iou_subj = 0, 0

        with torch.no_grad():
            for batch in tqdm(trainer.val_loader):
                num_graphs = len(batch.ptr) - 1
                img_feature_token = batch.y.to(DEVICE).view(num_graphs, img_feat_dim)

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

                for k, (t, b) in enumerate(zip(graph_triplets, bbox)):
                    num = min(t.size(0), max_triplets)
                    obj_labels[k, :num] = t[:num, 0]
                    pred_labels[k, :num] = t[:num, 1]
                    subj_labels[k, :num] = t[:num, 2]

                    obj_bboxes[k, :num, :] = b[:num, 0, :]
                    subj_bboxes[k, :num, :] = b[:num, 1, :]

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

                # Classification losses
                loss_subj = criterion(out['subj_logits'].view(-1, num_obj_classes), subj_labels.view(-1))
                loss_obj = criterion(out['obj_logits'].view(-1, num_obj_classes), obj_labels.view(-1))
                loss_pred = criterion(out['pred_logits'].view(-1, num_pred_classes), pred_labels.view(-1))

                # Bounding box losses
                loss_subj_bbox = bbox_loss(out['subj_bboxes'], subj_bboxes)
                loss_obj_bbox = bbox_loss(out['obj_bboxes'], obj_bboxes)

                val_loss += (loss_subj + loss_obj + loss_pred + loss_subj_bbox + loss_obj_bbox).item()

                # Predicted labels
                pred_subj = out['subj_logits'].argmax(-1)
                pred_obj = out['obj_logits'].argmax(-1)
                pred_pred = out['pred_logits'].argmax(-1)

                mask = subj_labels != -100

                correct_subj += ((pred_subj == subj_labels) & mask).sum().item()
                correct_obj += ((pred_obj == obj_labels) & mask).sum().item()
                correct_pred += ((pred_pred == pred_labels) & mask).sum().item()
                total_subj += mask.sum().item()
                total_obj += mask.sum().item()
                total_pred += mask.sum().item()

                triplet_mask = mask & (pred_subj == subj_labels) & (pred_pred == pred_labels) & (pred_obj == obj_labels)
                triplet_correct += triplet_mask.sum().item()
                triplet_total += mask.sum().item()

                # IoU metrics
                valid_obj_mask = (obj_bboxes.sum(-1) > 0)  # mask padded GT boxes
                valid_subj_mask = (subj_bboxes.sum(-1) > 0)

                iou_obj = box_iou_cxcywh(out['obj_bboxes'], obj_bboxes)  # shape: (B, max_triplets, max_triplets)
                iou_subj = box_iou_cxcywh(out['subj_bboxes'], subj_bboxes)

                # For each valid GT, take the diagonal (pred vs GT) for matched pairs
                total_iou_obj = 0.0
                for b, iou_mat in enumerate(iou_obj):  # iou_obj is a list
                    diag = torch.diag(iou_mat)  # (max_triplets,)
                    total_iou_obj += diag[valid_obj_mask[b]].sum().item()
                    num_iou_obj += valid_obj_mask[b].sum().item()

                total_iou_subj = 0.0
                for b, iou_mat in enumerate(iou_subj):  # iou_obj is a list
                    diag = torch.diag(iou_mat)  # (max_triplets,)
                    total_iou_subj += diag[valid_subj_mask[b]].sum().item()
                    num_iou_subj += valid_subj_mask[b].sum().item()

        # Final metrics
        mean_iou_obj = total_iou_obj / max(1, num_iou_obj)
        mean_iou_subj = total_iou_subj / max(1, num_iou_subj)

        print(f"Epoch {i+1} Validation Loss: {val_loss / len(trainer.val_loader):.4f}")
        print(f"Validation Accuracy - Subj: {correct_subj/total_subj:.4f}, Obj: {correct_obj/total_obj:.4f}, Pred: {correct_pred/total_pred:.4f}")
        print(f"Validation Triplet Accuracy: {triplet_correct/triplet_total:.4f}")
        print(f"Mean IoU - Obj: {mean_iou_obj:.4f}, Subj: {mean_iou_subj:.4f}")



def bbox_loss(pred_bboxes, gt_bboxes):
    """
    pred_bboxes: (B, num_triplets, 4) - predicted normalized [cx,cy,w,h]
    gt_bboxes:   (B, num_triplets, 4) - target normalized [cx,cy,w,h] with zeros for padding
    """
    # Mask: 1 if the GT box is non-zero, 0 if padding
    valid_mask = ~(gt_bboxes == 0).all(dim=-1)  # (B, num_triplets)

    if valid_mask.sum() == 0:
        return torch.tensor(0., device=pred_bboxes.device)

    # Select only valid boxes
    pred_valid = pred_bboxes[valid_mask]  # (num_valid, 4)
    gt_valid = gt_bboxes[valid_mask]      # (num_valid, 4)

    # Smooth L1 loss
    loss = torch.nn.functional.smooth_l1_loss(pred_valid, gt_valid, reduction='mean')
    return loss


def box_iou_cxcywh(gt_boxes_list, pred_boxes):
    """
    Compute IoU between variable-length GT boxes (list of tensors) and fixed-size predicted boxes.
    
    Args:
        gt_boxes_list: list of length B, each element (num_gt_i, 4) in [cx, cy, w, h].
        pred_boxes: (B, P, 4) predicted boxes in [cx, cy, w, h].
    
    Returns:
        iou_per_batch: list of length B, each element (num_gt_i, P) IoU matrix.
    """
    B, P, _ = pred_boxes.shape
    iou_per_batch = []

    def cxcywh_to_xyxy(boxes):
        x_c, y_c, w, h = boxes.unbind(-1)
        x_min = x_c - 0.5 * w
        y_min = y_c - 0.5 * h
        x_max = x_c + 0.5 * w
        y_max = y_c + 0.5 * h
        return torch.stack((x_min, y_min, x_max, y_max), dim=-1)

    for b in range(B):
        gt_b = gt_boxes_list[b]
        pred_b = pred_boxes[b]  # (P,4)

        if gt_b.numel() == 0:
            iou_per_batch.append(torch.empty((0, P), device=pred_boxes.device))
            continue

        gt_xyxy = cxcywh_to_xyxy(gt_b)       # (num_gt_i, 4)
        pred_xyxy = cxcywh_to_xyxy(pred_b)   # (P, 4)

        area_gt = (gt_xyxy[:, 2] - gt_xyxy[:, 0]) * (gt_xyxy[:, 3] - gt_xyxy[:, 1])
        area_pred = (pred_xyxy[:, 2] - pred_xyxy[:, 0]) * (pred_xyxy[:, 3] - pred_xyxy[:, 1])

        lt = torch.max(gt_xyxy[:, None, :2], pred_xyxy[None, :, :2])  # (num_gt_i, P, 2)
        rb = torch.min(gt_xyxy[:, None, 2:], pred_xyxy[None, :, 2:])  # (num_gt_i, P, 2)
        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]

        union = area_gt[:, None] + area_pred[None, :] - inter
        iou = inter / (union + 1e-6)

        iou_per_batch.append(iou)

    return iou_per_batch

def match_pred_to_gt_unique(bboxes, pred_boxes, iou_fn):
    """
    obj_bboxes: (num_gt, 4) GT boxes [cx,cy,w,h] (with possible duplicates)
    pred_boxes: (num_pred, 4) Predicted boxes [cx,cy,w,h]
    iou_fn: function(boxes1, boxes2) -> IoU matrix (N,M)
    threshold: IoU threshold for matching

    Returns:
        matches_original: (num_gt,) index of best pred for each GT (-1 if none)
        unique_gt_boxes: unique GT boxes
        unique_to_original: mapping from unique box idx -> original GT indices
    """
    # 1. Get unique boxes + mapping back to original
    unique_boxes, inverse_indices = unique_boxes_batched(bboxes)

    # 2. Compute IoU between unique GT boxes and predictions
    iou_matrix = iou_fn(unique_boxes, pred_boxes)  # (num_unique_gt, num_pred)

    B = bboxes.size(0)
    matched_pred_boxes = bboxes.clone()
    matched_pred_indices = torch.full((B, bboxes.size(1)), -1, device=bboxes.device, dtype=torch.long)

    for b in range(B):
        iou_mat = iou_matrix[b]          # (num_unique_gt_b, num_pred_b)
        inv_idx = inverse_indices[b]     # maps valid (non-zero) GT → unique index
        unique_gt = unique_boxes[b]      # (num_unique_gt_b, 4)
        pred = pred_boxes[b]
        
        if iou_mat.numel() == 0:
            print("WARNING NO MATCHES FOR PREDICTED BBOXES AND GT BOXES")
            continue

        # 3. Best prediction for each unique GT box
        best_pred_per_unique = iou_mat.argmax(dim=1)      # (num_unique_gt_b,)
        best_pred = pred[best_pred_per_unique]
        
        valid_mask = ~(bboxes[b] == 0).all(dim=1)
        valid_indices = torch.where(valid_mask)[0]

        # Assign predicted boxes for valid GTs
        matched_pred_boxes[b, valid_indices] = best_pred[inv_idx[valid_indices]]
        matched_pred_indices[b, valid_indices] = best_pred_per_unique[inv_idx[valid_indices]]
    
    return matched_pred_boxes, matched_pred_indices


def unique_boxes_batched(boxes: torch.Tensor, tol: float = 1e-4, pad_val: float = 0.0):
    """
    Deduplicate boxes per graph in a batch.
    
    boxes: (B, T, 4) where B=batch_size, T=num_triplets, coords in cxcywh.
    pad_val: padded boxes (all coords==pad_val) are ignored.
    
    Returns:
        unique_boxes_per_batch: list of (K_i,4) unique boxes for each batch item.
        inverse_per_batch: list of (T,) mapping each triplet to its unique box id (-1 for pad).
    """
    B, T, _ = boxes.shape
    unique_boxes_per_batch = []
    inverse_per_batch = []

    for b in range(B):
        boxes_b = boxes[b]  # (T,4)
        valid_mask = ~(boxes_b == pad_val).all(dim=1)
        boxes_valid = boxes_b[valid_mask]

        if boxes_valid.numel() == 0:
            unique_boxes_per_batch.append(torch.empty((0,4), device=boxes.device))
            inverse_per_batch.append(torch.full((T,), -1, device=boxes.device, dtype=torch.long))
            continue

        scale = 1.0 / tol
        q = torch.round(boxes_valid * scale).to(torch.long)

        uniq_q, inv = torch.unique(q, dim=0, return_inverse=True)

        unique_boxes_per_batch.append(uniq_q.to(torch.float32) / scale)

        inverse_full = torch.full((T,), -1, device=boxes.device, dtype=torch.long)
        inverse_full[valid_mask] = inv
        inverse_per_batch.append(inverse_full)

    return unique_boxes_per_batch, inverse_per_batch


class SGGDecoder(nn.Module):
    def __init__(self,
                 img_feature_dim=512,       # CLIP global embedding dim
                 box_feature_dim=256,       # BB features (from ROIAlign or object detector)
                 hidden_dim=512,
                 num_layers=6,
                 num_heads=8,
                 num_object_classes=151,
                 num_predicate_classes=51,
                 max_triplets=100):
        super().__init__()

        # GPT2-style transformer decoder
        config = GPT2Config(
            vocab_size=1,  # no text vocab
            n_embd=hidden_dim,
            n_layer=num_layers,
            n_head=num_heads,
            use_cache=False
        )
        self.decoder = GPT2Model(config)

        self.max_triplets = max_triplets

        # Projections
        self.img_proj = nn.Linear(img_feature_dim, hidden_dim)    # global img/graph feature
        self.box_proj = nn.Linear(box_feature_dim, hidden_dim)    # object/subject feature
        self.coord_proj = nn.Linear(4, hidden_dim)                # bounding box coordinates (cx, cy, w, h)

        # Base query embeddings
        self.query_embeds = nn.Parameter(torch.randn(1, max_triplets * 4, hidden_dim))

        # Positional & group embeddings
        self.pos_emb = nn.Embedding(max_triplets * 5, hidden_dim)
        self.group_emb = nn.Embedding(max_triplets, hidden_dim)

        # Output heads
        self.obj_classifier = nn.Linear(hidden_dim * 3, num_object_classes)
        self.subj_classifier = nn.Linear(hidden_dim * 3, num_object_classes)
        self.pred_classifier = nn.Linear(hidden_dim * 5, num_predicate_classes)
        self.obj_bbox_head = nn.Linear(hidden_dim, 4)   # normalized [cx, cy, w, h]
        self.subj_bbox_head = nn.Linear(hidden_dim, 4)

    def forward(self, img_feat, obj_feats, subj_feats, obj_boxes, subj_boxes, num_triplets=None):
        """
        img_feat: (B, img_feature_dim)          - global embedding
        obj_feats: (B, num_triplets, box_feature_dim) - object BB features
        subj_feats: (B, num_triplets, box_feature_dim) - subject BB features
        obj_boxes: (B, num_triplets, 4)         - normalized [cx, cy, w, h]
        subj_boxes: (B, num_triplets, 4)        - normalized [cx, cy, w, h]
        """
        B = img_feat.size(0)
        num_triplets = num_triplets or self.max_triplets
        num_local_tokens = 4 * num_triplets  # obj_feat, obj_box, subj_feat, subj_box per triplet
        total_tokens = 1 + num_local_tokens  # +1 for global image token

        # Base local queries
        queries = self.query_embeds.expand(B, num_local_tokens, -1)  # (B, num_local_tokens, hidden_dim)

        # Positional embeddings
        pos_ids = torch.arange(total_tokens, device=img_feat.device).unsqueeze(0).repeat(B, 1)
        global_pos = self.pos_emb(pos_ids[:, 0:1])          # global token
        local_pos = self.pos_emb(pos_ids[:, 1:])           # local tokens

        # Group embeddings
        # global token gets no group embedding
        local_group_ids = torch.arange(num_triplets, device=img_feat.device).unsqueeze(0).repeat(B, 1)
        local_group_ids = local_group_ids.repeat_interleave(4, dim=1)  # each triplet's 4 tokens share same id
        local_group = self.group_emb(local_group_ids)

        # Global token embedding
        img_token = self.img_proj(img_feat).unsqueeze(1) + global_pos  # no group embedding

        # Local token embeddings
        queries = queries + local_pos + local_group

        # Split and project object/subject tokens
        obj_feat_tokens = self.box_proj(obj_feats)
        subj_feat_tokens = self.box_proj(subj_feats)
        obj_box_tokens = self.coord_proj(obj_boxes)
        subj_box_tokens = self.coord_proj(subj_boxes)

        # Interleave tokens: [obj_feat, obj_box, subj_feat, subj_box] per triplet
        interleaved_tokens = torch.stack([obj_feat_tokens, obj_box_tokens, subj_feat_tokens, subj_box_tokens], dim=2)
        interleaved_tokens = interleaved_tokens.view(B, -1, obj_feat_tokens.size(-1))  # (B, num_local_tokens, hidden_dim)
        queries = queries + interleaved_tokens

        # Concatenate global token
        queries = torch.cat([img_token, queries], dim=1)  # (B, total_tokens, hidden_dim)

        # Transformer decoding
        out = self.decoder(inputs_embeds=queries).last_hidden_state  # (B, total_tokens, hidden_dim)

        # Split outputs
        local_out = out[:, 1:, :]
        obj_tokens = local_out[:, 0::4, :]
        obj_bbox_tokens = local_out[:, 1::4, :]
        subj_tokens = local_out[:, 2::4, :]
        subj_bbox_tokens = local_out[:, 3::4, :]

        # Predict corrected BB coordinates
        obj_bboxes = torch.sigmoid(self.obj_bbox_head(obj_bbox_tokens))
        subj_bboxes = torch.sigmoid(self.subj_bbox_head(subj_bbox_tokens))

        B, num_triplets, H = obj_tokens.shape

        # Broadcast global token to all triplets
        global_token = out[:, 0:1, :]
        global_broadcast = global_token.expand(B, num_triplets, H) 
        
        # Predict classes
        combined_tokens_obj = torch.cat([global_broadcast, obj_tokens, obj_bbox_tokens], dim=-1)
        combined_tokens_subj = torch.cat([global_broadcast, subj_tokens, subj_bbox_tokens], dim=-1)
        obj_logits = self.obj_classifier(combined_tokens_obj)
        subj_logits = self.subj_classifier(combined_tokens_subj)

        # Predict predicates using all tokens including global
        combined_tokens = torch.cat([global_broadcast, obj_tokens, obj_bbox_tokens, subj_tokens, subj_bbox_tokens], dim=-1)
        pred_logits = self.pred_classifier(combined_tokens)

        return {
            "obj_logits": obj_logits,
            "subj_logits": subj_logits,
            "pred_logits": pred_logits,
            "obj_bboxes": obj_bboxes,
            "subj_bboxes": subj_bboxes,
            "global_token": out[:, 0, :]
        }

