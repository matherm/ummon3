import numpy as np
import torch
from sklearn.metrics import jaccard_score, precision_recall_curve

from ummon.metrics.accuracy import classify
from .base import *

__all__ = ['IoU', 'JaccardScore', 'InstanceAveragePrecision', 'InstanceIoU', 'InstanceSegmentationMetrics']


class SegmentationMetrics(OfflineMetric):
    def __call__(self, output, target):
        #results = [self.func(c1, c2) for c1, c2 in zip(output, target)]
        result = self.func(output,target)
        return result

    @classmethod
    def __repr__(cls):
        return cls.__name__


class InstanceSegmentationMetrics(OfflineMetric):
    def __call__(self, output, target):
        assert all([o.dim() == 2 and o.shape[1] == 2 for o in output]), 'output vector must have shape [Nx2]'
        assert all([t.dim() == 2 and t.shape[1] == 2 for t in target]), 'target vector must have shape [Nx2]'
        result = self.func(output, target)
        return result

    @classmethod
    def __repr__(cls):
        return cls.__name__


class IoU(SegmentationMetrics):
    """Compute the Intersection over Union (IoU) of two arbitrary nd segmentations
        Usage:  iou = IoU()
                # list of np.arrays of output and target labels.
                iou(output_labels: list, target_labels: list, num_classes : int)
        Attributes:
            func (TYPE): function used in parent class
        """

    def __init__(self):
        self.func = self.calc_iou_def
        self.meanIoUs = {}



    def calc_iou_def(self, output, target):
        categories = {torch.unique(x).item() for x in {b.category for b in target}} # TODo: list??

        ious = {c: [] for c in categories}
        for o, t in zip(output,target):
            for cat_index in range(0, len(t.category)):
                # gathering all labels of a category first and batch calculating the iou might be more efficient
                batch_indices = torch.where(t.batch == cat_index)
                mean_iou, class_iou = self._calc_iou_single(o[batch_indices], t.y[batch_indices])
                #ious[t.category[cat_index]].append([12])
                ious[t.category[cat_index].item()].append(mean_iou)
                u = 2

        for cat in categories:
            ious[cat] = torch.tensor(ious[cat]).mean().item()

        return torch.tensor(list(ious.values())).mean().item()


        #TODO:
        # loop through batch objects
        # Calc per class iou per sample of batch
        # Add per class iou in container [0,...,len[categories]]
        # Take mean of per class per sample
        # Take mean of per sample
        a = 2

    def _calc_iou_single(self, output, target):
        # target = [np.array([1, 3, 2, 1, 0, 0, 1, 3])]
        # output = [np.array([0, 3, 2, 2, 1, 0, 1, 2])]

        # class:    0       1       2       3
        # -----------------------------------------------------------------------
        # TP        1       1       1       1
        # TN        5       4       5       6
        # FP        1       1       2       0
        # FN        1       2       0       1

        # pred = output
        # num_classes = 4
        # conf_matrix = np.zeros((num_classes, num_classes))
        # for o, t in zip(pred, target):
        #     conf_matrix[o, t] += 1
        # comment out the next  lines until print(conf_matrix) to get it running with the local main example

        pred = output.max(dim=1)[1]
        num_classes_target = torch.max(target) + 1
        num_classes_output = torch.max(pred) + 1
        #if num_classes_output.item() != num_classes_target.item():
            #print("Number of classes in target and output does not match!\n")

        num_classes = num_classes_target if num_classes_target > num_classes_output else num_classes_output

        conf_matrix = np.zeros((num_classes, num_classes))
        for o, t in zip(pred, target):
            conf_matrix[o, t] += 1

        #print(conf_matrix)

        # debug_conf_matrix_entries= []
        per_class_iou = np.array([])
        TP = TN = FP = FN = 0
        for i in range(0, num_classes):
            TP = conf_matrix[i, i]  # True Positives. Predicted correct target
            FP = np.sum(conf_matrix[i, :]) - TP  # False Positives. Predicted true class is false target
            FN = np.sum(conf_matrix[:, i]) - TP  # False Negatives. Predicted false class is true target
            TN = np.sum(conf_matrix) - (TP + FP + FN)  # True Negatives. Predicted false class is false target

            # debug_conf_matrix_entries.append((TP, TN, FP,FN))

            intersection = TP
            union = TP + FP + FN
            per_class_iou = np.append(per_class_iou, intersection / union)

        # TODO: check for NaNs and set them to 1!
        per_class_iou[np.isnan(per_class_iou)] = 1
        # print(debug_conf_matrix_entries)
        mean_iou = np.mean(per_class_iou)

        return mean_iou, per_class_iou


class JaccardScore(SegmentationMetrics):
    """
    Computes the similarity between two samples sets (tensors), whereas geometric_metrics#IoU is defined for bounding
    box similarities.

    The predicted output can be either

    - list of 1-d tensors, where each entry contains probability to be a `hit` (detection), or

    - list of 2-d tensors with 2nd dimension of size 2 (N, 2), where output[n,0] < output[n,1] means hit and vice versa (binary class segmentation), or

    - list of 2-d tensors with 2nd dimension of size greater than 2 (N, X: X > 2), where the index of maximum entry defines the class (multiclass segmentation)

    Returns:

    - case of detection and binary class segmentation: the Jaccard cooefficient (Intersection over Union) for a hit

    - case of multiclass segmentation: weighted mean coefficient over all classes and tuple of per class coefficient
    """

    def __init__(self):
        self.func = self.calc_score

    @staticmethod
    def calc_score(output_list, targets_list):
        preds_list = []
        if len(targets_list) == 0:
            return 0.0

        if not isinstance(targets_list[0], torch.Tensor):
            targets_list = [t.y for t in targets_list]

        multiclass = False
        for output in output_list:
            if output.dim() == 2 and output.size()[1] > 1:
                detections = classify(output)
                if output.size()[1] > 2:
                    multiclass = True
            else:
                detections = torch.zeros_like(output)
                detections[output >= 0.0] = 1
            preds_list.append(detections)

        targets = torch.cat(targets_list).cpu()
        preds = torch.cat(preds_list).cpu()

        if multiclass:
            return (jaccard_score(targets, preds, average='weighted') * 100,
                    (jaccard_score(targets, preds, average=None) * 100).tolist())
        else:
            return jaccard_score(targets, preds, average='binary') * 100


def find_correspondences(o, t, iou_threshold):
    target_classes = t[:, 0]
    target_obj_ids = t[:, 1]
    pred_classes = o[:, 0]
    pred_obj_ids = o[:, 1]

    unique_obj_ids = torch.unique(target_obj_ids[target_classes == 1])
    unique_pred_ids = torch.unique(pred_obj_ids[pred_classes >= 0])
    unique_pred_ids = unique_pred_ids[unique_pred_ids != -1]

    ious = torch.full((len(unique_obj_ids), len(unique_pred_ids)), -1)
    n_targets = len(unique_obj_ids)
    n_pred = len(unique_pred_ids)
    target_to_output = torch.full((n_targets,), -1)
    output_to_target = torch.full((n_pred,), -1)
    for i, obj_id in enumerate(unique_obj_ids):
        target_mask = target_obj_ids == obj_id
        uni, count = torch.unique(pred_obj_ids[target_mask], return_counts=True)
        maybe_hits_indices = np.argwhere(count >= iou_threshold * count.sum())

        for maybe_idx in maybe_hits_indices.flatten():
            candidate_id = uni[maybe_idx]

            # Overlap must be at least threshold
            if candidate_id not in unique_pred_ids:
                continue
            pred_mask = pred_obj_ids == candidate_id
            single_target = np.zeros_like(target_obj_ids)
            single_pred = np.zeros_like(target_obj_ids)
            target_or_pred_mask = target_mask | pred_mask
            single_target[target_mask] = 1
            single_pred[pred_mask] = 1
            single_target = single_target[target_or_pred_mask]
            single_pred = single_pred[target_or_pred_mask]
            iou = jaccard_score(single_target, single_pred)
            o_i = np.argwhere(unique_pred_ids == candidate_id)[0]
            ious[i, o_i] = iou

    for i, pred_id in enumerate(unique_pred_ids):
        pred_mask = pred_obj_ids == pred_id
        uni, count = torch.unique(pred_obj_ids[pred_mask], return_counts=True)
        maybe_hits_indices = np.argwhere(count >= iou_threshold * count.sum())

        for maybe_idx in maybe_hits_indices.flatten():
            candidate_id = uni[maybe_idx]
            o_i = np.argwhere(unique_pred_ids == candidate_id)[0]

            # Overlap must be at least threshold
            if candidate_id not in unique_obj_ids or ious[i, o_i] != -1:
                continue
            target_mask = target_obj_ids == candidate_id
            single_target = np.zeros_like(pred_obj_ids)
            single_pred = np.zeros_like(pred_obj_ids)
            target_or_pred_mask = target_mask | pred_mask
            single_target[target_mask] = 1
            single_pred[pred_mask] = 1
            single_target = single_target[target_or_pred_mask]
            single_pred = single_pred[target_or_pred_mask]
            iou = jaccard_score(single_target, single_pred)
            ious[i, o_i] = max(iou, ious[i, o_i])
    iou_argsort = np.argsort(-ious.reshape(-1))

    for i in range(n_pred * n_targets):
        t = int(iou_argsort[i] / n_pred)
        o = iou_argsort[i] % n_pred
        # to count as correct detection area the iou must exceed the threshold
        if ious[t, o] > iou_threshold and output_to_target[o] == -1 and target_to_output[t] == -1:
            output_to_target[o] = t
            target_to_output[t] = o

    return output_to_target, target_to_output


class InstanceIoU(InstanceSegmentationMetrics):
    def __init__(self, ):
        self.func = self.calc_score

    @staticmethod
    def calc_score(output_list, targets_list):
        TP = FP = TN = 0
        for o, t in zip(output_list, targets_list):  # iter over scenes
            output_to_target, target_to_output = find_correspondences(o, t, 0.5)

            assert (output_to_target != -1).sum() == (target_to_output != -1).sum()
            TP += (output_to_target != -1).sum().item()
            FP += (output_to_target == -1).sum().item()
            TN += (target_to_output == -1).sum().item()

        return TP / (TP + FP + TN)


class InstanceAveragePrecision(InstanceSegmentationMetrics):
    """ Compute the Average Precision of target and prediction vector.

    output_list and targets_list are lists containing output and target vectors of each scene

    Vectors are required to be in shape [Nx2] where N is number of points.

    First column in target: classes (0 or 1)

    First column in output: prediction score (pred >= 0 is a hit)

    Second column (both): object id
    """

    def __init__(self, return_prec_rec_curve=False):
        self.func = self.calc_score
        self.return_prec_rec_curve = return_prec_rec_curve

    def calc_score(self, output_list, targets_list):
        if len(targets_list) == 0:
            return 0.0

        bbox_labels_list = []
        scores_list = []

        for o, t in zip(output_list, targets_list):
            output_to_target, target_to_output = find_correspondences(o, t, 0.5)
            n_pred = len(output_to_target)
            n_fn = (target_to_output == -1).sum()
            bbox_labels = np.ones(n_pred + n_fn)
            bbox_labels[np.where(output_to_target == -1)] = 0
            scores = np.zeros_like(bbox_labels)
            pred_obj_ids = np.unique(o[:, 1][o[:, 0] >= 0.0])
            pred_obj_ids = pred_obj_ids[pred_obj_ids != -1]
            scores[:n_pred] = [torch.mean(o[:, 0][o[:, 1] == pred_id]) for pred_id in pred_obj_ids]

            bbox_labels_list.append(bbox_labels)
            scores_list.append(scores)

        bbox_labels_list = np.concatenate(bbox_labels_list)
        scores_list = np.concatenate(scores_list)
        precision, recall, threshold = precision_recall_curve(bbox_labels_list, scores_list)
        if threshold[0] == 0:
            average_precision = -np.sum(np.diff(recall[1:]) * np.array(precision)[1:-1])
        else:
            average_precision = -np.sum(np.diff(recall) * np.array(precision)[:-1])
        if self.return_prec_rec_curve:
            return average_precision, (precision, recall, threshold)
        return average_precision


if __name__ == "__main__":
    import iosdata.utils.visualization as v

    pos = torch.rand((8192, 3))
    y = torch.zeros((8192, 2))
    obj_mask_1 = (pos[:, 0] > 0.7) & (pos[:, 0] < 0.9) & (pos[:, 1] > 0.7) & (pos[:, 1] < 0.9) & (pos[:, 2] > 0.7) & \
                 (pos[:, 2] < 0.9)
    obj_mask_2 = (pos[:, 0] > 0.1) & (pos[:, 0] < 0.6) & (pos[:, 1] > 0.5) & (pos[:, 1] < 0.6) & (pos[:, 2] > 0.5) & \
                 (pos[:, 2] < 0.8)
    obj_mask_3 = (pos[:, 0] > 0.2) & (pos[:, 0] < 0.4) & (pos[:, 1] > 0.2) & (pos[:, 1] < 0.4) & (pos[:, 2] > 0.2) & \
                 (pos[:, 2] < 0.4)
    y[obj_mask_1 | obj_mask_2 | obj_mask_3, 0] = 1
    y[obj_mask_1, 1] = 1
    y[obj_mask_2, 1] = 2
    y[obj_mask_3, 1] = 3
    obj_ids = y[:, 1]
    pos = pos.numpy()
    color = np.zeros_like(pos)
    color[obj_ids == 1] = [255, 0, 0]
    color[obj_ids == 2] = [0, 255, 0]
    color[obj_ids == 3] = [255, 255, 0]

    pred = torch.full((8192, 2), -1.)
    pred[:, 0] = torch.rand(8192) * -10
    pred_mask_1 = (pos[:, 0] > 0.7) & (pos[:, 0] < 0.9) & (pos[:, 1] > 0.7) & (pos[:, 1] < 0.9) & (pos[:, 2] > 0.7) & \
                 (pos[:, 2] < 0.9)
    # pred_mask_3 = (pos[:, 0] > 0.2) & (pos[:, 0] < 0.3) & (pos[:, 1] > 0.2) & (pos[:, 1] < 0.4) & (pos[:, 2] > 0.2) & \
    #               (pos[:, 2] < 0.4)
    pred_mask_2 = (pos[:, 0] > 0.28) & (pos[:, 0] < 0.4) & (pos[:, 1] > 0.2) & (pos[:, 1] < 0.4) & (pos[:, 2] > 0.2) & \
                  (pos[:, 2] < 0.4)

    # pred[pred_mask_1 | pred_mask_2 | pred_mask_3, 0] = torch.rand((pred_mask_1 | pred_mask_2 | pred_mask_3).sum()) * 10
    pred[pred_mask_1 | pred_mask_2, 0] = torch.rand((pred_mask_1 | pred_mask_2).sum()) * 10
    pred[pred_mask_1, 1] = 1
    pred[pred_mask_2, 1] = 2
    # pred[pred_mask_3, 1] = 3

    pred_ids = pred[:, 1]
    pred_color = np.zeros_like(pos)
    pred_color[pred_ids == 1] = [255, 0, 0]
    pred_color[pred_ids == 2] = [0, 255, 0]
    pred_color[pred_ids == 3] = [255, 255, 0]
    v.add_point_cloud(pos, color)
    v.add_point_cloud(pos, pred_color)

    ap = InstanceAveragePrecision()([pred], [y])
    print(ap)
    v.show_cloud()

    exit()
    jaccard = JaccardScore()
    print('+++ Detection +++')
    target = [torch.tensor([0, 0, 0, 1, 0, 1]),
              torch.tensor([1, 1, 0, 1, 0, 1])]
    output = [torch.tensor([-0.4, -0.1, -0.1, 0.9, -0.1, 0.9]),  # Contains probability that an object is detected
              torch.tensor([0.9, 0.9, -0.1, 0.9, -0.1, 0.9])]
    result_1 = jaccard(output, target)

    print(f'\t[RESULT 1] {"JaccardScore (should be 100%)":55s} = {result_1}')
    output = [torch.tensor([-0.3, -0.1, -0.2, 0.9, 0.8, -0.1]),
              torch.tensor([0.51, -0.2, -0.3, 0.98, -0.3, -0.4])]
    result_2 = jaccard(output, target)
    print(f'\t[RESULT 2] {"JaccardScore (should be less than 100%)":55s} = {result_2}')

    print()
    print('+++ Binary-Label Segmentation +++')
    output = [torch.tensor([[1, 0],
                            [1, 0],
                            [1, 0],
                            [0, 1],
                            [1, 0],
                            [0, 1]]),
              torch.tensor([[0, 1],
                            [0, 1],
                            [1, 0],
                            [0, 1],
                            [1, 0],
                            [0, 1]])]
    result_3 = jaccard(output, target)
    print(f'\t[RESULT 3] {"JaccardScore (should be same result as RESULT 1)":55s} = {result_3}')

    output = [torch.tensor([[1, 0],
                            [1, 0],
                            [1, 0],
                            [0, 1],
                            [0, 1],
                            [1, 0]]),
              torch.tensor([[0, 1],
                            [1, 0],
                            [1, 0],
                            [0, 1],
                            [1, 0],
                            [1, 0]])]
    result_4 = jaccard(output, target)
    print(f'\t[RESULT 4] {"JaccardScore (should be same result as RESULT 2)":55s} = {result_4}')

    print()
    print('+++ Multi-Label Segmentation +++')
    target = [torch.tensor([1, 3, 2, 0]),
              torch.tensor([0, 2, 2, 1])]
    output = [torch.tensor([[0, 1, 0, 0],
                            [0, 0, 0, 1],
                            [0, 0, 1, 0],
                            [1, 0, 0, 0]]),
              torch.tensor([[1, 0, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 1, 0],
                            [0, 1, 0, 0]])]
    result_5 = jaccard(output, target)
    print(f'\t[RESULT 5] {"JaccardScore (should be 100% for each class)":55s} = {result_5}')
    output = [torch.tensor([[0, 1, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [1, 0, 0, 0]]),
              torch.tensor([[0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 1, 0],
                            [0, 1, 0, 0]])]
    result_6 = jaccard(output, target)
    print(f'\t[RESULT 6] {"JaccardScore (some classes should be less than 100%)":55s} = {result_6}')



    # Not runnable with current code
    if False:
        iou = IoU()

        #output = [np.array([1, 1, 2, 2, 3, 3]), np.array([1, 1, 2, 2, 3, 3])]
        #target = [np.array([1, 1, 2, 2, 3, 3]), np.array([1, 1, 2, 2, 3, 3])]

        target = [np.array([1, 3, 2, 1, 0, 0, 1, 3])]
        output = [np.array([0, 3, 2, 2, 1, 0, 1, 2])]
        mean_iou, per_class_iou = iou(output, target)

        print("Mean IoU:" + str(mean_iou) + "\t per class IoU:" + str(per_class_iou))
