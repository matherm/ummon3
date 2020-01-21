import numpy as np
import torch

from .base import *

__all__ =['IoU']

class SegmentationMetrics(OfflineMetric):
    def __call__(self, output, target):
        #results = [self.func(c1, c2) for c1, c2 in zip(output, target)]
        result = self.func(output,target)
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


if __name__ == "__main__":
    iou = IoU()

    #output = [np.array([1, 1, 2, 2, 3, 3]), np.array([1, 1, 2, 2, 3, 3])]
    #target = [np.array([1, 1, 2, 2, 3, 3]), np.array([1, 1, 2, 2, 3, 3])]

    target = [np.array([1, 3, 2, 1, 0, 0, 1, 3])]
    output = [np.array([0, 3, 2, 2, 1, 0, 1, 2])]
    mean_iou, per_class_iou = iou(output, target)

    print("Mean IoU:" + str(mean_iou) + "\t per class IoU:" + str(per_class_iou))