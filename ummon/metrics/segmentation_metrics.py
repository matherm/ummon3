import numpy as np
import torch

__all__ =['IoU']

class SegmentationMetrics():
    def __call__(self, output: list, target: list, input):
        results = [self.func(c1, c2, input.category) for c1, c2 in zip(output, target)]
        return [i[0] for i in results], [i[1] for i in results]

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
        self.func = self.calc_iou

    @staticmethod
    def calc_iou(output : np.array, target : np.array, categories):
        #target = [np.array([1, 3, 2, 1, 0, 0, 1, 3])]
        #output = [np.array([0, 3, 2, 2, 1, 0, 1, 2])]

        # class:    0       1       2       3
        #-----------------------------------------------------------------------
        # TP        1       1       1       1
        # TN        5       4       5       6
        # FP        1       1       2       0
        # FN        1       2       0       1

        num_classes_target = len(np.unqiue(target))
        num_classes_output = len(np.unique(output))

        if num_classes_output != num_classes_target:
            print("Number of classes in target and output does not match!\n");

        num_classes = num_classes_target

        conf_matrix = np.zeros((num_classes, num_classes))
        for o, t in zip(output, target):
            conf_matrix[o, t] += 1

        print(conf_matrix)

        #debug_conf_matrix_entries= []
        per_class_iou = []
        TP = TN = FP = FN = 0
        for i in range(0,num_classes):
            TP = conf_matrix[i, i]                  # True Positives. Predicted correct target
            FP = np.sum(conf_matrix[i, :]) - TP     # False Positives. Predicted true class is false target
            FN = np.sum(conf_matrix[:, i]) - TP     # False Negatives. Predicted false class is true target
            TN = np.sum(conf_matrix) - (TP+FP+FN)   # True Negatives. Predicted false class is false target

            #debug_conf_matrix_entries.append((TP, TN, FP,FN))

            intersection = TP
            union = TP + FP + FN
            per_class_iou.append(intersection / union)

        #print(debug_conf_matrix_entries)
        mean_iou = np.mean(per_class_iou)

        return mean_iou, per_class_iou

if __name__ == "__main__":
    iou = IoU()

    #output = [np.array([1, 1, 2, 2, 3, 3]), np.array([1, 1, 2, 2, 3, 3])]
    #target = [np.array([1, 1, 2, 2, 3, 3]), np.array([1, 1, 2, 2, 3, 3])]

    target = [np.array([1, 3, 2, 1, 0, 0, 1, 3])]
    output = [np.array([0, 3, 2, 2, 1, 0, 1, 2])]
    mean_iou, per_class_iou = iou(output, target, 4)

    print("Mean IoU:" + str(mean_iou) + "\t per class IoU:" + str(per_class_iou))


