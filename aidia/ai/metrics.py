import tensorflow as tf


def iou(box1, box2):
    """Calculates IoU of box1 and box2.

    Parameters
    ----------
    box1: 1D vector [y1, x1, y2, x2]
    box2: 1D vector [y1, x1, y2, x2]

    Returns
    -------
    iou: float
        Intersection Over Union value.
    """
    # Calculate intersection areas.
    y1 = max(box1[0], box2[0])
    y2 = min(box1[2], box2[2])
    x1 = max(box1[1], box2[1])
    x2 = min(box1[3], box2[3])
    intersection = max(x2 - x1, 0) * max(y2 - y1, 0)

    # Compute IoU.
    box1_area = max(box1[2] - box1[0], 0) * max(box1[3] - box1[1], 0)
    box2_area = max(box2[2] - box2[0], 0) * max(box2[3] - box2[1], 0)
    union = box1_area + box2_area - intersection
    iou = intersection / union
    return iou


class MultiMetrics(tf.keras.metrics.Metric):
    def __init__(self, threshold=0.5, class_id=None, name='MultiMetrics', **kwargs):
        super().__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.true_negatives = self.add_weight(name="tn", initializer="zeros")
        # self.false_positives = self.add_weight(name="fp", initializer="zeros")
        # self.false_negatives = self.add_weight(name="fn", initializer="zeros")
        self.sum_ytrue = self.add_weight(name='tp+fn', initializer='zeros')
        self.sum_ypred = self.add_weight(name='tp+fp', initializer='zeros')
        self.sum_inv_ytrue = self.add_weight(name='tn+fp', initializer='zeros')
        # self.sum_inv_ypred = self.add_weight(name='tn+fn', initializer='zeros')
        self.threshold = threshold
        self.class_id = class_id

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.class_id is not None:
            y_true = y_true[..., self.class_id]
            y_pred = y_pred[..., self.class_id]
        y_true = tf.cast(y_true, tf.bool)
        inv_y_true = tf.logical_not(y_true)
        y_pred = tf.greater_equal(y_pred, self.threshold)
        inv_y_pred = tf.logical_not(y_pred)

        # TP
        values = tf.logical_and(y_true, y_pred)
        values = tf.cast(values, self.dtype)
        self.true_positives.assign_add(tf.reduce_sum(values))
        self.true_positives.assign_add(tf.cast(1e-6, self.dtype))

        # TN
        values = tf.logical_and(inv_y_true, inv_y_pred)
        values = tf.cast(values, self.dtype)
        self.true_negatives.assign_add(tf.reduce_sum(values))
        self.true_negatives.assign_add(tf.cast(1e-6, self.dtype))

        # FP
        # values = tf.logical_and(inv_y_true, _y_pred)
        # values = tf.cast(values, self.dtype)
        # self.false_positives.assign_add(tf.reduce_sum(values))

        # FN
        # values = tf.logical_and(y_true, inv_y_pred)
        # values = tf.cast(values, self.dtype)
        # self.false_negatives.assign_add(tf.reduce_sum(values))

        # TP + FN
        y_true = tf.cast(y_true, self.dtype)
        self.sum_ytrue.assign_add(tf.reduce_sum(y_true))
        self.sum_ytrue.assign_add(tf.cast(1e-6, self.dtype))

        # TP + FP
        y_pred = tf.cast(y_pred, self.dtype)
        self.sum_ypred.assign_add(tf.reduce_sum(y_pred))
        self.sum_ypred.assign_add(tf.cast(1e-6, self.dtype))

        # TN + FP
        inv_y_true = tf.cast(inv_y_true, self.dtype)
        self.sum_inv_ytrue.assign_add(tf.reduce_sum(inv_y_true))
        self.sum_inv_ytrue.assign_add(tf.cast(1e-6, self.dtype))

        # TN + FN
        # inv_y_pred = tf.cast(inv_y_pred, self.dtype)
        # self.sum_inv_ypred.assign_add(tf.reduce_sum(inv_y_pred))
    
    def result(self):
        precision = tf.divide(self.true_positives, self.sum_ypred)
        recall = tf.divide(self.true_positives, self.sum_ytrue)
        specificity = tf.divide(self.true_negatives, self.sum_inv_ytrue)
        tpr = recall
        fpr = tf.subtract(tf.cast(1, self.dtype), specificity)
        a = tf.multiply(tf.multiply(precision, recall), tf.cast(2, self.dtype))
        b = tf.add(precision, recall)
        f1 = tf.divide(a, b)
        return [precision, recall, specificity, tpr, fpr, f1]

        
