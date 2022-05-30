import tensorflow as tf
from tensorflow import keras


class MultiClassConfusionMatrix(keras.metrics.Metric):

    def __init__(self, num_classes, name="multi_class_confusion_matrix", **kwargs):
        super(MultiClassConfusionMatrix, self).__init__(name=name, **kwargs)

        self.num_classes = num_classes
        
        self.mctp_conf_matrix = self.add_weight(
            name="confusion_matrix", shape=(num_classes, num_classes), 
            dtype=tf.dtypes.int32, initializer='zeros'
        )
        
    def update_state(self, y_true, y_pred, sample_weight=None):

        y_true = tf.argmax(y_true, axis=1)
        y_pred = tf.argmax(y_pred, axis=1)
        
        tmp = tf.math.confusion_matrix(y_true, y_pred, self.num_classes)
        res = tf.math.add(tmp, self.mctp_conf_matrix)

        self.mctp_conf_matrix.assign(res)

    def result(self):
        return self.mctp_conf_matrix
    
    def reset_state(self):
        self.mctp_conf_matrix.assign(
            tf.zeros((self.num_classes, self.num_classes), 
            dtype=tf.dtypes.int32))


class MultiClassMeanRecall(keras.metrics.Metric):

    def __init__(self, num_classes, name="mean_recall", **kwargs):
        super(MultiClassMeanRecall, self).__init__(name=name, **kwargs)

        self.num_classes = num_classes
        
        self.mctp_conf_matrix = self.add_weight(
            name="confusion_matrix", shape=(num_classes, num_classes), 
            dtype=tf.dtypes.int32, initializer='zeros'
        )
        
    def update_state(self, y_true, y_pred, sample_weight=None):

        y_true = tf.argmax(y_true, axis=1)
        y_pred = tf.argmax(y_pred, axis=1)
        
        tmp = tf.math.confusion_matrix(y_true, y_pred, self.num_classes)
        res = tf.math.add(tmp, self.mctp_conf_matrix)

        self.mctp_conf_matrix.assign(res)

    def result(self):
        return tf.reduce_mean(
            tf.linalg.diag_part(self.mctp_conf_matrix)/tf.reduce_sum(self.mctp_conf_matrix, axis=1)
            )
    
    def reset_state(self):
        self.mctp_conf_matrix.assign(
            tf.zeros((self.num_classes, self.num_classes), 
            dtype=tf.dtypes.int32))


class MultiClassMeanPrecision(keras.metrics.Metric):

    def __init__(self, num_classes, name="mean_precision", **kwargs):
        super(MultiClassMeanPrecision, self).__init__(name=name, **kwargs)

        self.num_classes = num_classes
        
        self.mctp_conf_matrix = self.add_weight(
            name="confusion_matrix", shape=(num_classes, num_classes), 
            dtype=tf.dtypes.int32, initializer='zeros'
        )
        
    def update_state(self, y_true, y_pred, sample_weight=None):

        y_true = tf.argmax(y_true, axis=1)
        y_pred = tf.argmax(y_pred, axis=1)
        
        tmp = tf.math.confusion_matrix(y_true, y_pred, self.num_classes)
        res = tf.math.add(tmp, self.mctp_conf_matrix)

        self.mctp_conf_matrix.assign(res)

    def result(self):
        return tf.reduce_mean(
            tf.linalg.diag_part(self.mctp_conf_matrix)/tf.reduce_sum(self.mctp_conf_matrix, axis=0)
            )
    
    def reset_state(self):
        self.mctp_conf_matrix.assign(
            tf.zeros((self.num_classes, self.num_classes), 
            dtype=tf.dtypes.int32)) 