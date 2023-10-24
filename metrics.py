import numpy as np

# metrics assume predictions and ground-truths are Boolean or binary-valued

def precision(true_positives, predicted_positives):
    return true_positives/max(predicted_positives, 1)

def recall(true_positives, positives):
    return true_positives/max(positives, 1)

def f_measure(precision, recall):
    if (precision+recall):
        return 2*precision*recall/(precision+recall)
    else:
        return 0

def get_true_positives(pred, target):
    return np.sum(pred*target)

def get_predicted_positives(pred):
    return np.sum(pred)

def get_positives(target):
    return np.sum(target)

def get_image_level_metrics(pred, target):
    TP = get_true_positives(pred, target)
    PP = get_predicted_positives(pred)
    P = get_positives(target)
    return TP, PP, P

def get_blob_metrics(pred, blob_target, ioa_threshold):
    '''
    Compute blob recall
    '''
    n_blobs = blob_target.shape[0]
    n_recovered = 0
    for i in range(n_blobs):
        curr_blob = blob_target[i]
        overlap = curr_blob*pred
        if np.sum(overlap)/np.sum(curr_blob) >= ioa_threshold:
            n_recovered += 1
    return n_recovered, n_recovered/n_blobs
