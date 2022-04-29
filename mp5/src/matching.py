import numpy as np
from matching_utils import iou

def data_association(dets, trks, threshold=0.2, algm='greedy'):
    """
    Q1. Assigns detections to tracked object

    dets:       a list of Box3D object
    trks:       a list of Box3D object
    threshold:  only mark a det-trk pair as a match if their iou distance is less than the threshold
    algm:       for extra credit, implement the hungarian algorithm as well

    Returns 3 lists:
        matches, kx2 np array of match indices, i.e. [[det_idx1, trk_idx1], [det_idx2, trk_idx2], ...]
        unmatched_dets, a 1d array of indices of unmatched detections
        unmatched_trks, a 1d array of indices of unmatched trackers
    """
    # Hint: you should use the provided iou(box_a, box_b) function to compute distance/cost between pairs of box3d objects
    # iou() is an implementation of a 3D box IoU

    matches = []
    unmatched_dets = []
    unmatched_trks = []
    # --------------------------- Begin your code here ---------------------------------------------


    # --------------------------- End your code here   ---------------------------------------------

    return matches, np.array(unmatched_dets), np.array(unmatched_trks)