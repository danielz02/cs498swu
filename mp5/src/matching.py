from itertools import product
from typing import List

import numpy as np
from matching_utils import iou
from mp5.src.utils import Box3D


def data_association(dets: List[Box3D], trks: List[Box3D], threshold=-0.2, algm='greedy'):
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
    # Hint: you should use the provided iou(box_a, box_b) function to compute distance/cost between pairs of box3d
    # objects iou() is an implementation of a 3D box IoU

    # --------------------------- Begin your code here ---------------------------------------------
    if len(dets) == 0:
        return np.array([]), np.array([]), np.arange(len(trks))
    elif len(trks) == 0:
        return np.array([]), np.arange(len(dets)), np.array([])

    matches = []
    pairwise_iou = np.vectorize(lambda x, y: iou(dets[x], trks[y]))(
        np.arange(len(dets))[:, np.newaxis], np.arange(len(trks))
    )
    pairwise_iou[pairwise_iou < threshold] = np.nan
    for i, match in enumerate(pairwise_iou):
        if np.all(np.isnan(match)):
            # unmatched_dets.append(i)
            continue
        trk_match = np.nanargmax(match)
        matches.append([i, trk_match])
        pairwise_iou[:, trk_match] = np.nan
    matches = np.array(matches)
    if len(matches) == 0:
        unmatched_dets = np.arange(len(dets))
        unmatched_trks = np.arange(len(trks))
    else:
        unmatched_dets = np.setdiff1d(np.arange(len(dets)), matches[:, 0])
        unmatched_trks = np.setdiff1d(np.arange(len(trks)), matches[:, 1])

    # matches = np.stack([np.arange(len(dets)), np.nanargmax(pairwise_iou, axis=1)], axis=1)
    # matches = matches[~np.all(np.isinf(pairwise_iou), axis=1)]
    # unmatched_dets = np.setdiff1d(np.arange(len(dets)), matches[:, 0])
    # unmatched_trks = np.setdiff1d(np.arange(len(trks)), matches[:, 1])

    # --------------------------- End your code here   ---------------------------------------------

    return matches, unmatched_dets, unmatched_trks
