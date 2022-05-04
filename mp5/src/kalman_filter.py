import numpy as np
from numpy import dot, zeros, eye, isscalar
from copy import deepcopy
from math import log
import sys

from utils import within_range


class Kalman:
    # we've set the dimensions of x, z for you
    def __init__(self, bbox_3d, info, tracker_id, dim_x=10, dim_z=7, dim_u=0):
        self.initial_pos = bbox_3d  # the initial_pos also initializes self.x below
        self.time_since_update = 0  # keep track of how long since a detection was matched to this tracker
        self.hits = 1  # number of total hits including the first detection
        self.info = info  # some additional info, mainly used for evaluation code
        self.ID = tracker_id  # each tracker has a unique ID, so that we can see consistency across frames

        # -------------------- above are some bookkeeping params, below is the Kalman Filter

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        self.x = zeros((dim_x, 1))  # state
        self.Sigma = eye(dim_x)  # uncertainty covariance
        self.Q = eye(dim_x)  # process uncertainty

        self.A = eye(dim_x)  # state transition matrix
        self.H = zeros((dim_z, dim_x))  # measurement function
        self.R = eye(dim_z)  # measurement uncertainty

        self.z = np.array([[None] * self.dim_z]).T

        # Kalman gain and residual are computed during the update step. We
        # save them so that in case you want to inspect them for various
        # purposes
        self.K = np.zeros((dim_x, dim_z))  # kalman gain
        self.y = zeros((dim_z, 1))
        self.S = np.zeros((dim_z, dim_z))  # system uncertainty
        self.SI = np.zeros((dim_z, dim_z))  # inverse system uncertainty

        # initialize data
        self.x[:7] = self.initial_pos.reshape((7, 1))

        # IMPORTANT, call define_model
        self.define_model()

    # Q3.
    # Here is where you define the model, namely self.A and self.H
    # Though you may likely want to make modifications to uncertainty as well,
    #   ie. self.R, self.Q, self.Sigma
    # TODO: Your code
    def define_model(self):
        # state x dimension 10: x, y, z, theta, l, w, h, dx, dy, dz
        # constant velocity model: x' = x + dx, y' = y + dy, z' = z + dz
        # while all others (theta, l, w, h, dx, dy, dz) remain the same
        # self.A = ...

        # Measurement function, dim_z * dim_x, the first 7 dimensions of the measurement correspond to the state
        # self.H = ...

        # Hint: the initial velocity is very uncertain for new detections

        # --------------------------- Begin your code here ---------------------------------------------
        self.A[list(range(3)), list(reversed(range(-3, 0)))] = 1
        self.H[:self.dim_z, :self.dim_z] = np.eye(self.dim_z)
        self.Q[:self.dim_z, :self.dim_z] = 0

        # --------------------------- End your code here   ---------------------------------------------

    # Q5.
    # TODO: Your code
    def predict(self):
        """
        Predict next state (prior) using the Kalman filter state propagation
        equations.
        Parameters
        ----------
        Note that often we might give predict a control vector u,
        but of course we don't know what controls each tracked object is applying
        """
        # Hint: you should be modifying self.x and self.Sigma
        # --------------------------- Begin your code here ---------------------------------------------
        self.x = self.A @ self.x
        self.Sigma = self.A @ self.Sigma @ self.A.T + self.Q
        # --------------------------- End your code here   ---------------------------------------------

        # Leave this at the end, within_range ensures that the angle is between -pi and pi
        self.x[3] = within_range(self.x[3])
        return

    # Q2 and Q4.
    # TODO: Your code
    def update(self, z, trivial=False):
        """
        Add a new measurement (z) to the Kalman filter.
        ----------
        z : (dim_z, 1): array_like measurement for this update.
        """
        z = z.reshape(-1, 1)

        # --------------------------- Begin your code here ---------------------------------------------
        if trivial:
            self.x[:self.dim_z] = z
            self.x[self.dim_z:] = 0
        else:
            self.y = z - self.H @ self.x
            sigma_ht = self.Sigma @ self.H.T
            self.S = self.H @ sigma_ht + self.R
            self.SI = np.linalg.inv(self.S)
            self.K = sigma_ht @ self.SI
            self.x = self.x + self.K @ self.y
            # P = (I-KH)P(I-KH)' + KRK'
            # self.Sigma = self.Sigma - self.K @ self.H @ self.Sigma
            self.Sigma = (np.eye(self.dim_x) - self.K @ self.H) + self.K @ self.R @ self.K.T

            self.z = z.copy()

        # --------------------------- End your code here   ---------------------------------------------

        # Leave this at the end, within_range ensures that the angle is between -pi and pi
        self.x[3] = within_range(self.x[3])
