"""
Racing track.

Please contact the author(s) of this library if you have any questions.
Author:  Haimin Hu (haiminh@princeton.edu)
"""

import numpy as np
from typing import Tuple
from matplotlib import pyplot as plt
from pyspline.pyCurve import Curve


class Track:

  def __init__(
      self, center_line: np.ndarray = None, width_left: float = None, width_right: float = None,
      loop: bool = True
  ):
    """
    Constructs a track with a fixed width.

    Args:
        center_line (np.ndarray, optional): 2D numpy array containing samples
            of track center line [[x1,x2,...], [y1,y2,...]]. Defaults to None.
        width_left (float, optional): width of the track on the left side.
            Defaults to None.
        width_right (float, optional): width of the track on the right side.
            Defaults to None.
        loop (bool, optional): flag indicating if the track has loop.
            Defaults to True.
    """
    self.width_left = width_left
    self.width_right = width_right
    self.loop = loop

    _spline_order = 3

    if center_line is not None:
      self.center_line = Curve(x=center_line[0, :], y=center_line[1, :], k=_spline_order)
      self.length = self.center_line.getLength()
    else:
      self.length = None
      self.center_line = None

    self.track_bound = None
    self.track_center = None

  def _interp_s(self, s: list) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a list of s (progress since start), returns corresponing (x,y) points
    on the track. In addition, return slope of trangent line on those points.

    Args:
        s (list): progress since start.

    Returns:
        np.ndarray: (x,y) points on the track.
        np.ndarray: slopes of trangent line.
    """
    try:
      n = len(s)
    except:
      s = np.array([s])
      n = len(s)

    interp_pt = self.center_line.getValue(s)
    slope = np.zeros(n)

    for i in range(n):
      deri = self.center_line.getDerivative(s[i])
      slope[i] = np.arctan2(deri[1], deri[0])
    return interp_pt.T, slope

  def interp(self, theta_list: list) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a list of theta (progress since start), return corresponing (x,y)
    points on the track. In addition, return slope of trangent line on those
    points.

    Args:
        theta_list (list): progress since start

    Returns:
        np.ndarray: (x,y) points on the track.
        np.ndarray: slopes of trangent line.
    """
    if self.loop:
      s = np.remainder(theta_list, self.length) / self.length
    else:
      s = np.array(theta_list) / self.length
      s[s > 1] = 1
    return self._interp_s(s)

  def get_closest_pts(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Gets the closest points (w.r.t. the center line).

    Args:
        points (np.ndarray): points on the track of [2xn] shape.

    Returns:
        np.ndarray: (x,y) points on the track.
        np.ndarray: slopes of trangent line.
        np.ndarray: projected progress
    """
    s, _ = self.center_line.projectPoint(points.T, eps=1e-2)
    closest_pt, slope = self._interp_s(s)
    return closest_pt, slope, s * self.length

  def get_track_width(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gets the width of the track.

    Args:
        theta (np.ndarray): progress since start.

    Returns:
        np.ndarray: left width array.
        np.ndarray: right width array.
    """
    temp = np.ones_like(theta)
    return self.width_left * temp, self.width_right * temp

  def plot_track(self, custom_bound=False, N_pts=400):
    """
    Plots the track.
    """
    N = N_pts
    if custom_bound or self.track_bound is None:
      theta_sample = np.linspace(0, 1, N, endpoint=False) * self.length
      interp_pt, slope = self.interp(theta_sample)

      if self.loop:
        self.track_bound = np.zeros((4, N + 1))
      else:
        self.track_bound = np.zeros((4, N))

      self.track_bound[0, :N] = interp_pt[0, :] - np.sin(slope) * self.width_left
      self.track_bound[1, :N] = interp_pt[1, :] + np.cos(slope) * self.width_left

      self.track_bound[2, :N] = interp_pt[0, :] + np.sin(slope) * self.width_right
      self.track_bound[3, :N] = interp_pt[1, :] - np.cos(slope) * self.width_right

      if self.loop:
        self.track_bound[:, -1] = self.track_bound[:, 0]

    plt.plot(self.track_bound[0, :], self.track_bound[1, :], 'k-')
    plt.plot(self.track_bound[2, :], self.track_bound[3, :], 'k-')

  def plot_track_center(self):
    """
    Plots the center of the track.
    """
    N = 500
    if self.track_center is None:
      theta_sample = np.linspace(0, 1, N, endpoint=False) * self.length
      interp_pt, slope = self.interp(theta_sample)
      self.track_center = interp_pt
      print(len(slope))

    plt.plot(self.track_center[0, :], self.track_center[1, :], 'r--')


class TrackVariableWidth(Track):

  def __init__(
      self, ref_line: np.ndarray, inner_edge: np.ndarray, outer_edge: np.ndarray, loop: bool = True
  ):
    """
    Constructs a track with variable width.

    Args:
        ref_line (np.ndarray, optional): 4D numpy array containing samples of racing line, height,
            and ref speed [[x1,x2,...], [y1,y2,...], [z1,z2,...], [v1,v2,...]].
        inner_edge (np.ndarray, optional): 2D numpy array containing samples of inner track
            boundary [[x1,x2,...], [y1,y2,...]].
        outer_edge (np.ndarray, optional): 2D numpy array containing samples of outer track
            boundary [[x1,x2,...], [y1,y2,...]].
        loop (bool, optional): flag indicating if the track has loop. Defaults to True.
    """

    center_line = (inner_edge[:2, :] + outer_edge[:2, :]) / 2.

    super(TrackVariableWidth, self).__init__(center_line=center_line, loop=loop)

    _spline_order = 3
    self.ref_line = Curve(x=ref_line[0, :], y=ref_line[1, :], k=_spline_order)
    self.ref_line_with_vel = Curve(
        x=ref_line[0, :], y=ref_line[1, :], z=ref_line[3, :], k=_spline_order
    )
    self.ref_line_xyz = Curve(x=ref_line[0, :], y=ref_line[1, :], z=ref_line[2, :], k=_spline_order)
    self.inner_edge = Curve(x=inner_edge[0, :], y=inner_edge[1, :], k=_spline_order)
    self.outer_edge = Curve(x=outer_edge[0, :], y=outer_edge[1, :], k=_spline_order)

  def _interp_s(self, s: list, mode: str, return_slope=True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a list of s (progress since start), returns corresponing (x,y) points on the track.
    In addition, return slope of trangent line on those points.

    Args:
        s (list): progress since start.
        mode (str): 'ref', 'ref_vel', 'inner', 'outer', 'center', 'xyz'

    Returns:
        np.ndarray: (x,y) points on the track.
        np.ndarray: slopes of trangent line.
    """
    try:
      n = len(s)
    except:
      s = np.array([s])
      n = len(s)

    if mode == 'ref':
      interp_pt = self.ref_line.getValue(s)
    elif mode == 'ref_vel':
      interp_pt = self.ref_line_with_vel.getValue(s)
    elif mode == 'inner':
      interp_pt = self.inner_edge.getValue(s)
    elif mode == 'outer':
      interp_pt = self.outer_edge.getValue(s)
    elif mode == 'center':
      interp_pt = self.center_line.getValue(s)
    elif mode == 'xyz':
      interp_pt = self.ref_line_xyz.getValue(s)
    else:
      raise ValueError

    slope = np.zeros(n)

    if return_slope:
      for i in range(n):
        if mode == 'ref':
          deri = self.ref_line.getDerivative(s[i])
        elif mode == 'ref_vel':
          deri = self.ref_line_with_vel.getDerivative(s[i])
        elif mode == 'inner':
          deri = self.inner_edge.getDerivative(s[i])
        elif mode == 'outer':
          deri = self.outer_edge.getDerivative(s[i])
        elif mode == 'center':
          deri = self.center_line.getDerivative(s[i])
        elif mode == 'xyz':
          deri = self.ref_line_xyz.getDerivative(s)
        else:
          raise ValueError
        slope[i] = np.arctan2(deri[1], deri[0])

    return interp_pt.T, slope

  def interp(self, theta_list: list, mode: str, return_slope=True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a list of theta (progress since start), return corresponing (x,y) points on the track.
    In addition, return slope of trangent line on those points.

    Args:
        theta_list (list): progress since start.
        mode (str): 'ref', 'ref_vel', 'inner', 'outer', 'center'.

    Returns:
        np.ndarray: (x,y) points on the track.
        np.ndarray: slopes of trangent line.
    """
    if self.loop:
      s = np.remainder(theta_list, self.length) / self.length
    else:
      s = np.array(theta_list) / self.length
      s[s > 1] = 1
    return self._interp_s(s, mode, return_slope)

  def get_closest_pts(self, points: np.ndarray, mode) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Gets the closest points.

    Args:
        points (np.ndarray): points on the track of [2xn] shape.
        mode (str): 'ref', 'ref_vel', 'inner', 'outer', 'center', 'xyz'

    Returns:
        np.ndarray: (x,y) points on the track.
        np.ndarray: slopes of trangent line.
        np.ndarray: projected progress
    """
    if mode == 'ref':
      s, _ = self.ref_line.projectPoint(points.T, eps=1e-3)
    elif mode == 'ref_vel':
      s, _ = self.ref_line_with_vel.projectPoint(points.T, eps=1e-3)
    elif mode == 'inner':
      s, _ = self.inner_edge.projectPoint(points.T, eps=1e-3)
    elif mode == 'outer':
      s, _ = self.outer_edge.projectPoint(points.T, eps=1e-3)
    elif mode == 'center':
      s, _ = self.center_line.projectPoint(points.T, eps=1e-3)
    elif mode == 'xyz':
      s, _ = self.ref_line_xyz.projectPoint(points.T, eps=1e-3)
    else:
      raise ValueError

    closest_pt, slope = self._interp_s(s, mode)
    return closest_pt, slope, s * self.length

  def plot_track(self, N_pts=1000, plot_raceline=True):
    """
    Plots the track.
    """
    N = N_pts

    theta_sample = np.linspace(0, 1, N, endpoint=False) * self.length

    interp_pt_ref, _ = self.interp(theta_sample, mode='ref', return_slope=False)
    interp_pt_inner, _ = self.interp(theta_sample, mode='inner', return_slope=False)
    interp_pt_outer, _ = self.interp(theta_sample, mode='outer', return_slope=False)
    # interp_pt_center, _ = self.interp(theta_sample, mode='center')

    interp_pt_ref = np.hstack((interp_pt_ref, interp_pt_ref[:, :1]))
    interp_pt_inner = np.hstack((interp_pt_inner, interp_pt_inner[:, :1]))
    interp_pt_outer = np.hstack((interp_pt_outer, interp_pt_outer[:, :1]))

    if plot_raceline:
      plt.plot(interp_pt_ref[0, :], interp_pt_ref[1, :], 'k--')
    plt.plot(interp_pt_inner[0, :], interp_pt_inner[1, :], 'k-')
    plt.plot(interp_pt_outer[0, :], interp_pt_outer[1, :], 'k-')
    # plt.plot(interp_pt_center[0, :], interp_pt_center[1, :], 'k--')


if __name__ == '__main__':
  import csv

  track_file = 'tracks/thunderhill_track.csv'

  x_ref = []
  y_ref = []
  z_ref = []
  v_ref = []
  x_inner = []
  y_inner = []
  x_outer = []
  y_outer = []

  with open(track_file, newline='') as f:
    spamreader = csv.reader(f, delimiter=',')
    for i, row in enumerate(spamreader):
      if i > 0:
        x_ref.append(float(row[3]))
        y_ref.append(float(row[4]))
        z_ref.append(float(row[5]))
        v_ref.append(float(row[6]))
        x_inner.append(float(row[7]))
        y_inner.append(float(row[8]))
        x_outer.append(float(row[9]))
        y_outer.append(float(row[10]))

  ref_line = np.array([x_ref, y_ref, z_ref, v_ref])
  inner_edge = np.array([x_inner, y_inner])
  outer_edge = np.array([x_outer, y_outer])
  track = TrackVariableWidth(
      ref_line=ref_line, inner_edge=inner_edge, outer_edge=outer_edge, loop=True
  )

  track.plot_track()
  plt.axis('equal')
  plt.show()
