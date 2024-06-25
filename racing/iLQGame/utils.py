"""
Supporting functions.

Please contact the author(s) of this library if you have any questions.
Author: Haimin Hu (haiminh@princeton.edu)
"""

import numpy as np
import ml_collections
from typing import Tuple, Sequence

from jax import lax
import jax.numpy as jnp
from flax import struct
from flax import linen as nn

from iLQR import Struct


class ExplicitMLP(nn.Module):
  features: Sequence[int]
  positive_dims: Sequence[int] = None
  cutoff: bool = False
  config: Struct = None
  px_rel_range: float = None
  px_rel_min: float = None
  py_rel_range: float = None
  py_rel_min: float = None
  is_softmax: bool = False

  def setup(self):
    self.layers = [nn.Dense(feat) for feat in self.features]

  def __call__(self, inputs):

    def _is_interactive(px_rel, py_rel):
      px_rel_raw = px_rel * self.px_rel_range + self.px_rel_min
      py_rel_raw = py_rel * self.py_rel_range + self.py_rel_min
      ego_ado_sep_raw = jnp.sqrt(px_rel_raw**2 + py_rel_raw**2)
      return (self.config.SEP_LB <= ego_ado_sep_raw) & (ego_ado_sep_raw <= self.config.SEP_UB)

    def _true_fn(x):
      return x

    def _false_fn(x):
      return 0. * x

    def _clip_wgt(weight):

      def _true_fn(weight):
        return weight

      def _false_fn(weight):
        return 0.

      return lax.cond(weight >= 0.55, _true_fn, _false_fn, weight)

    x = inputs
    for i, lyr in enumerate(self.layers):
      x = lyr(x)
      if i != len(self.layers) - 1:
        x = nn.relu(x)

    if self.positive_dims:
      for dim in self.positive_dims:
        x = x.at[dim].set(nn.softplus(x[dim]))

    if self.is_softmax:
      turn_wgt_ego = nn.softmax(jnp.array((x[2], x[3])))
      turn_wgt_ado = nn.softmax(jnp.array((x[5], x[6])))
      x = x.at[2].set(self.config.TURN_WGT_EGO * _clip_wgt(turn_wgt_ego[0]))
      x = x.at[3].set(self.config.TURN_WGT_EGO * _clip_wgt(turn_wgt_ego[1]))
      x = x.at[5].set(self.config.TURN_WGT_ADO * _clip_wgt(turn_wgt_ado[0]))
      x = x.at[6].set(self.config.TURN_WGT_ADO * _clip_wgt(turn_wgt_ado[1]))

    if self.cutoff:
      return lax.cond(_is_interactive(px_rel=inputs[0], py_rel=inputs[1]), _true_fn, _false_fn, x)

    return x


def get_nn_config(config: Struct):
  """Gets the default hyperparameter configuration for the general MLP."""
  config_nn = ml_collections.ConfigDict()
  config_nn.learning_rate = config.LEARNING_RATE
  config_nn.random_seed = config.RANDOM_SEED
  config_nn.network_dim_in = 26
  config_nn.network_dim_out = 7  # OV, ST, Le, Re, BL, La, Ra
  config_nn.network_features = [256, 256, 256, 256, 256, config_nn.network_dim_out]
  config_nn.positive_dims = []
  return config_nn


@struct.dataclass
class StaticCostParams():
  W_OVERTAKE: float = 0.
  W_STAYBACK: float = 0.
  W_INNER_EGO: float = 0.
  W_OUTER_EGO: float = 0.
  W_BLOCK: float = 0.
  W_INNER_ADO: float = 0.
  W_OUTER_ADO: float = 0.


def is_safe(config: Struct, obs_tuple: Tuple, road_thr_scale=1.3, coll_thres=4.0) -> bool:
  """
  Checks safety for the ego and ado cars.
  """

  _r = config.WIDTH / 2.
  road_thr = config.ROAD_THR * road_thr_scale
  _circ_r = 2.0  # [m]
  _circ_sep = 2. * _circ_r  # [m]

  state, _, closest_pt, slope, _ = obs_tuple
  state = np.asarray(state[:, 0])
  closest_pt = np.asarray(closest_pt[:, 0])
  slope = np.asarray(slope[:, 0])

  sr = np.sin(slope[0])
  cr = np.cos(slope[0])
  dx = state[0] - closest_pt[0]
  dy = state[1] - closest_pt[1]
  dis = sr*dx - cr*dy

  # Checks the right track boundary (outer edge).
  width_R = np.sqrt((closest_pt[0] - closest_pt[4])**2 + (closest_pt[1] - closest_pt[5])**2)
  b_r = dis - (width_R-_r+road_thr)

  if b_r > 0.:
    return False

  # Checks the left track boundary (inner edge).
  width_L = np.sqrt((closest_pt[0] - closest_pt[2])**2 + (closest_pt[1] - closest_pt[3])**2)
  b_l = -dis - (width_L-_r+road_thr)

  if b_l > 0.:
    return False

  # Checks collision.
  ego_x, ego_y, ego_psi = state[0], state[1], state[3]
  ado_x, ado_y, ado_psi = state[4], state[5], state[7]
  centers_ego = np.vstack((
      np.array((ego_x, ego_y)),
      np.array((ego_x + _circ_sep * np.cos(ego_psi), ego_y + _circ_sep * np.sin(ego_psi))),
  ))
  centers_ado = np.vstack((
      np.array((ado_x, ado_y)),
      np.array((ado_x + _circ_sep * np.cos(ado_psi), ado_y + _circ_sep * np.sin(ado_psi))),
  ))
  for ii in range(2):
    for jj in range(2):
      dx = centers_ego[ii, 0] - centers_ado[jj, 0]
      dy = centers_ego[ii, 1] - centers_ado[jj, 1]
      dist = np.sqrt(dx**2 + dy**2) - _circ_sep * config.PENALTY_SCALE + coll_thres
      if dist < 0.:
        return False

  return True


def get_perfect_obs_two_player(
    x_cur: np.ndarray, u_cur: np.ndarray, solver, data_tuple_in: Tuple = None
) -> Tuple:
  """
  Gets the iLQGame planning data (one cycle). This is only for the 2-player case (ego car, ado car).
  The output contains observation information needed for solving the inverse game.

  Args:
      x_cur (np.ndarray): current state (8,)
      u_cur (np.ndarray): current controls (2, N_sys), where N_sys = 2
      solver (iLQSolver)
      data_tuple_in (Tuple): (xs, us, closest_pts, slopes, thetas) Defaults to None.

  Returns:
      data_tuple_out (Tuple)
  """
  x_cur = x_cur[:, np.newaxis]
  u_cur = u_cur[:, np.newaxis, :]

  track = solver.track
  closest_pt_ref, slope, theta = track.get_closest_pts(x_cur[:2], mode='ref')
  closest_pt_inner, _, _ = track.get_closest_pts(x_cur[:2], mode='inner')
  closest_pt_outer, _, _ = track.get_closest_pts(x_cur[:2], mode='outer')
  closest_pt_ref_vel, _, _ = track.get_closest_pts(x_cur[:3], mode='ref_vel')
  closest_pt = np.hstack(
      (closest_pt_ref, closest_pt_inner, closest_pt_outer, closest_pt_ref_vel[2:3])
  )[:, np.newaxis]
  slope = slope[:, np.newaxis]
  theta = np.array([[theta]])

  if data_tuple_in is None:
    data_tuple_out = (x_cur, u_cur, closest_pt, slope, theta)
  else:
    xs, us, closest_pts, slopes, thetas = data_tuple_in
    xs = np.hstack((xs, x_cur))
    us = np.concatenate((us, u_cur), axis=1)
    closest_pts = np.hstack((closest_pts, closest_pt))
    slopes = np.hstack((slopes, slope))
    thetas = np.hstack((thetas, theta))
    data_tuple_out = (xs, us, closest_pts, slopes, thetas)

  return data_tuple_out
