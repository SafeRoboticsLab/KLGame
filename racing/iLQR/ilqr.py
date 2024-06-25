"""
Jaxified iterative Linear Quadrative Regulator (iLQR).

Please contact the author(s) of this library if you have any questions.
Author:  Haimin Hu (haiminh@princeton.edu)
"""

import time
import numpy as np
from typing import Tuple

from .dynamics import Dynamics
from .cost import Cost, Cost_racing

from functools import partial
from jax import jit, lax
from jaxlib.xla_extension import ArrayImpl
import jax.numpy as jnp


class iLQR():

  def __init__(self, track, config, safety=True):

    self.track = track
    self.horizon = config.N
    self.max_iter = config.MAX_ITER

    self.tol = 1e-2
    self.lambad = 10
    self.lambad_min = 1e-3

    self.dynamics = Dynamics(config)
    self.alphas = 1.1**(-np.arange(10)**2)

    self.dim_x = self.dynamics.dim_x
    self.dim_u = self.dynamics.dim_u

    self.cost = Cost(config, safety)

  def forward_pass(
      self, nominal_states: ArrayImpl, nominal_controls: ArrayImpl, Ks: ArrayImpl, ks: ArrayImpl,
      alpha: float
  ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Forward pass wrapper.
    Calls forward_pass_jax to speed up computation.

    Args:
        nominal_states (ArrayImpl): (dim_x, N)
        nominal_controls (ArrayImpl): (dim_u, N)
        Ks (ArrayImpl): gain matrices (dim_u, dim_x, N - 1)
        ks (ArrayImpl): gain vectors (dim_u, N - 1)
        alpha (float): scalar parameter

    Returns:
        X (np.ndarray): (dim_x, N)
        U (np.ndarray): (dim_u, N)
        J (float): total cost.
        closest_pts (np.ndarray): 2xN array of each state's closest point [x,y] on the track.
        slope (np.ndarray): 1xN array of track's slopes (rad) at closest points.
        theta (np.ndarray): 1xN array of the progress at each state.
    """
    Xs, Us = self.forward_pass_jax(nominal_states, nominal_controls, Ks, ks, alpha)
    Xs = np.asarray(Xs)
    Us = np.asarray(Us)

    closest_pt, slope, theta = self.track.get_closest_pts(Xs[:2, :])
    J = self.cost.get_cost(Xs, Us, closest_pt, slope, theta)
    return Xs, Us, J, closest_pt, slope, theta

  def backward_pass(
      self, nominal_states: ArrayImpl, nominal_controls: ArrayImpl, closest_pts: ArrayImpl,
      slopes: ArrayImpl
  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Backward pass wrapper.
    Calls get_AB_matrix_jax and backward_pass_jax to speed up computation.

    Args:
        nominal_states (ArrayImpl): (dim_x, N)
        nominal_controls (ArrayImpl): (dim_u, N)
        closest_pts (ArrayImpl): (2, N)
        slopes (ArrayImpl): (1, N)

    Returns:
        Ks (np.ndarray): gain matrices (dim_u, dim_x, N - 1)
        ks (np.ndarray): gain vectors (dim_u, N - 1)
    """
    # t0 = time.time()
    L_x, L_xx, L_u, L_uu, L_ux = self.cost.get_derivatives_jax(
        nominal_states, nominal_controls, closest_pts, slopes
    )
    fx, fu = self.dynamics.get_AB_matrix_jax(nominal_states, nominal_controls)
    Ks, ks = self.backward_pass_jax(L_x, L_xx, L_u, L_uu, L_ux, fx, fu)
    # print("backward pass use: ", time.time()-t0)
    return np.asarray(Ks), np.asarray(ks)

  def solve(
      self, cur_state: np.ndarray, controls: np.ndarray = None, obs_list: list = [],
      record: bool = False
  ) -> Tuple[np.ndarray, np.ndarray, float, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Solves the iLQR problem.

    Args:
        cur_state (np.ndarray): (dim_x,)
        controls (np.ndarray, optional): (self.dim_u,). Defaults to None.
        obs_list (list, optional): obstacle list. Defaults to [].
        record (bool, optional): Defaults to False.

    Returns:
        states: np.ndarray
        controls: np.ndarray
        t_process: float
        status: int
        theta: np.ndarray
        Ks: np.ndarray
        fx: np.ndarray
        fu: np.ndarray
    """
    status = 0
    time0 = time.time()

    # Initial forward pass.
    if controls is None:
      controls = np.zeros((self.dim_u, self.horizon))
    states = np.zeros((self.dim_x, self.horizon))
    states[:, 0] = cur_state

    for i in range(1, self.horizon):
      states[:, i], _ = self.dynamics.forward_step(states[:, i - 1], controls[:, i - 1])
    closest_pt, slope, theta = self.track.get_closest_pts(states[:2, :])

    self.cost.update_obs(obs_list)

    J = self.cost.get_cost(states, controls, closest_pt, slope, theta)

    converged = False

    # Main loop.
    for i in range(self.max_iter):

      slope = slope[np.newaxis, :]
      Ks, ks = self.backward_pass(states, controls, closest_pt, slope)

      updated = False
      for alpha in self.alphas:
        X_new, U_new, J_new, closest_pt_new, slope_new, theta_new = (
            self.forward_pass(states, controls, Ks, ks, alpha)
        )
        if J_new <= J:
          if np.abs((J-J_new) / J) < self.tol:
            converged = True
          J = J_new
          states = X_new
          controls = U_new
          closest_pt = closest_pt_new
          slope = slope_new
          theta = theta_new
          updated = True
          break
      if updated:
        self.lambad *= 0.7
      else:
        status = 2
        break
      self.lambad = max(self.lambad_min, self.lambad)

      if converged:
        status = 1
        break
    t_process = time.time() - time0

    if record:
      Ks, _ = self.backward_pass_jax(states, controls, closest_pt, slope)
      fx, fu = self.dynamics.get_AB_matrix_jax(states, controls)
    else:
      Ks = None
      fx = None
      fu = None
    return states, controls, t_process, status, theta, Ks, fx, fu

  # ----------------------------- Jitted functions -----------------------------
  @partial(jit, static_argnames='self')
  def forward_pass_jax(
      self, nominal_states: ArrayImpl, nominal_controls: ArrayImpl, Ks: ArrayImpl, ks: ArrayImpl,
      alpha: float
  ) -> Tuple[ArrayImpl, ArrayImpl]:
    """
    Jitted forward pass looped computation.

    Args:
        nominal_states (ArrayImpl): (dim_x, N)
        nominal_controls (ArrayImpl): (dim_u, N)
        Ks (ArrayImpl): gain matrices (dim_u, dim_x, N - 1)
        ks (ArrayImpl): gain vectors (dim_u, N - 1)
        alpha (float): scalar parameter

    Returns:
        Xs (ArrayImpl): (dim_x, N)
        Us (ArrayImpl): (dim_u, N)
    """

    @jit
    def forward_pass_looper(i, _carry):
      Xs, Us = _carry
      u = (
          nominal_controls[:, i] + alpha * ks[:, i]
          + Ks[:, :, i] @ (Xs[:, i] - nominal_states[:, i])
      )
      X_next, U_next = self.dynamics.integrate_forward_jax(Xs[:, i], u)
      Xs = Xs.at[:, i + 1].set(X_next)
      Us = Us.at[:, i].set(U_next)
      return Xs, Us

    Xs = jnp.zeros((self.dim_x, self.horizon))
    Us = jnp.zeros((self.dim_u, self.horizon))
    Xs = Xs.at[:, 0].set(nominal_states[:, 0])
    Xs, Us = lax.fori_loop(0, self.horizon - 1, forward_pass_looper, (Xs, Us))
    return Xs, Us

  @partial(jit, static_argnames='self')
  def backward_pass_jax(
      self, L_x: ArrayImpl, L_xx: ArrayImpl, L_u: ArrayImpl, L_uu: ArrayImpl, L_ux: ArrayImpl,
      fx: ArrayImpl, fu: ArrayImpl
  ) -> Tuple[ArrayImpl, ArrayImpl]:
    """
    Jitted backward pass looped computation.

    Args:
        L_x (ArrayImpl): (dim_x, N)
        L_xx (ArrayImpl): (dim_x, dim_x, N)
        L_u (ArrayImpl): (dim_u, N)
        L_uu (ArrayImpl): (dim_u, dim_u, N)
        L_ux (ArrayImpl): (dim_u, dim_x, N)
        fx (ArrayImpl): (dim_x, dim_x, N)
        fu (ArrayImpl): (dim_x, dim_u, N)

    Returns:
        Ks (ArrayImpl): gain matrices (dim_u, dim_x, N - 1)
        ks (ArrayImpl): gain vectors (dim_u, N - 1)
    """

    @jit
    def backward_pass_looper(i, _carry):
      V_x, V_xx, ks, Ks = _carry
      n = self.horizon - 2 - i

      Q_x = L_x[:, n] + fx[:, :, n].T @ V_x
      Q_u = L_u[:, n] + fu[:, :, n].T @ V_x
      Q_xx = L_xx[:, :, n] + fx[:, :, n].T @ V_xx @ fx[:, :, n]
      Q_ux = L_ux[:, :, n] + fu[:, :, n].T @ V_xx @ fx[:, :, n]
      Q_uu = L_uu[:, :, n] + fu[:, :, n].T @ V_xx @ fu[:, :, n]

      Q_uu_inv = jnp.linalg.inv(Q_uu + reg_mat)

      Ks = Ks.at[:, :, n].set(-Q_uu_inv @ Q_ux)
      ks = ks.at[:, n].set(-Q_uu_inv @ Q_u)

      V_x = Q_x - Ks[:, :, n].T @ Q_uu @ ks[:, n]
      V_xx = Q_xx - Ks[:, :, n].T @ Q_uu @ Ks[:, :, n]

      return V_x, V_xx, ks, Ks

    Ks = jnp.zeros((self.dim_u, self.dim_x, self.horizon - 1))
    ks = jnp.zeros((self.dim_u, self.horizon - 1))

    V_x = L_x[:, -1]
    V_xx = L_xx[:, :, -1]

    reg_mat = self.lambad * jnp.eye(self.dim_u)

    V_x, V_xx, ks, Ks = lax.fori_loop(
        0, self.horizon - 1, backward_pass_looper, (V_x, V_xx, ks, Ks)
    )
    return Ks, ks


class iLQR_racing(iLQR):

  def __init__(self, ref_path, config, safety=True):
    """
    Specialized iLQR solver that accounts for race tracks with variable width.
    """

    super(iLQR_racing, self).__init__(ref_path, config, safety)

    self.cost = Cost_racing(config, safety)

  def forward_pass(
      self, nominal_states: ArrayImpl, nominal_controls: ArrayImpl, Ks: ArrayImpl, ks: ArrayImpl,
      alpha: float
  ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Forward pass wrapper.
    Calls forward_pass_jax to speed up computation.

    Args:
        nominal_states (ArrayImpl): (dim_x, N)
        nominal_controls (ArrayImpl): (dim_u, N)
        Ks (ArrayImpl): gain matrices (dim_u, dim_x, N - 1)
        ks (ArrayImpl): gain vectors (dim_u, N - 1)
        alpha (float): scalar parameter

    Returns:
        X (np.ndarray): (dim_x, N)
        U (np.ndarray): (dim_u, N)
        J (float): total cost.
        closest_pts (np.ndarray): 6xN array of each state's closest point [x,y] on the track.
        slope (np.ndarray): 1xN array of track's slopes (rad) at closest points.
        theta (np.ndarray): 1xN array of the progress at each state.
    """
    Xs, Us = self.forward_pass_jax(nominal_states, nominal_controls, Ks, ks, alpha)
    Xs = np.asarray(Xs)
    Us = np.asarray(Us)

    closest_pts_ref, slope, theta = self.track.get_closest_pts(Xs[:2, :], mode='ref')
    closest_pts_inner, _, _ = self.track.get_closest_pts(Xs[:2, :], mode='inner')
    closest_pts_outer, _, _ = self.track.get_closest_pts(Xs[:2, :], mode='outer')
    closest_pts_ref_vel, _, _ = self.track.get_closest_pts(Xs[:3, :], mode='ref_vel')

    closest_pts = np.vstack(
        (closest_pts_ref, closest_pts_inner, closest_pts_outer, closest_pts_ref_vel[2, :])
    )

    J = self.cost.get_cost(Xs, Us, closest_pts, slope, theta)

    return Xs, Us, J, closest_pts, slope, theta

  def solve(
      self, cur_state: np.ndarray, controls: np.ndarray = None, obs_list: list = [],
      record: bool = False
  ) -> Tuple[np.ndarray, np.ndarray, float, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Solves the iLQR problem.

    Args:
        cur_state (np.ndarray): (dim_x,)
        controls (np.ndarray, optional): (self.dim_u,). Defaults to None.
        obs_list (list, optional): obstacle list. Defaults to [].
        record (bool, optional): Defaults to False.

    Returns:
        states: np.ndarray
        controls: np.ndarray
        t_process: float
        status: int
        theta: np.ndarray
        Ks: np.ndarray
        fx: np.ndarray
        fu: np.ndarray
    """
    status = 0
    time0 = time.time()

    if controls is None:
      controls = np.zeros((self.dim_u, self.horizon))
    states = np.zeros((self.dim_x, self.horizon))
    states[:, 0] = cur_state

    for i in range(1, self.horizon):
      states[:, i], _ = self.dynamics.forward_step(states[:, i - 1], controls[:, i - 1])

    closest_pts_ref, slope, theta = self.track.get_closest_pts(states[:2, :], mode='ref')
    closest_pts_inner, _, _ = self.track.get_closest_pts(states[:2, :], mode='inner')
    closest_pts_outer, _, _ = self.track.get_closest_pts(states[:2, :], mode='outer')
    closest_pts_ref_vel, _, _ = self.track.get_closest_pts(states[:3, :], mode='ref_vel')

    closest_pts = np.vstack(
        (closest_pts_ref, closest_pts_inner, closest_pts_outer, closest_pts_ref_vel[2, :])
    )

    self.cost.update_obs(obs_list)

    J = self.cost.get_cost(states, controls, closest_pts, slope, theta)

    converged = False

    for i in range(self.max_iter):
      slope = slope[np.newaxis, :]
      Ks, ks = self.backward_pass(states, controls, closest_pts, slope)

      updated = False
      for alpha in self.alphas:
        X_new, U_new, J_new, closest_pts_new, slope_new, theta_new = (
            self.forward_pass(states, controls, Ks, ks, alpha)
        )
        if J_new <= J:
          if np.abs((J-J_new) / J) < self.tol:
            converged = True
          J = J_new
          states = X_new
          controls = U_new
          closest_pts = closest_pts_new
          slope = slope_new
          theta = theta_new
          updated = True
          break
      if updated:
        self.lambad *= 0.7
      else:
        status = 2
        break
      self.lambad = max(self.lambad_min, self.lambad)

      if converged:
        status = 1
        break
    t_process = time.time() - time0

    if record:
      Ks, _ = self.backward_pass_jax(states, controls, closest_pts, slope)
      fx, fu = self.dynamics.get_AB_matrix_jax(states, controls)
    else:
      Ks = None
      fx = None
      fu = None
    return states, controls, t_process, status, theta, Ks, fx, fu
