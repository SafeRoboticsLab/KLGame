"""
Jaxified differentiable ILQGame and KLGame solvers.

Reference: https://github.com/HJReachability/ilqgames

Please contact the author(s) of this library if you have any questions.
Author: Haimin Hu (haiminh@princeton.edu)
"""

import time
import numpy as np
from typing import List, Tuple
from functools import partial

from flax import linen as nn
from jax import jit, lax, vmap
from jaxlib.xla_extension import ArrayImpl
import jax.numpy as jnp
from flax.core import FrozenDict

from iLQR import Track, Struct
from .cost import MultiPlayerRacingCost
from .utils import StaticCostParams, get_perfect_obs_two_player, is_safe
from .multiplayer_dynamical_system import MultiPlayerDynamicalSystem


class ILQSolver(object):

  def __init__(
      self, track: Track, config: Struct, dynamics: MultiPlayerDynamicalSystem, verbose: str = False
  ):
    """
    Initializer.
    """

    self.track = track
    self.config = config
    self.horizon = config.N
    self.max_iter = config.MAX_ITER
    self.dynamics = dynamics
    self.num_players = dynamics._num_players
    self.line_search_scaling = 1.1**(-np.arange(config.LS_NUM)**2)

    self.dim_x = self.dynamics._x_dim
    self.dim_u_ss = self.dynamics._subsystem.dim_u

    # Create costs for each player.
    self.costs = []
    for ii in range(self.num_players):
      if ii == 0:
        is_prox_cost = True
        is_ego = True
      else:
        is_prox_cost = False
        is_ego = False

      self.costs.append(
          MultiPlayerRacingCost(
              config, self.dim_x, self.dynamics._subsystems[ii], self.dynamics._LMx,
              is_prox_cost=is_prox_cost, is_ego=is_ego
          )
      )

    self.lambad = 10.  # NOTE: for ilqr only
    self.lambad_min = 1e-3

    self.verbose = verbose
    self.reset()

  def solve(self, cur_state: np.ndarray, us_warmstart: np.ndarray):
    """
    Runs the iLQGame algorithm.

    Args:
        cur_state (np.ndarray): (nx,) Current state.
        us_warmstart (np.ndarray, optional): (nui, N, num_players) Warmstart controls.

    Returns:
        states: np.ndarray
        controls: np.ndarray
        t_process: float
        status: int
        theta: np.ndarray
    """
    status = 0
    time0 = time.time()

    # Initial forward pass.
    x0, us = jnp.asarray(cur_state), jnp.asarray(us_warmstart)
    xs, cost_init, closest_pts, slopes, thetas = self.initial_forward_pass(x0, us)
    cost_best = cost_init

    # Resets the solver.
    self.reset(_current_x=xs, _current_u=us, _current_J=cost_init)

    # Main loop.
    for iter in range(self.max_iter):

      t_start = time.time()

      As, Bs = self.linearize_dynamics(xs, us)

      lxs, lus, Hxxs, Huus = self.quadraticize_costs(xs, us, closest_pts, slopes, thetas)

      Ps, alphas_bpass, _, _ = self.backward_pass(As, Bs, lxs, lus, Hxxs, Huus)

      for line_search_scaling in self.line_search_scaling:
        alphas_ls = alphas_bpass * line_search_scaling
        xs_ls, us_ls, cost_ls, closest_pts_ls, slopes_ls, thetas_ls = self.compute_operating_point(
            xs, us, Ps, alphas_ls, cur_state
        )
        if cost_ls < cost_best:
          xs = xs_ls
          us = us_ls
          closest_pts = closest_pts_ls
          slopes = slopes_ls
          thetas = thetas_ls
          cost_best = cost_ls

          # iLQR only
          self.lambad *= 0.7
          self.lambad = max(self.lambad_min, self.lambad)
          break

      # region: Updates operating points.
      self.last_operating_point = self.current_operating_point
      self.current_operating_point = (xs, us)

      self.last_social_cost = self.current_social_cost
      self.current_social_cost = cost_best

      if self.current_social_cost < self.best_social_cost:
        self.best_operating_point = self.current_operating_point
        self.best_social_cost = self.current_social_cost
      # endregion

      # region: Checks convergence.
      if self.is_converged_cost():
        status = 1
        if self.verbose:
          print(
              "[iLQGame] Social cost (", round(self.current_social_cost, 2), ") has converged! \n"
          )
        break
      # endregion

      t_iter = time.time() - t_start
      if self.verbose:
        print(
            "[iLQGame] Iteration", iter, "| Social cost: ", round(self.current_social_cost, 2),
            " | Iter. time: ", t_iter
        )

    t_process = time.time() - time0

    return np.asarray(xs), np.asarray(us), t_process, status, np.asarray(thetas)

  def solve_parametric(self, cost_param, cur_state: np.ndarray, us_warmstart: np.ndarray):
    """
    Runs the iLQGame algorithm. Takes in the cost param.

    Args:
        cost_param: Static parameters or parameters of the cost model (e.g. NN)
        cur_state (np.ndarray): (nx,) Current state.
        us_warmstart (np.ndarray): (nui, N, num_players) Warmstart controls.

    Returns:
        states: ArrayImpl
        controls: ArrayImpl
        closest_pts: ArrayImpl
        slopes: ArrayImpl
        thetas: ArrayImpl
        t_process: float
        status: int
    """
    # status = 0
    time0 = time.time()

    # Initial forward pass.
    x0, us = jnp.asarray(cur_state), jnp.asarray(us_warmstart)
    xs, cost_init, closest_pts, slopes, thetas = self.initial_forward_pass(x0, us, cost_param)
    cost_best = cost_init

    # Resets the solver.
    self.reset(_current_x=xs, _current_u=us, _current_J=cost_init)

    # Main loop.
    for iter in range(self.max_iter):

      t_start = time.time()

      As, Bs = self.linearize_dynamics(xs, us)

      lxs, lus, Hxxs, Huus = self.quadraticize_costs(
          xs, us, closest_pts, slopes, thetas, cost_param
      )

      Ps, alphas_bpass, Qs_inv, _ = self.backward_pass(As, Bs, lxs, lus, Hxxs, Huus)

      for line_search_scaling in self.line_search_scaling:
        alphas_ls = alphas_bpass * line_search_scaling
        xs_ls, us_ls, cost_ls, closest_pts_ls, slopes_ls, thetas_ls = self.compute_operating_point(
            xs, us, Ps, alphas_ls, cur_state, cost_param
        )
        if cost_ls < cost_best:
          xs = xs_ls
          us = us_ls
          closest_pts = closest_pts_ls
          slopes = slopes_ls
          thetas = thetas_ls
          cost_best = cost_ls

          # iLQR only
          self.lambad *= 0.7
          self.lambad = max(self.lambad_min, self.lambad)
          break

      # region: Updates operating points.
      self.last_operating_point = self.current_operating_point
      self.current_operating_point = (xs, us)

      self.last_social_cost = self.current_social_cost
      self.current_social_cost = cost_best

      if self.current_social_cost < self.best_social_cost:
        self.best_operating_point = self.current_operating_point
        self.best_social_cost = self.current_social_cost
      # endregion

      # region: Checks convergence.
      if self.is_converged_cost():
        if self.verbose:
          print(
              "[Parametric iLQGame] Social cost (", round(self.current_social_cost, 2),
              ") has converged! \n"
          )
        break
      # endregion

      t_iter = time.time() - t_start
      if self.verbose:
        print(
            "[Parametric iLQGame] Iteration", iter, "| Social cost: ",
            round(self.current_social_cost, 2), " | Iter. time: ", t_iter
        )

    t_process = time.time() - time0

    return xs, us, closest_pts, slopes, thetas, t_process, Qs_inv

  def set_neutral_weights(self):
    self.config.W_OVERTAKE = 300.
    self.config.W_STAYBACK = 0.
    self.config.W_INNER_EGO = 0.
    self.config.W_OUTER_EGO = 0.
    self.config.W_BLOCK = -20.
    self.config.W_INNER_ADO = 0.
    self.config.W_OUTER_ADO = 0.

  @partial(jit, static_argnames='self')
  def _solve_lq_game_static_param(
      self, cost_param, xs_nominal: ArrayImpl, us_nominal: ArrayImpl, closest_pts: ArrayImpl,
      slopes: ArrayImpl, thetas: ArrayImpl
  ):
    """
    Internal method for solving one-step LQ game for the inverse game. Should not be used elsewhere.
    Assumes static cost parameters or general MLP.

    NOTE: For Jax compatibility, we do not recompute closest_pts, slopes and thetas in the fwd pass.

    Args:
        cost_param: Static parameters or parameters of the cost model (e.g. NN)
        xs_nominal (ArrayImpl): (nx, N) Equilibrated state trajectory.
        us_nominal (ArrayImpl): (nui, N, num_players) Equilibrated control trajectory.
        closest_pts (ArrayImpl): (7, N, num_players) Equilibrated closest points.
        slopes (ArrayImpl): (1, N, num_players) Equilibrated slopes.
        thetas (ArrayImpl): (1, N, num_players) Equilibrated thetas.

    Returns:
        xs (ArrayImpl): (nx, N) Solution state trajectory.
        us (ArrayImpl): (nui, N, num_players) Solution control trajectory.
    """
    As, Bs = self.linearize_dynamics(xs_nominal, us_nominal)
    lxs, lus, Hxxs, Huus = self.quadraticize_costs(
        xs_nominal, us_nominal, closest_pts, slopes, thetas, cost_param
    )
    Ps, alphas, _, _ = self.backward_pass(As, Bs, lxs, lus, Hxxs, Huus)  # Backward pass.
    xs, us = self._compute_operating_point_jax(
        xs_nominal, us_nominal, Ps, alphas, xs_nominal[:, 0]
    )  # Forward pass.

    return xs, us

  def is_converged_cost(self):
    """
    Checks convergence based on social cost difference.
    """
    TOLERANCE_RATE = self.config.TOLERANCE_RATE
    # COST_LB = 1e6

    if self.last_social_cost is None:
      return False

    cost_diff_rate = np.abs(
        (self.current_social_cost - self.last_social_cost) / self.last_social_cost
    )

    if cost_diff_rate > TOLERANCE_RATE:  #or self.current_social_cost > COST_LB:
      return False
    else:
      return True

  def reset(self, _current_x=None, _current_u=None, _current_J=None):
    """
    Resets the solver and warmstarts it if possible.
    """

    if _current_x is None:
      _current_x = jnp.zeros((self.dim_x, self.horizon))

    if _current_u is None:
      _current_u = jnp.zeros((self.dim_u_ss, self.horizon, self.num_players))

    self.last_operating_point = None
    self.current_operating_point = (_current_x, _current_u)
    self.best_operating_point = (_current_x, _current_u)

    self.last_social_cost = np.Inf
    self.current_social_cost = _current_J
    self.best_social_cost = _current_J

  def initial_forward_pass(self, cur_state: ArrayImpl, controls: ArrayImpl, cost_param=None):
    """
    Performs the initial forward pass given warmstart controls.

    Args:
        cur_state (ArrayImpl): current state.
        controls (ArrayImpl): warmstart controls.
        cost_param: Static parameters or parameters of the cost model (e.g. NN)

    Returns:
        states (ArrayImpl): states (nx, N)
        social_cost (float): sum of all players costs
        closest_pts (ArrayImpl): (7, N, num_players)
        slopes (ArrayImpl): (1, N, num_players)
    """
    # Forward sim.
    states, closest_pts, slopes, thetas = self._initial_forward_pass_sim(cur_state, controls)

    # Evaluates costs.
    cost_sum = self._evaluate_costs(states, controls, closest_pts, slopes, thetas, cost_param)

    return states, cost_sum, closest_pts, slopes, thetas

  def _initial_forward_pass_sim(self, cur_state: ArrayImpl, controls: ArrayImpl):
    states = jnp.zeros((self.dim_x, self.horizon))
    states = states.at[:, 0].set(cur_state)

    # Forward pass.
    for k in range(1, self.horizon):
      states_next, _ = self.dynamics.integrate_forward(states[:, k - 1], controls[:, k - 1, :])
      states = states.at[:, k].set(states_next)

    # Computes track-related information.
    closest_pts = np.zeros((7, self.horizon, self.num_players))
    slopes = np.zeros((1, self.horizon, self.num_players))
    thetas = np.zeros((1, self.horizon, self.num_players))

    for ii in range(self.num_players):
      states_ii = np.asarray(self.dynamics._LMx[:, :, ii] @ states)
      closest_pts_ref, slope, theta = self.track.get_closest_pts(states_ii[:2, :], mode='ref')
      closest_pts_inner, _, _ = self.track.get_closest_pts(states_ii[:2, :], mode='inner')
      closest_pts_outer, _, _ = self.track.get_closest_pts(states_ii[:2, :], mode='outer')
      closest_pts_ref_vel, _, _ = self.track.get_closest_pts(states_ii[:3, :], mode='ref_vel')
      closest_pts_ii = np.vstack(
          (closest_pts_ref, closest_pts_inner, closest_pts_outer, closest_pts_ref_vel[2, :])
      )
      closest_pts[:, :, ii] = closest_pts_ii
      slopes[:, :, ii] = slope[np.newaxis, :]
      thetas[:, :, ii] = theta

    closest_pts = jnp.asarray(closest_pts)
    slopes = jnp.asarray(slopes)
    thetas = jnp.asarray(thetas)

    return states, closest_pts, slopes, thetas

  def compute_operating_point(
      self, current_xs: ArrayImpl, current_us: ArrayImpl, Ps: ArrayImpl, alphas: ArrayImpl,
      cur_state: ArrayImpl, cost_param=None
  ) -> Tuple[ArrayImpl, ArrayImpl, float, ArrayImpl, ArrayImpl, ArrayImpl]:
    """
    Computes current operating point by propagating through dynamics.
    This function is a wrapper of _compute_operating_point_jax()

    Args:
        current_xs (ArrayImpl): (nx, N) current state traj, used as nominal
        current_us (ArrayImpl): (nui, N, num_players) current player controls, used as nominal
        Ps (ArrayImpl): (nui, nx, N, num_players)
        alphas (ArrayImpl): (nui, N, num_players)
        cur_state (ArrayImpl): (nx,) current (initial) state
        cost_param: Static parameters or parameters of the cost model (e.g. NN)

    Returns:
        xs (ArrayImpl): updated states (nx, N)
        us (ArrayImpl): updated player controls (nui, N, num_players)
        social_cost (float): sum of all players costs
        closest_pts (ArrayImpl): (7, N, num_players)
        slopes (ArrayImpl): (1, N, num_players)
        thetas (ArrayImpl): (1, N, num_players)
    """
    # Computes track info.
    xs, us, closest_pts, slopes, thetas = self._compute_state_control(
        current_xs, current_us, Ps, alphas, cur_state
    )

    # Evaluates costs.
    cost_sum = self._evaluate_costs(xs, us, closest_pts, slopes, thetas, cost_param)

    return xs, us, cost_sum, closest_pts, slopes, thetas

  def _compute_state_control(self, current_xs, current_us, Ps, alphas, cur_state):
    xs, us = self._compute_operating_point_jax(current_xs, current_us, Ps, alphas, cur_state)
    closest_pts, slopes, thetas = self._compute_track_info(xs, self.horizon)
    return xs, us, jnp.asarray(closest_pts), jnp.asarray(slopes), jnp.asarray(thetas)

  def _compute_track_info(self, xs, horizon):
    # Computes track-related information.
    closest_pts = np.zeros((7, horizon, self.num_players))
    slopes = np.zeros((1, horizon, self.num_players))
    thetas = np.zeros((1, horizon, self.num_players))

    for ii in range(self.num_players):
      xs_ii = np.asarray(self.dynamics._LMx[:, :, ii] @ xs)
      closest_pts_ref, slope, theta = self.track.get_closest_pts(xs_ii[:2, :], mode='ref')
      closest_pts_inner, _, _ = self.track.get_closest_pts(xs_ii[:2, :], mode='inner')
      closest_pts_outer, _, _ = self.track.get_closest_pts(xs_ii[:2, :], mode='outer')
      closest_pts_ref_vel, _, _ = self.track.get_closest_pts(xs_ii[:3, :], mode='ref_vel')
      if horizon > 1:
        closest_pts_ii = np.vstack(
            (closest_pts_ref, closest_pts_inner, closest_pts_outer, closest_pts_ref_vel[2, :])
        )
      else:
        closest_pts_ii = np.concatenate(
            (closest_pts_ref, closest_pts_inner, closest_pts_outer, closest_pts_ref_vel[2:3])
        )[:, np.newaxis]
      closest_pts[:, :, ii] = closest_pts_ii
      slopes[:, :, ii] = slope[np.newaxis, :]
      thetas[:, :, ii] = theta
    return closest_pts, slopes, thetas

  @partial(jit, static_argnames='self')
  def _evaluate_costs(self, xs, us, closest_pts, slopes, thetas, cost_param=None):
    cost_weights = self._get_cost_param(cost_param, xs, us, closest_pts, slopes, thetas)
    costs = self._compute_cost_items(xs, us, closest_pts, slopes, thetas, cost_weights)
    return jnp.sum(costs)

  @partial(jit, static_argnames='self')
  def _evaluate_costs_nod(self, xs, us, closest_pts, slopes, thetas, z0, cost_param):
    zs, cost_weights = self._get_cost_param_nod(cost_param, xs, us, closest_pts, slopes, thetas, z0)
    costs = self._compute_cost_items(xs, us, closest_pts, slopes, thetas, cost_weights)
    return zs, jnp.sum(costs)

  @partial(jit, static_argnames='self')
  def _compute_cost_items(self, xs, us, closest_pts, slopes, thetas, cost_weights):
    costs = jnp.zeros((self.num_players))
    for ii in range(self.num_players):
      costs = costs.at[ii].set(
          self.costs[ii].get_cost(
              xs, us[:, :, ii], closest_pts[:, :, ii], slopes, thetas, cost_weights
          )
      )
    return costs

  @partial(jit, static_argnames='self')
  def _compute_performance_cost_items(self, xs, us, closest_pts, slopes, thetas):
    costs = jnp.zeros((self.num_players))
    for ii in range(self.num_players):
      costs = costs.at[ii].set(
          self.costs[ii].get_cost_performance(
              xs, us[:, :, ii], closest_pts[:, :, ii], slopes, thetas
          )
      )
    return costs

  @partial(jit, static_argnames='self')
  def _compute_operating_point_jax(
      self, nominal_states: ArrayImpl, nominal_controls: ArrayImpl, Ps: ArrayImpl,
      alphas: ArrayImpl, cur_state: ArrayImpl
  ) -> Tuple[ArrayImpl, ArrayImpl]:
    """
    Computes current operating point by propagating through dynamics.

    Args:
        nominal_states (ArrayImpl): (nx, N)
        nominal_controls (ArrayImpl): (nui, N, num_players)
        Ps (ArrayImpl): (nui, nx, N, num_players)
        alphas (ArrayImpl): (nui, N, num_players)
        cur_state (ArrayImpl): (nx,) current init. state

    Returns:
        xs (ArrayImpl): updated states (nx, N)
        us (ArrayImpl): updated player controls (nui, N, num_players)
    """

    def forward_pass_looper(k, _carry):

      def compute_agent_control(x, x_ref, uii_ref, Pii, alphaii):
        return uii_ref - Pii @ (x-x_ref) - alphaii

      compute_all_agents_controls = vmap(
          compute_agent_control, in_axes=(None, None, 1, 2, 1), out_axes=(1)
      )

      xs, us = _carry
      us_tmp = compute_all_agents_controls(
          xs[:, k], nominal_states[:, k], nominal_controls[:, k, :], Ps[:, :, k, :], alphas[:, k, :]
      )
      X_next, U_next = self.dynamics.integrate_forward(xs[:, k], us_tmp)
      xs = xs.at[:, k + 1].set(X_next)
      us = us.at[:, k, :].set(U_next)
      return xs, us

    xs = jnp.zeros_like(nominal_states)
    us = jnp.zeros_like(nominal_controls)
    xs = xs.at[:, 0].set(cur_state)
    xs, us = lax.fori_loop(0, self.horizon - 1, forward_pass_looper, (xs, us))
    return xs, us

  @partial(jit, static_argnames='self')
  def linearize_dynamics(self, xs: ArrayImpl, us: ArrayImpl) -> Tuple[ArrayImpl, ArrayImpl]:
    """
    Linearizes dynamics at the current operating point.

    Args:
        xs (ArrayImpl): (nx, N) nominal state traj
        us (ArrayImpl): (nui, N, num_players) nominal player controls

    Returns:
        As (ArrayImpl): (nx, nx, N) A matrices
        Bs (ArrayImpl): (nx, nui, N, num_players) B matrices
    """

    def linearize_single_time(x, u):
      A, B = self.dynamics.linearize_discrete_jitted(x, u)
      return A, B

    linearize_along_horizon = vmap(linearize_single_time, in_axes=(1, 1), out_axes=(2, 2))
    As, Bs = linearize_along_horizon(xs, us)

    return As, Bs

  @partial(jit, static_argnames='self')
  def quadraticize_costs(
      self, xs: ArrayImpl, us: ArrayImpl, closest_pts: ArrayImpl, slopes: ArrayImpl,
      thetas: ArrayImpl, cost_param=None
  ) -> Tuple[ArrayImpl, ArrayImpl, ArrayImpl, ArrayImpl]:
    """
    Quadraticizes costs of all players at the current operating point.

    Args:
        xs (ArrayImpl): (nx, N) nominal state trajectory
        us (ArrayImpl): (nui, N, num_players) nominal player controls
        closest_pts (ArrayImpl): (7, N, num_players)
        slopes (ArrayImpl): (1, N, num_players)
        thetas (ArrayImpl): (1, N, num_players)
        cost_param: Static parameters or parameters of the cost model (e.g. NN)

    Returns:
        lxs (ArrayImpl): (nx, N, num_players) gradients lx = dc/dx of all players
        lus (ArrayImpl): (nui, N, num_players) gradients lu = dc/du of all players
        Hxxs (ArrayImpl): (nx, nx, N, num_players) Hessians Hxx of all players
        Huus (ArrayImpl): (nui, nui, N, num_players) Hessians Huu of all players
    """
    cost_weights = self._get_cost_param(cost_param, xs, us, closest_pts, slopes, thetas)
    lxs, lus, Hxxs, Huus = self._compute_cost_deriv(
        xs, us, closest_pts, slopes, thetas, cost_weights
    )
    return lxs, lus, Hxxs, Huus

  @partial(jit, static_argnames='self')
  def quadraticize_costs_nod(
      self, xs: ArrayImpl, us: ArrayImpl, closest_pts: ArrayImpl, slopes: ArrayImpl,
      thetas: ArrayImpl, cur_opn: ArrayImpl, cost_param: FrozenDict
  ) -> Tuple[ArrayImpl, ArrayImpl, ArrayImpl, ArrayImpl]:
    """
    Quadraticizes costs of all players at the current operating point. NOD case only.

    Args:
        xs (ArrayImpl): (nx, N) nominal state trajectory
        us (ArrayImpl): (nui, N, num_players) nominal player controls
        closest_pts (ArrayImpl): (7, N, num_players)
        slopes (ArrayImpl): (1, N, num_players)
        thetas (ArrayImpl): (1, N, num_players)
        cur_opn (ArrayImpl): current opinion state.
        cost_param: Static parameters or parameters of the cost model (e.g. NN)

    Returns:
        lxs (ArrayImpl): (nx, N, num_players) gradients lx = dc/dx of all players
        lus (ArrayImpl): (nui, N, num_players) gradients lu = dc/du of all players
        Hxxs (ArrayImpl): (nx, nx, N, num_players) Hessians Hxx of all players
        Huus (ArrayImpl): (nui, nui, N, num_players) Hessians Huu of all players
    """
    _, cost_weights = self._get_cost_param_nod(
        cost_param, xs, us, closest_pts, slopes, thetas, cur_opn
    )
    lxs, lus, Hxxs, Huus = self._compute_cost_deriv(
        xs, us, closest_pts, slopes, thetas, cost_weights
    )
    return lxs, lus, Hxxs, Huus

  @partial(jit, static_argnames='self')
  def _compute_cost_deriv(self, xs, us, closest_pts, slopes, thetas, cost_weights):
    lxs = jnp.zeros((self.dim_x, self.horizon, self.num_players))
    lus = jnp.zeros((self.dim_u_ss, self.horizon, self.num_players))
    Hxxs = jnp.zeros((self.dim_x, self.dim_x, self.horizon, self.num_players))
    Huus = jnp.zeros((self.dim_u_ss, self.dim_u_ss, self.horizon, self.num_players))

    for ii in range(self.num_players):
      lxs_ii, lus_ii, Hxxs_ii, Huus_ii = self.costs[ii].get_cost_derivatives(
          xs, us[:, :, ii], closest_pts[:, :, ii], slopes, thetas, cost_weights
      )
      lxs = lxs.at[:, :, ii].set(lxs_ii)
      lus = lus.at[:, :, ii].set(lus_ii)
      Hxxs = Hxxs.at[:, :, :, ii].set(Hxxs_ii)
      Huus = Huus.at[:, :, :, ii].set(Huus_ii)

    return lxs, lus, Hxxs, Huus

  @partial(jit, static_argnames='self')
  def backward_pass(
      self, As: ArrayImpl, Bs: ArrayImpl, lxs: ArrayImpl, lus: ArrayImpl, Hxxs: ArrayImpl,
      Huus: ArrayImpl
  ) -> Tuple[ArrayImpl, ArrayImpl, ArrayImpl, ArrayImpl]:
    """
    Solves a time-varying, finite horizon LQ game (finds closed-loop Nash
    feedback strategies for both players).
    Assumes that dynamics are given by
            ``` dx_{k+1} = A_k dx_k + \sum_i Bs[i]_k du[i]_k ```

    Derivation can be found in:
        https://github.com/HJReachability/ilqgames/blob/master/derivations/feedback_lq_nash.pdf

    Args:
        As (ArrayImpl): (nx, nx, N) A matrices
        Bs (ArrayImpl): (nui, nui, N, num_players) B matrices
        lxs (ArrayImpl): (nx, N, num_players) gradients lx = dc/dx of all players
        lus (ArrayImpl): (nui, N, num_players) gradients lu = dc/du of all players
        Hxxs (ArrayImpl): (nx, nx, N, num_players) Hessians Hxx of all players
        Huus (ArrayImpl): (nui, nui, N, num_players) Hessians Huu of all players

    Returns:
        ArrayImpl: Ps (dim_u_ss, dim_x, N-1, num_players)
        ArrayImpl: alphas (dim_u_ss, N-1, num_players)
    """

    @jit
    def backward_pass_looper(k, _carry):
      Ps, alphas, Z, zeta, Qs_inv, qs = _carry
      n = horizon - 1 - k

      # Computes Ps given previously computed Z.
      S = jnp.array(()).reshape(0, sum(self.dynamics._u_dims))
      Y1 = jnp.array(()).reshape(0, dim_x)
      for ii in range(num_players):
        Sii = jnp.array(()).reshape(dim_u_ss, 0)
        for jj in range(num_players):
          if jj == ii:
            Sii = jnp.hstack(
                (Sii, Bs[:, :, n, ii].T @ Z[:, :, ii] @ Bs[:, :, n, jj] + Huus[:, :, n, ii])
            )
          else:
            Sii = jnp.hstack((Sii, Bs[:, :, n, ii].T @ Z[:, :, ii] @ Bs[:, :, n, jj]))
        S = jnp.vstack((S, Sii))

        Y1ii = Bs[:, :, n, ii].T @ Z[:, :, ii] @ As[:, :, n]
        Y1 = jnp.vstack((Y1, Y1ii))

      P, _, _, _ = jnp.linalg.lstsq(a=S, b=Y1, rcond=None)
      # Sinv = jnp.linalg.pinv(S)
      # P = Sinv @ Y1

      for ii in range(num_players):
        Pii = self.dynamics._LMu[:, :, ii] @ P
        Ps = Ps.at[:, :, n, ii].set(Pii)

      # Computes F_k = A_k - B1_k P1_k - B2_k P2_k -...
      F = As[:, :, n]
      for ii in range(num_players):
        F -= Bs[:, :, n, ii] @ Ps[:, :, n, ii]

      # Computes alphas using previously computed zetas.
      Y2 = jnp.array(()).reshape(0, 1)
      for ii in range(num_players):
        # Y2ii = (Bs[:, :, n, ii].T @ zeta[:, ii]).reshape((dim_u_ss, 1))
        Y2ii = (Bs[:, :, n, ii].T @ zeta[:, ii] + lus[:, n, ii]).reshape((dim_u_ss, 1))
        Y2 = jnp.vstack((Y2, Y2ii))

      alpha, _, _, _ = jnp.linalg.lstsq(a=S, b=Y2, rcond=None)
      # alpha = Sinv @ Y2

      for ii in range(num_players):
        alphaii = self.dynamics._LMu[:, :, ii] @ alpha
        alphas = alphas.at[:, n, ii].set(alphaii[:, 0])

      # Computes beta_k = -B1_k alpha1 - B2_k alpha2_k -...
      beta = 0.
      for ii in range(num_players):
        beta -= Bs[:, :, n, ii] @ alphas[:, n, ii]

      # Updates zeta.
      for ii in range(num_players):
        _FZb = F.T @ (zeta[:, ii] + Z[:, :, ii] @ beta)
        _PRa = Ps[:, :, n, ii].T @ Huus[:, :, n, ii] @ alphas[:, n, ii]
        zeta = zeta.at[:, ii].set(_FZb + _PRa + lxs[:, n, ii])

      # Updates Z.
      for ii in range(num_players):
        _FZF = F.T @ Z[:, :, ii] @ F
        _PRP = Ps[:, :, n, ii].T @ Huus[:, :, n, ii] @ Ps[:, :, n, ii]
        Z = Z.at[:, :, ii].set(_FZF + _PRP + Hxxs[:, :, n, ii])

      # Updates Q-value function.
      for ii in range(num_players):
        Q_tmp = Bs[:, :, n, ii].T @ Z[:, :, ii] @ Bs[:, :, n, ii] + Huus[:, :, n, ii]
        Qs_inv = Qs_inv.at[:, :, n, ii].set(jnp.linalg.inv(Q_tmp))
        qs = qs.at[:, n, ii].set(Bs[:, :, n, ii].T @ zeta[:, ii] + lus[:, n, ii])

      return Ps, alphas, Z, zeta, Qs_inv, qs

    # Unpacks horizon and number of players.
    horizon = self.horizon
    num_players = self.num_players

    # Caches dimensions of state and controls for each player.
    dim_x = self.dim_x
    dim_u_ss = self.dim_u_ss

    # Recursively computes all intermediate and final variables.
    Z = Hxxs[:, :, -1, :]
    zeta = lxs[:, -1, :]

    # Initializes strategy matrices.
    Ps = jnp.zeros((dim_u_ss, dim_x, horizon, num_players))
    alphas = jnp.zeros((dim_u_ss, horizon, num_players))

    # Initializes value function matrices.
    Qs_inv = jnp.zeros((dim_u_ss, dim_u_ss, horizon, num_players))
    qs = jnp.zeros((dim_u_ss, horizon, num_players))
    for ii in range(num_players):
      Q_tmp = Bs[:, :, -1, ii].T @ Hxxs[:, :, -1, ii] @ Bs[:, :, -1, ii] + Huus[:, :, -1, ii]
      Qs_inv = Qs_inv.at[:, :, -1, ii].set(jnp.linalg.inv(Q_tmp))
      qs = qs.at[:, -1, ii].set(Bs[:, :, -1, ii].T @ lxs[:, -1, ii] + lus[:, -1, ii])

    # Backward pass.
    Ps, alphas, Z, zeta, Qs_inv, qs = lax.fori_loop(
        0, self.horizon, backward_pass_looper, (Ps, alphas, Z, zeta, Qs_inv, qs)
    )

    return Ps, alphas, Qs_inv, qs

  def normalize_nn_input(self, data):
    """
    Normalizes the dataset (N_features, N_data_pts).
    """
    mean = jnp.mean(data, axis=1)
    var = jnp.var(data, axis=1)
    min_data = jnp.min(data, axis=1)
    max_data = jnp.max(data, axis=1)

    self._data_mean = mean[:, jnp.newaxis]
    self._data_var = var[:, jnp.newaxis]
    self._data_min = min_data[:, jnp.newaxis]
    self._data_max = max_data[:, jnp.newaxis]

  @partial(jit, static_argnames='self')
  def _get_nn_inputs(self, xs, us, closest_pts, slopes, thetas):
    """
    Gets NN input data.
    """
    # states and controls
    pos_rel = xs[0:2, :] - xs[4:6, :]
    psi_rel = xs[3:4, :] - xs[7:8, :]
    prog_rel = thetas[:, :, 0] - thetas[:, :, 1]
    v_ego = xs[2:3, :]
    v_ado = xs[6:7, :]
    u_ego = us[:, :, 0]
    u_ado = us[:, :, 1]

    # track-related
    closest_pt_ego = closest_pts[:, :, 0]
    closest_pt_ado = closest_pts[:, :, 1]
    slope_ego = slopes[:, :, 0]
    slope_ado = slopes[:, :, 1]

    nn_inputs = jnp.concatenate((
        pos_rel, psi_rel, prog_rel, v_ego, v_ado, u_ego, u_ado, closest_pt_ego, slope_ego,
        closest_pt_ado, slope_ado
    ), axis=0)

    if self.config.NORMALIZE_NN_INPUTS:
      nn_inputs = (nn_inputs - self._data_min) / (self._data_max - self._data_min)
      nn_inputs = jnp.nan_to_num(nn_inputs)

    return nn_inputs

  @partial(jit, static_argnames='self')
  def _get_cost_param(self, cost_param, xs, us, closest_pts, slopes, thetas) -> ArrayImpl:
    """
    Manages the cost parameters.

    Args:
        cost_param: Static parameters or parameters of the cost model (e.g. NN)
        xs (ArrayImpl): (nx, N)
        us (ArrayImpl): (nui, N, num_players)
        closest_pts (ArrayImpl): (7, N, num_players)
        slopes (ArrayImpl): (1, N, num_players)
        thetas (ArrayImpl): (1, N, num_players)

    Returns:
        ArrayImpl (N_param, N)
    """
    if cost_param is None:
      W_overtake = self.config.W_OVERTAKE * jnp.ones((self.horizon,))
      W_stayback = self.config.W_STAYBACK * jnp.ones((self.horizon,))
      W_inner_ego = self.config.W_INNER_EGO * jnp.ones((self.horizon,))
      W_outer_ego = self.config.W_OUTER_EGO * jnp.ones((self.horizon,))
      W_block_ado = self.config.W_BLOCK * jnp.ones((self.horizon,))
      W_inner_ado = self.config.W_INNER_ADO * jnp.ones((self.horizon,))
      W_outer_ado = self.config.W_OUTER_ADO * jnp.ones((self.horizon,))
    elif isinstance(cost_param, StaticCostParams):
      _w_ego = nn.softmax(jnp.array([cost_param.W_INNER_EGO, cost_param.W_OUTER_EGO]))
      _w_ado = nn.softmax(jnp.array([cost_param.W_INNER_ADO, cost_param.W_OUTER_ADO]))
      W_overtake = nn.softplus(cost_param.W_OVERTAKE) * jnp.ones((self.horizon,))
      W_stayback = nn.softplus(cost_param.W_STAYBACK) * jnp.ones((self.horizon,))
      W_inner_ego = 2. * _w_ego[0] * jnp.ones((self.horizon,))
      W_outer_ego = 2. * _w_ego[1] * jnp.ones((self.horizon,))
      W_block_ado = nn.softplus(cost_param.W_BLOCK) * jnp.ones((self.horizon,))
      W_inner_ado = _w_ado[0] * jnp.ones((self.horizon,))
      W_outer_ado = _w_ado[1] * jnp.ones((self.horizon,))
    else:
      nn_inputs = self._get_nn_inputs(xs, us, closest_pts, slopes, thetas)
      W_pred = vmap(self.cost_param_model.apply, in_axes=(None, 1),
                    out_axes=(1))(cost_param, nn_inputs)
      W_overtake, W_stayback, W_inner_ego, W_outer_ego, W_block_ado, W_inner_ado, W_outer_ado = (
          W_pred[0, :], W_pred[1, :], W_pred[2, :], W_pred[3, :], W_pred[4, :], W_pred[5, :],
          W_pred[6, :]
      )

    return jnp.vstack(
        (W_overtake, W_stayback, W_inner_ego, W_outer_ego, W_block_ado, W_inner_ado, W_outer_ado)
    )


class KLGameSolver(ILQSolver):

  def __init__(
      self, track: Track, config: Struct, dynamics: MultiPlayerDynamicalSystem, verbose: str = False
  ):
    """
    Initializer.
    """

    ILQSolver.__init__(self, track, config, dynamics, verbose)

  def solve(
      self, cur_state: np.ndarray, us_warmstart: np.ndarray, rMus: List[np.ndarray],
      rSigmas: List[np.ndarray], reg: List[float], ot_mode_idx=0
  ):
    """
    Runs the iLQGame algorithm.

    Args:
        cur_state (np.ndarray): (nx,) Current state.
        us_warmstart (np.ndarray): (nui, N, num_players) Warmstart controls.
        rMus (List[np.ndarray]): multi-modal reference policy means
        rSigmas (List[np.ndarray]): multi-modal reference policy covariance matrices
        reg (List[float]): KL regularization weight

    Returns:
        states: np.ndarray
        controls: np.ndarray
        t_process: float
        status: int
        theta: np.ndarray
    """
    status = 0
    time0 = time.time()
    xs_res = []
    us_res = []
    thetas_res = []

    for rMus_m, rSigmas_m in zip(rMus, rSigmas):

      # Initial forward pass.
      x0, us = jnp.asarray(cur_state), jnp.asarray(us_warmstart)
      xs, cost_init, closest_pts, slopes, thetas = self.initial_forward_pass(x0, us)
      cost_best = cost_init

      # Resets the solver.
      self.reset(_current_x=xs, _current_u=us, _current_J=cost_init)

      # Main loop.
      for iter in range(self.max_iter):

        t_start = time.time()

        # Linearizes joint dynamics about this operating point.
        As, Bs = self.linearize_dynamics(xs, us)

        # Quadraticizes costs.
        lxs, lus, Hxxs, Huus = self.quadraticize_costs(xs, us, closest_pts, slopes, thetas)

        # Backward pass to obtain policy parameters.
        Ps, alphas_bpass, _, _ = self.backward_pass(
            As, Bs, lxs, lus, Hxxs, Huus, rMus_m, rSigmas_m, jnp.asarray(reg)
        )

        # Forward Pass: Computes the operating point and performs line search for alphas.
        for line_search_scaling in self.line_search_scaling:
          alphas_ls = alphas_bpass * line_search_scaling
          xs_ls, us_ls, cost_ls, closest_pts_ls, slopes_ls, thetas_ls = self.compute_operating_point(
              xs, us, Ps, alphas_ls, cur_state
          )
          if cost_ls < cost_best:
            xs = xs_ls
            us = us_ls
            closest_pts = closest_pts_ls
            slopes = slopes_ls
            thetas = thetas_ls
            cost_best = cost_ls
            break

        # region: Updates operating points.
        self.last_operating_point = self.current_operating_point
        self.current_operating_point = (xs, us)

        self.last_social_cost = self.current_social_cost
        self.current_social_cost = cost_best

        if self.current_social_cost < self.best_social_cost:
          self.best_operating_point = self.current_operating_point
          self.best_social_cost = self.current_social_cost
        # endregion

        # region: Checks convergence.
        if self.is_converged_cost():
          status = 1
          if self.verbose:
            print(
                "[KLGame] Social cost (", round(self.current_social_cost, 2), ") has converged! \n"
            )
          break

        if self.verbose:
          print(
              "[KLGame] Iteration", iter, "| Social cost: ", round(self.current_social_cost, 2),
              " | Iter. time: ",
              time.time() - t_start
          )
        # endregion

      # region: Reports results.
      xs_res.append(np.asarray(xs))
      us_res.append(np.asarray(us))
      thetas_res.append(np.asarray(thetas))
      # endregion

    t_process = time.time() - time0

    # region: Returns the KLGame trajecotry.
    xs_klg, us_klg, thetas_klg, mode = self._gets_kl_game_policy(
        xs_res, us_res, thetas_res, ot_mode_idx
    )
    # endregion

    return xs_klg, us_klg, t_process, status, thetas_klg, mode

  def _gets_kl_game_policy(self, xs_res, us_res, thetas_res, ot_mode_idx):
    if len(xs_res) == 1:
      return xs_res[0], us_res[0], thetas_res[0], 0
    else:
      idx = ot_mode_idx
      secondary_idx = 1
      for k in range(self.horizon):
        obs_tuple_cur = get_perfect_obs_two_player(
            xs_res[ot_mode_idx][:, k], us_res[ot_mode_idx][:, k, :], self
        )
        if not is_safe(self.config, obs_tuple_cur):
          idx = secondary_idx
          break
      return xs_res[idx], us_res[idx], thetas_res[idx], idx

  @partial(jit, static_argnames='self')
  def backward_pass(
      self, As: ArrayImpl, Bs: ArrayImpl, lxs: ArrayImpl, lus: ArrayImpl, Hxxs: ArrayImpl,
      Huus: ArrayImpl, rMus: ArrayImpl, rSigmas: ArrayImpl, reg: ArrayImpl
  ) -> Tuple[ArrayImpl, ArrayImpl, ArrayImpl, ArrayImpl]:
    """
    KLGame backward pass.

    Args:
        As (ArrayImpl): (nx, nx, N) A matrices
        Bs (ArrayImpl): (nui, nui, N, num_players) B matrices
        lxs (ArrayImpl): (nx, N, num_players) gradients lx = dc/dx of all players
        lus (ArrayImpl): (nui, N, num_players) gradients lu = dc/du of all players
        Hxxs (ArrayImpl): (nx, nx, N, num_players) Hessians Hxx of all players
        Huus (ArrayImpl): (nui, nui, N, num_players) Hessians Huu of all players
        rMus (ArrayImpl): (nui, N, num_players) reference policy mean of all players
        rSigmas (ArrayImpl): (nui, nui, N, num_players) reference policy covariance of all players
        reg (ArrayImpl): (num_players,) KL regularization weight

    Returns:
        ArrayImpl: Ps (dim_u_ss, dim_x, N-1, num_players)
        ArrayImpl: alphas (dim_u_ss, N-1, num_players)
    """

    @jit
    def backward_pass_looper(k, _carry):
      Ps, alphas, Z, zeta = _carry
      n = horizon - 1 - k

      # Computes Ps given previously computed Z.
      S = jnp.array(()).reshape(0, sum(self.dynamics._u_dims))
      Y1 = jnp.array(()).reshape(0, dim_x)
      for ii in range(num_players):
        Sii = jnp.array(()).reshape(dim_u_ss, 0)
        for jj in range(num_players):
          if jj == ii:
            Sii = jnp.hstack((
                Sii, Bs[:, :, n, ii].T @ Z[:, :, ii] @ Bs[:, :, n, jj] + Huus[:, :, n, ii]
                + reg[ii] * jnp.linalg.inv(rSigmas[:, :, n, ii])
            ))
          else:
            Sii = jnp.hstack((Sii, Bs[:, :, n, ii].T @ Z[:, :, ii] @ Bs[:, :, n, jj]))
        S = jnp.vstack((S, Sii))

        Y1ii = Bs[:, :, n, ii].T @ Z[:, :, ii] @ As[:, :, n]
        Y1 = jnp.vstack((Y1, Y1ii))

      P, _, _, _ = jnp.linalg.lstsq(a=S, b=Y1, rcond=None)
      # Sinv = jnp.linalg.pinv(S)
      # P = Sinv @ Y1

      for ii in range(num_players):
        Pii = self.dynamics._LMu[:, :, ii] @ P
        Ps = Ps.at[:, :, n, ii].set(Pii)

      # Computes F_k = A_k - B1_k P1_k - B2_k P2_k -...
      F = As[:, :, n]
      for ii in range(num_players):
        F -= Bs[:, :, n, ii] @ Ps[:, :, n, ii]

      # Computes alphas using previously computed zetas.
      Y2 = jnp.array(()).reshape(0, 1)
      for ii in range(num_players):
        # Y2ii = (Bs[:, :, n, ii].T @ zeta[:, ii]).reshape((dim_u_ss, 1))
        Y2ii = (
            Bs[:, :, n, ii].T @ zeta[:, ii] + lus[:, n, ii]
            - reg[ii] * jnp.linalg.inv(rSigmas[:, :, n, ii]) @ rMus[:, n, ii]
        ).reshape((dim_u_ss, 1))
        Y2 = jnp.vstack((Y2, Y2ii))

      alpha, _, _, _ = jnp.linalg.lstsq(a=S, b=Y2, rcond=None)
      # alpha = Sinv @ Y2

      for ii in range(num_players):
        alphaii = self.dynamics._LMu[:, :, ii] @ alpha
        alphas = alphas.at[:, n, ii].set(alphaii[:, 0])

      # Computes beta_k = -B1_k alpha1 - B2_k alpha2_k -...
      beta = 0.
      for ii in range(num_players):
        beta -= Bs[:, :, n, ii] @ alphas[:, n, ii]

      # Updates zeta.
      for ii in range(num_players):
        _FZb = F.T @ (zeta[:, ii] + Z[:, :, ii] @ beta)
        _PRa = Ps[:, :, n, ii].T @ Huus[:, :, n, ii] @ alphas[:, n, ii]
        _PSa = (
            reg[ii] * Ps[:, :, n, ii].T @ jnp.linalg.inv(rSigmas[:, :, n, ii])
            @ (alphas[:, n, ii] - rMus[:, n, ii])
        )
        zeta = zeta.at[:, ii].set(_FZb + _PRa + lxs[:, n, ii] + _PSa)

      # Updates Z.
      for ii in range(num_players):
        _FZF = F.T @ Z[:, :, ii] @ F
        _PRP = Ps[:, :, n, ii].T @ Huus[:, :, n, ii] @ Ps[:, :, n, ii]
        _PSP = reg[ii] * Ps[:, :, n, ii].T @ jnp.linalg.inv(rSigmas[:, :, n, ii]) @ Ps[:, :, n, ii]
        Z = Z.at[:, :, ii].set(_FZF + _PRP + Hxxs[:, :, n, ii] + _PSP)

      return Ps, alphas, Z, zeta

    # Unpacks horizon and number of players.
    horizon = self.horizon
    num_players = self.num_players

    # Caches dimensions of state and controls for each player.
    dim_x = self.dim_x
    dim_u_ss = self.dim_u_ss

    # Recursively computes all intermediate and final variables.
    Z = Hxxs[:, :, -1, :]
    zeta = lxs[:, -1, :]

    # Initializes strategy matrices.
    Ps = jnp.zeros((dim_u_ss, dim_x, horizon, num_players))
    alphas = jnp.zeros((dim_u_ss, horizon, num_players))

    # Backward pass.
    Ps, alphas, Z, zeta = lax.fori_loop(
        0, self.horizon, backward_pass_looper, (Ps, alphas, Z, zeta)
    )

    return Ps, alphas, Z, zeta
