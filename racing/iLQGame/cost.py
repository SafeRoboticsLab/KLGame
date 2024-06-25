"""
Game costs, gradients, and Hessians.

Please contact the author(s) of this library if you have any questions.
Author: Haimin Hu (haiminh@princeton.edu)
"""

from typing import Tuple

from functools import partial
from jax import lax, jit, vmap, jacfwd, hessian
from jaxlib.xla_extension import ArrayImpl
import jax.numpy as jnp

from iLQR import Struct, Dynamics


class MultiPlayerRacingCost:

  def __init__(
      self, config: Struct, dim_x: int, subsys: Dynamics, LMx: ArrayImpl, is_prox_cost: bool = True,
      is_ego: bool = True
  ):
    self.config = config

    # Planning parameters.
    self.N = config.N  # number of planning steps.

    # System parameters.
    self.dim_x = dim_x
    self.subsys = subsys
    self.dim_x_ss = self.subsys.dim_x
    self.dim_u_ss = self.subsys.dim_u

    # Lifting matrix
    assert isinstance(LMx, ArrayImpl)
    self.LMx = LMx[:, :, subsys.idx]
    if self.subsys.idx == 0:
      self.LMx_ado = LMx[:, :, 1]
    elif self.subsys.idx == 1:
      self.LMx_ado = LMx[:, :, 0]
    else:
      raise NotImplementedError

    # Racing cost parameters.
    self.w_vel = config.W_VEL
    self.w_contour = config.W_CONTOUR
    self.w_theta = config.W_THETA
    self.w_accel = config.W_ACCEL
    self.w_delta = config.W_DELTA
    self.wheelbase = config.WHEELBASE
    self.track_offset = config.TRACK_OFFSET
    self.W_state = jnp.array([[self.w_contour, 0], [0, self.w_vel]])
    self.W_state_contour = 2. * jnp.array([[self.w_contour, 0], [0, 0]])
    self.W_control = jnp.array([[self.w_accel, 0], [0, self.w_delta]])
    self.ref_vel_scaling = self.subsys.ref_vel_scaling

    # Soft constraint parameters.
    self.q1_v = config.Q1_V
    self.q2_v = config.Q2_V
    self.q1_road = config.Q1_ROAD
    self.q2_road = config.Q2_ROAD
    self.q1_lat = config.Q1_LAT
    self.q2_lat = config.Q2_LAT
    self.q1_prox = config.Q1_PROX
    self.q2_prox = config.Q2_PROX
    self.barrier_thr = config.BARRIER_THR
    self.barrier_thr_road = config.BARRIER_THR_ROAD
    self.barrier_thr_lat = config.BARRIER_THR_LAT
    self.barrier_thr_prox = config.BARRIER_THR_PROX

    self.v_min = config.V_MIN
    self.v_max = config.V_MAX
    self.a_min = config.A_MIN
    self.a_max = config.A_MAX
    self.delta_min = config.DELTA_MIN
    self.delta_max = config.DELTA_MAX
    self.alat_min = config.ALAT_MIN
    self.alat_max = config.ALAT_MAX
    self.r = config.WIDTH / 2.
    self.road_thr = config.ROAD_THR

    # Assigns indices for each subsystem [x, y, psi] (for proximity_cost).
    if self.subsys.idx == 0:
      self.ego_idx = (0, 1, 3)
      self.ado_idx = (4, 5, 7)
    elif self.subsys.idx == 1:
      self.ego_idx = (4, 5, 7)
      self.ado_idx = (0, 1, 3)
    else:
      raise NotImplementedError

    # Assigns subsystem indices (for opinion-related costs)
    if self.subsys.idx == 0:
      self.ego_subsys_idx = 0
      self.ado_subsys_idx = 1
    elif self.subsys.idx == 1:
      self.ego_subsys_idx = 1
      self.ado_subsys_idx = 0
    else:
      raise NotImplementedError

    self.is_prox_cost = is_prox_cost
    self.is_ego = is_ego

  @partial(jit, static_argnames='self')
  def get_cost(
      self, states: ArrayImpl, controls: ArrayImpl, closest_pts: ArrayImpl, slopes: ArrayImpl,
      thetas: ArrayImpl, cost_param: ArrayImpl
  ) -> ArrayImpl:
    """
    Calculates the cost given state and control trajectories.

    Args:
        states (ArrayImpl): (dim_x, N) array of planned trajectory.
        controls (ArrayImpl): (2, N) array of planned control.
        closest_pts (ArrayImpl): (7, N)
        slopes (ArrayImpl): (1, N, num_players)
        thetas (ArrayImpl): (1, N, num_players)
        cost_param (ArrayImpl): (N_param, N)

    Returns:
        ArrayImpl: total cost.
    """
    # Gets cost weights.
    w_overtake = cost_param[0, :]
    w_stayback = cost_param[1, :]
    w_inner_ego = cost_param[2, :]
    w_outer_ego = cost_param[3, :]
    w_block_ado = cost_param[4, :]
    w_inner_ado = cost_param[5, :]
    w_outer_ado = cost_param[6, :]

    # vmap all costs.
    c_state_vmap = vmap(self.state_cost_stage, in_axes=(1, 1, 1), out_axes=(0))
    c_inner_vmap = vmap(self.drive_inner_cost_stage, in_axes=(1, 1, 1), out_axes=(0))
    c_outer_vmap = vmap(self.drive_outer_cost_stage, in_axes=(1, 1, 1), out_axes=(0))
    c_cntrl_vmap = vmap(self.control_cost_stage, in_axes=(1), out_axes=(0))
    c_progs_vmap = vmap(self.progress_cost_stage, in_axes=(1), out_axes=(0))
    c_latac_vmap = vmap(self.lat_acc_cost_stage, in_axes=(1, 1), out_axes=(0))
    c_rdbry_vmap = vmap(self.road_boundary_cost_stage, in_axes=(1, 1, 1), out_axes=(0))
    c_velbd_vmap = vmap(self.vel_bound_cost_stage, in_axes=(1), out_axes=(0))
    c_proxi_vmap = vmap(self.proximity_cost_stage, in_axes=(1), out_axes=(0))
    c_ovrtk_vmap = vmap(self.overtake_cost_stage, in_axes=(1), out_axes=(0))
    c_stybk_vmap = vmap(self.stayback_cost_stage, in_axes=(1), out_axes=(0))
    c_block_vmap = vmap(self.block_cost_stage, in_axes=(1), out_axes=(0))

    # Evaluates all cost terms.
    c_state = c_state_vmap(states, closest_pts, slopes)
    c_inner = c_inner_vmap(states, closest_pts, slopes)
    c_outer = c_outer_vmap(states, closest_pts, slopes)
    c_cntrl = c_cntrl_vmap(controls)
    c_progs = c_progs_vmap(thetas)
    c_latac = c_latac_vmap(states, controls)
    c_rdbry = c_rdbry_vmap(states, closest_pts, slopes)
    c_velbd = c_velbd_vmap(states)
    c_proxi = c_proxi_vmap(states)
    c_ovrtk = c_ovrtk_vmap(thetas)
    c_stybk = c_stybk_vmap(thetas)
    c_block = c_block_vmap(thetas)

    # Sums up all cost terms.
    J = jnp.sum(c_state + c_cntrl + c_progs + c_latac + c_rdbry + c_velbd)
    if self.is_prox_cost:
      J += jnp.sum(c_proxi)
    if self.is_ego:
      J += jnp.inner(w_overtake, c_ovrtk) + jnp.inner(w_stayback, c_stybk)
      J += jnp.inner(w_inner_ego, c_inner) + jnp.inner(w_outer_ego, c_outer)
    else:
      J += jnp.inner(w_block_ado, c_block)
      J += jnp.inner(w_inner_ado, c_inner) + jnp.inner(w_outer_ado, c_outer)

    return J

  @partial(jit, static_argnames='self')
  def get_cost_performance(
      self, states: ArrayImpl, controls: ArrayImpl, closest_pts: ArrayImpl, slopes: ArrayImpl,
      thetas: ArrayImpl
  ) -> ArrayImpl:
    """
    Calculates the performance cost given state and control trajectories.

    Args:
        states (ArrayImpl): (dim_x, N) array of planned trajectory.
        controls (ArrayImpl): (2, N) array of planned control.
        closest_pts (ArrayImpl): (7, N)
        slopes (ArrayImpl): (1, N, num_players)
        thetas (ArrayImpl): (1, N, num_players)

    Returns:
        ArrayImpl: total cost.
    """

    # vmap all costs.
    c_state_vmap = vmap(self.state_cost_stage, in_axes=(1, 1, 1), out_axes=(0))
    c_cntrl_vmap = vmap(self.control_cost_stage, in_axes=(1), out_axes=(0))
    # c_progs_vmap = vmap(self.progress_cost_stage, in_axes=(1), out_axes=(0))
    c_latac_vmap = vmap(self.lat_acc_cost_stage, in_axes=(1, 1), out_axes=(0))
    c_rdbry_vmap = vmap(self.road_boundary_cost_stage, in_axes=(1, 1, 1), out_axes=(0))
    c_velbd_vmap = vmap(self.vel_bound_cost_stage, in_axes=(1), out_axes=(0))
    c_proxi_vmap = vmap(self.proximity_cost_stage, in_axes=(1), out_axes=(0))

    # Evaluates all cost terms.
    c_state = c_state_vmap(states, closest_pts, slopes)
    c_cntrl = c_cntrl_vmap(controls)
    # c_progs = c_progs_vmap(thetas)
    c_latac = c_latac_vmap(states, controls)
    c_rdbry = c_rdbry_vmap(states, closest_pts, slopes)
    c_velbd = c_velbd_vmap(states)
    c_proxi = c_proxi_vmap(states)

    # Sums up all cost terms.
    J = jnp.sum(c_state + c_cntrl + c_latac + c_rdbry + c_velbd)
    if self.is_prox_cost:
      J += jnp.sum(c_proxi)

    return J

  @partial(jit, static_argnames='self')
  def get_cost_derivatives(
      self, states: ArrayImpl, controls: ArrayImpl, closest_pts: ArrayImpl, slopes: ArrayImpl,
      thetas: ArrayImpl, cost_param: ArrayImpl
  ) -> Tuple[ArrayImpl, ArrayImpl, ArrayImpl, ArrayImpl]:
    """
    Calculates gradients and Hessians of the costs.

    Args:
        states (ArrayImpl): (dim_x, N) array of planned trajectory.
        controls (ArrayImpl): (2, N) array of planned control.
        closest_pts (ArrayImpl): (7, N)
        slopes (ArrayImpl): (1, N, num_players)
        thetas (ArrayImpl): (1, N, num_players)
        cost_param (ArrayImpl): (2, N) array of (W_OVERTAKE, W_STAYBACK)

    Returns:
        lxs (ArrayImpl): (nx, N) gradients lx = dc/dx
        lus (ArrayImpl): (nui, N) gradients lu = dc/du
        Hxxs (ArrayImpl): (nx, nx, N) Hessians Hxx
        Huus (ArrayImpl): (nui, nui, N) Hessians Huu
    """
    # Gets cost weights.
    w_overtake = cost_param[0, :]
    w_stayback = cost_param[1, :]
    w_inner_ego = cost_param[2, :]
    w_outer_ego = cost_param[3, :]
    w_block_ado = cost_param[4, :]
    w_inner_ado = cost_param[5, :]
    w_outer_ado = cost_param[6, :]

    # Creates cost gradient functions.
    lx_state_fn = jacfwd(self.state_cost_stage, argnums=0)
    lx_inner_fn = jacfwd(self.drive_inner_cost_stage, argnums=0)
    lx_outer_fn = jacfwd(self.drive_outer_cost_stage, argnums=0)
    lx_progs_fn = self.progress_cost_stage_deriv
    lx_latac_fn = jacfwd(self.lat_acc_cost_stage, argnums=0)
    lx_rdbry_fn = jacfwd(self.road_boundary_cost_stage, argnums=0)
    lx_velbd_fn = jacfwd(self.vel_bound_cost_stage, argnums=0)
    lx_proxi_fn = jacfwd(self.proximity_cost_stage, argnums=0)
    lx_ovrtk_fn = self.overtake_cost_stage_deriv
    lx_stybk_fn = self.stayback_cost_stage_deriv
    lx_block_fn = self.block_cost_stage_deriv

    lu_cntrl_fn = jacfwd(self.control_cost_stage, argnums=0)
    lu_latac_fn = jacfwd(self.lat_acc_cost_stage, argnums=1)

    # Creates cost Hessian functions.
    Hxx_state_fn = hessian(self.state_cost_stage, argnums=0)
    Hxx_inner_fn = hessian(self.drive_inner_cost_stage, argnums=0)
    Hxx_outer_fn = hessian(self.drive_outer_cost_stage, argnums=0)
    Hxx_latac_fn = hessian(self.lat_acc_cost_stage, argnums=0)
    Hxx_rdbry_fn = hessian(self.road_boundary_cost_stage, argnums=0)
    Hxx_velbd_fn = hessian(self.vel_bound_cost_stage, argnums=0)
    Hxx_proxi_fn = hessian(self.proximity_cost_stage, argnums=0)

    Huu_cntrl_fn = hessian(self.control_cost_stage, argnums=0)
    Huu_latac_fn = hessian(self.lat_acc_cost_stage, argnums=1)

    # vmap all gradients and Hessians.
    lx_state_vmap = vmap(lx_state_fn, in_axes=(1, 1, 1), out_axes=(1))
    lx_inner_vmap = vmap(lx_inner_fn, in_axes=(1, 1, 1), out_axes=(1))
    lx_outer_vmap = vmap(lx_outer_fn, in_axes=(1, 1, 1), out_axes=(1))
    lx_progs_vmap = vmap(lx_progs_fn, in_axes=(1), out_axes=(1))
    lx_latac_vmap = vmap(lx_latac_fn, in_axes=(1, 1), out_axes=(1))
    lx_rdbry_vmap = vmap(lx_rdbry_fn, in_axes=(1, 1, 1), out_axes=(1))
    lx_velbd_vmap = vmap(lx_velbd_fn, in_axes=(1), out_axes=(1))
    lx_proxi_vmap = vmap(lx_proxi_fn, in_axes=(1), out_axes=(1))
    lx_ovrtk_vmap = vmap(lx_ovrtk_fn, in_axes=(1, 1), out_axes=(1))
    lx_stybk_vmap = vmap(lx_stybk_fn, in_axes=(1, 1), out_axes=(1))
    lx_block_vmap = vmap(lx_block_fn, in_axes=(1, 1), out_axes=(1))

    lu_cntrl_vmap = vmap(lu_cntrl_fn, in_axes=(1), out_axes=(1))
    lu_latac_vmap = vmap(lu_latac_fn, in_axes=(1, 1), out_axes=(1))

    Hxx_state_vmap = vmap(Hxx_state_fn, in_axes=(1, 1, 1), out_axes=(2))
    Hxx_inner_vmap = vmap(Hxx_inner_fn, in_axes=(1, 1, 1), out_axes=(2))
    Hxx_outer_vmap = vmap(Hxx_outer_fn, in_axes=(1, 1, 1), out_axes=(2))
    Hxx_latac_vmap = vmap(Hxx_latac_fn, in_axes=(1, 1), out_axes=(2))
    Hxx_rdbry_vmap = vmap(Hxx_rdbry_fn, in_axes=(1, 1, 1), out_axes=(2))
    Hxx_velbd_vmap = vmap(Hxx_velbd_fn, in_axes=(1), out_axes=(2))
    Hxx_proxi_vmap = vmap(Hxx_proxi_fn, in_axes=(1), out_axes=(2))

    Huu_cntrl_vmap = vmap(Huu_cntrl_fn, in_axes=(1), out_axes=(2))
    Huu_latac_vmap = vmap(Huu_latac_fn, in_axes=(1, 1), out_axes=(2))

    # Computes cost gradients and Hessians.
    lx_state = lx_state_vmap(states, closest_pts, slopes)
    lx_inner = lx_inner_vmap(states, closest_pts, slopes)
    lx_outer = lx_outer_vmap(states, closest_pts, slopes)
    lx_progs = lx_progs_vmap(slopes)
    lx_latac = lx_latac_vmap(states, controls)
    lx_rdbry = lx_rdbry_vmap(states, closest_pts, slopes)
    lx_velbd = lx_velbd_vmap(states)
    lx_proxi = lx_proxi_vmap(states)
    lx_ovrtk = lx_ovrtk_vmap(thetas, slopes)
    lx_stybk = lx_stybk_vmap(thetas, slopes)
    lx_block = lx_block_vmap(thetas, slopes)

    lu_cntrl = lu_cntrl_vmap(controls)
    lu_latac = lu_latac_vmap(states, controls)

    Hxx_state = Hxx_state_vmap(states, closest_pts, slopes)
    Hxx_inner = Hxx_inner_vmap(states, closest_pts, slopes)
    Hxx_outer = Hxx_outer_vmap(states, closest_pts, slopes)
    Hxx_latac = Hxx_latac_vmap(states, controls)
    Hxx_rdbry = Hxx_rdbry_vmap(states, closest_pts, slopes)
    Hxx_velbd = Hxx_velbd_vmap(states)
    Hxx_proxi = Hxx_proxi_vmap(states)

    Huu_cntrl = Huu_cntrl_vmap(controls)
    Huu_latac = Huu_latac_vmap(states, controls)

    lxs = lx_state + lx_progs + lx_latac + lx_rdbry + lx_velbd
    lus = lu_cntrl + lu_latac
    Hxxs = Hxx_state + Hxx_latac + Hxx_rdbry + Hxx_velbd
    Huus = Huu_cntrl + Huu_latac

    if self.is_prox_cost:
      lxs += lx_proxi
      Hxxs += Hxx_proxi

    if self.is_ego:
      lxs += w_overtake*lx_ovrtk + w_stayback*lx_stybk
      lxs += w_inner_ego*lx_inner + w_outer_ego*lx_outer
      Hxxs += w_inner_ego*Hxx_inner + w_outer_ego*Hxx_outer
    else:
      lxs += w_block_ado * lx_block
      lxs += w_inner_ado*lx_inner + w_outer_ado*lx_outer

    return lxs, lus, Hxxs, Huus

  # --------------- Racing costs -------------------------------------------------------------------
  @partial(jit, static_argnames='self')
  def state_cost_stage(
      self, state: ArrayImpl, closest_pt: ArrayImpl, slope: ArrayImpl
  ) -> ArrayImpl:
    """
    Computes the stage state cost.

    Args:
        state (ArrayImpl): (dim_x,) Joint system state
        closest_pt (ArrayImpl): (7,)
        slope (ArrayImpl): (1, num_players)

    Returns:
        ArrayImpl: cost (scalar)
    """
    sr = jnp.sin(slope[0, self.ego_subsys_idx])
    cr = jnp.cos(slope[0, self.ego_subsys_idx])
    transform = jnp.array([[sr, -cr, 0., 0.], [0., 0., 1., 0.]])
    Q = transform.T @ self.W_state @ transform
    ref_state = jnp.zeros((self.dim_x_ss,))
    ref_state = ref_state.at[0].set(closest_pt[0] + sr * self.track_offset)
    ref_state = ref_state.at[1].set(closest_pt[1] - cr * self.track_offset)
    ref_state = ref_state.at[2].set(closest_pt[6] * self.ref_vel_scaling)
    return (self.LMx @ state - ref_state).T @ Q @ (self.LMx @ state - ref_state)

  @partial(jit, static_argnames='self')
  def control_cost_stage(self, control: ArrayImpl) -> ArrayImpl:
    """
    Computes the stage control cost c(u) = u.T @ R @ u, where u is the control.

    Args:
        control (ArrayImpl): (dim_u_ss,)

    Returns:
        ArrayImpl: cost (scalar)
    """
    return control.T @ self.W_control @ control

  @partial(jit, static_argnames='self')
  def progress_cost_stage(self, theta: ArrayImpl) -> ArrayImpl:
    """
    Computes the progress cost.

    Args:
        theta (ArrayImpl): (1, num_players) progress at each state.

    NOTE: Use progress_cost_stage_deriv() instead of autodiff.

    Returns:
        ArrayImpl: cost (scalar)
    """
    return -self.w_theta * theta[0, self.ego_subsys_idx]

  @partial(jit, static_argnames='self')
  def progress_cost_stage_deriv(self, slope: ArrayImpl) -> ArrayImpl:
    """
    Computes the derivative of the progress cost w.r.t. the slope.

    Args:
        slope (ArrayImpl): (1, num_players) track reference slopes.

    Returns:
        ArrayImpl: cost (scalar)
    """
    sr = jnp.sin(slope[0, self.ego_subsys_idx])
    cr = jnp.cos(slope[0, self.ego_subsys_idx])
    grad_subsys = -self.w_theta * jnp.hstack((cr, sr, 0., 0.))
    return grad_subsys.T @ self.LMx

  @partial(jit, static_argnames='self')
  def overtake_cost_stage(self, thetas: ArrayImpl) -> ArrayImpl:
    """
    Computes the overtake cost.

    Args:
        thetas (ArrayImpl): (1, num_players) progress at each state of each car.

    NOTE: Use overtake_cost_stage_deriv() instead of autodiff.

    Returns:
        ArrayImpl: cost (scalar)
    """
    _ego_subsys_idx = self.ego_subsys_idx
    _ado_subsys_idx = self.ado_subsys_idx
    penalty = -jnp.clip(
        thetas[0, _ego_subsys_idx] - thetas[0, _ado_subsys_idx], None, self.config.OVERTAKE_THRES
    )
    return penalty

  @partial(jit, static_argnames='self')
  def overtake_cost_stage_deriv(self, thetas: ArrayImpl, slopes: ArrayImpl) -> ArrayImpl:
    """
    Computes the derivative of the overtake cost w.r.t. the slope.

    Args:
        thetas (ArrayImpl): (1, num_players) progress at each state of each car.
        slopes (ArrayImpl): (1, num_players) track reference slopes.

    Returns:
        ArrayImpl: cost (scalar)
    """

    def true_fn(thetas):
      sr_ego, cr_ego = jnp.sin(slopes[0, _ego_subsys_idx]), jnp.cos(slopes[0, _ego_subsys_idx])
      sr_ado, cr_ado = jnp.sin(slopes[0, _ado_subsys_idx]), jnp.cos(slopes[0, _ado_subsys_idx])
      grad_ego = jnp.hstack((cr_ego, sr_ego, 0., 0.)).T @ self.LMx
      grad_ado = jnp.hstack((cr_ado, sr_ado, 0., 0.)).T @ self.LMx_ado
      return grad_ado - grad_ego

    def false_fn(thetas):
      return jnp.zeros((4,)).T @ self.LMx

    _ego_subsys_idx = self.ego_subsys_idx
    _ado_subsys_idx = self.ado_subsys_idx
    pred = thetas[0, _ego_subsys_idx] - thetas[0, _ado_subsys_idx] <= self.config.OVERTAKE_THRES

    return lax.cond(pred, true_fn, false_fn, thetas)

  @partial(jit, static_argnames='self')
  def drive_inner_cost_stage(
      self, state: ArrayImpl, closest_pt: ArrayImpl, slope: ArrayImpl
  ) -> ArrayImpl:
    """
    Computes the state cost for encouraging the inner lane.

    Args:
        state (ArrayImpl): (dim_x,) Joint system state
        closest_pt (ArrayImpl): (7,)
        slope (ArrayImpl): (1, num_players)

    Returns:
        ArrayImpl: cost (scalar)
    """
    sr = jnp.sin(slope[0, self.ego_subsys_idx])
    cr = jnp.cos(slope[0, self.ego_subsys_idx])
    transform = jnp.array([[sr, -cr, 0., 0.], [0., 0., 1., 0.]])
    Q = transform.T @ self.W_state_contour @ transform
    ref_state = jnp.zeros((self.dim_x_ss,))
    ref_state = ref_state.at[0].set((closest_pt[0] + closest_pt[2]) / 2. + sr * self.track_offset)
    ref_state = ref_state.at[1].set((closest_pt[1] + closest_pt[3]) / 2. - cr * self.track_offset)
    return (self.LMx @ state - ref_state).T @ Q @ (self.LMx @ state - ref_state)

  @partial(jit, static_argnames='self')
  def drive_outer_cost_stage(
      self, state: ArrayImpl, closest_pt: ArrayImpl, slope: ArrayImpl
  ) -> ArrayImpl:
    """
    Computes the state cost for encouraging the outer lane.

    Args:
        state (ArrayImpl): (dim_x,) Joint system state
        closest_pt (ArrayImpl): (7,)
        slope (ArrayImpl): (1, num_players)

    Returns:
        ArrayImpl: cost (scalar)
    """
    sr = jnp.sin(slope[0, self.ego_subsys_idx])
    cr = jnp.cos(slope[0, self.ego_subsys_idx])
    transform = jnp.array([[sr, -cr, 0., 0.], [0., 0., 1., 0.]])
    Q = transform.T @ self.W_state_contour @ transform
    ref_state = jnp.zeros((self.dim_x_ss,))
    ref_state = ref_state.at[0].set((closest_pt[0] + closest_pt[4]) / 2. + sr * self.track_offset)
    ref_state = ref_state.at[1].set((closest_pt[1] + closest_pt[5]) / 2. - cr * self.track_offset)
    return (self.LMx @ state - ref_state).T @ Q @ (self.LMx @ state - ref_state)

  @partial(jit, static_argnames='self')
  def block_cost_stage(self, thetas: ArrayImpl) -> ArrayImpl:
    """
    Computes the ado's block cost.

    Args:
        thetas (ArrayImpl): (1, num_players) progress at each state of each car.

    NOTE: Use block_cost_stage_deriv() instead of autodiff.

    Returns:
        ArrayImpl: cost (scalar)
    """
    _ego_subsys_idx = self.ego_subsys_idx
    _ado_subsys_idx = self.ado_subsys_idx
    penalty = jnp.clip(
        thetas[0, _ado_subsys_idx] - thetas[0, _ego_subsys_idx], self.config.BLOCK_THRES_LB,
        self.config.BLOCK_THRES_UB
    )
    return penalty

  @partial(jit, static_argnames='self')
  def block_cost_stage_deriv(self, thetas: ArrayImpl, slopes: ArrayImpl) -> ArrayImpl:
    """
    Computes the derivative of the block cost w.r.t. the slope.

    Args:
        thetas (ArrayImpl): (1, num_players) progress at each state of each car.
        slopes (ArrayImpl): (1, num_players) track reference slopes.

    Returns:
        ArrayImpl: cost (scalar)
    """

    def true_fn(thetas):
      sr_ego, cr_ego = jnp.sin(slopes[0, _ego_subsys_idx]), jnp.cos(slopes[0, _ego_subsys_idx])
      sr_ado, cr_ado = jnp.sin(slopes[0, _ado_subsys_idx]), jnp.cos(slopes[0, _ado_subsys_idx])
      grad_ego = jnp.hstack((cr_ego, sr_ego, 0., 0.)).T @ self.LMx
      grad_ado = jnp.hstack((cr_ado, sr_ado, 0., 0.)).T @ self.LMx_ado
      return grad_ado - grad_ego

    def false_fn(thetas):
      return jnp.zeros((4,)).T @ self.LMx

    _ego_subsys_idx = self.ego_subsys_idx
    _ado_subsys_idx = self.ado_subsys_idx
    pred_lb = self.config.BLOCK_THRES_LB <= thetas[0, _ado_subsys_idx] - thetas[0, _ego_subsys_idx]
    pred_ub = thetas[0, _ado_subsys_idx] - thetas[0, _ego_subsys_idx] <= self.config.BLOCK_THRES_UB
    pred = pred_lb & pred_ub

    return lax.cond(pred, true_fn, false_fn, thetas)

  @partial(jit, static_argnames='self')
  def yield_cost_stage(self, thetas: ArrayImpl) -> ArrayImpl:
    """
    Computes the ado's yield cost.

    Args:
        thetas (ArrayImpl): (1, num_players) progress at each state of each car.

    NOTE: Use block_cost_stage_deriv() instead of autodiff.

    Returns:
        ArrayImpl: cost (scalar)
    """
    _ego_subsys_idx = self.ego_subsys_idx
    _ado_subsys_idx = self.ado_subsys_idx
    penalty = jnp.clip(
        thetas[0, _ado_subsys_idx] - thetas[0, _ego_subsys_idx], self.config.YIELD_THRES_LB,
        self.config.YIELD_THRES_UB
    )
    return penalty

  @partial(jit, static_argnames='self')
  def yield_cost_stage_deriv(self, thetas: ArrayImpl, slopes: ArrayImpl) -> ArrayImpl:
    """
    Computes the derivative of the block cost w.r.t. the slope.

    Args:
        thetas (ArrayImpl): (1, num_players) progress at each state of each car.
        slopes (ArrayImpl): (1, num_players) track reference slopes.

    Returns:
        ArrayImpl: cost (scalar)
    """

    def true_fn(thetas):
      sr_ego, cr_ego = jnp.sin(slopes[0, _ego_subsys_idx]), jnp.cos(slopes[0, _ego_subsys_idx])
      sr_ado, cr_ado = jnp.sin(slopes[0, _ado_subsys_idx]), jnp.cos(slopes[0, _ado_subsys_idx])
      grad_ego = jnp.hstack((cr_ego, sr_ego, 0., 0.)).T @ self.LMx
      grad_ado = jnp.hstack((cr_ado, sr_ado, 0., 0.)).T @ self.LMx_ado
      return grad_ado - grad_ego

    def false_fn(thetas):
      return jnp.zeros((4,)).T @ self.LMx

    _ego_subsys_idx = self.ego_subsys_idx
    _ado_subsys_idx = self.ado_subsys_idx
    pred_lb = self.config.YIELD_THRES_LB <= thetas[0, _ado_subsys_idx] - thetas[0, _ego_subsys_idx]
    pred_ub = thetas[0, _ado_subsys_idx] - thetas[0, _ego_subsys_idx] <= self.config.YIELD_THRES_UB
    pred = pred_lb & pred_ub

    return lax.cond(pred, true_fn, false_fn, thetas)

  @partial(jit, static_argnames='self')
  def stayback_cost_stage(self, thetas: ArrayImpl) -> ArrayImpl:
    """
    Computes the stayback cost.

    Args:
        thetas (ArrayImpl): (1, num_players) progress at each state of each car.

    NOTE: Use stayback_cost_stage_deriv() instead of autodiff.

    Returns:
        ArrayImpl: cost (scalar)
    """

    _ego_subsys_idx = self.ego_subsys_idx
    _ado_subsys_idx = self.ado_subsys_idx
    penalty = jnp.clip(
        thetas[0, _ego_subsys_idx] - thetas[0, _ado_subsys_idx] - self.config.STAYBACK_THRES, 0.,
        None
    )
    return penalty

  @partial(jit, static_argnames='self')
  def stayback_cost_stage_deriv(self, thetas: ArrayImpl, slopes: ArrayImpl) -> ArrayImpl:
    """
    Computes the derivative of the stayback cost w.r.t. the slope.

    Args:
        thetas (ArrayImpl): (1, num_players) progress at each state of each car.
        slopes (ArrayImpl): (1, num_players) track reference slopes.

    Returns:
        ArrayImpl: cost (scalar)
    """

    def true_fn(thetas):
      sr_ego, cr_ego = jnp.sin(slopes[0, _ego_subsys_idx]), jnp.cos(slopes[0, _ego_subsys_idx])
      sr_ado, cr_ado = jnp.sin(slopes[0, _ado_subsys_idx]), jnp.cos(slopes[0, _ado_subsys_idx])
      grad_ego = jnp.hstack((cr_ego, sr_ego, 0., 0.)).T @ self.LMx
      grad_ado = jnp.hstack((cr_ado, sr_ado, 0., 0.)).T @ self.LMx_ado
      return grad_ego - grad_ado

    def false_fn(thetas):
      return jnp.zeros((4,)).T @ self.LMx

    _ego_subsys_idx = self.ego_subsys_idx
    _ado_subsys_idx = self.ado_subsys_idx
    pred = thetas[0, _ego_subsys_idx] - thetas[0, _ado_subsys_idx] >= self.config.STAYBACK_THRES

    return lax.cond(pred, true_fn, false_fn, thetas)

  # --------------- Soft constraint costs ----------------------------------------------------------
  @partial(jit, static_argnames='self')
  def lat_acc_cost_stage(self, state: ArrayImpl, control: ArrayImpl) -> ArrayImpl:
    """
    Calculates the lateral acceleration soft constraint cost.

    Args:
        state (ArrayImpl): (dim_x,) Joint system state
        control (ArrayImpl): (dim_u_ss,)

    Returns:
        ArrayImpl: cost (scalar)
    """
    state = self.LMx @ state
    accel = state[2]**2 * jnp.tan(control[1]) / self.wheelbase
    error_ub = accel - self.alat_max
    error_lb = self.alat_min - accel

    c_lat_ub = self.q1_lat * (
        jnp.exp(jnp.clip(self.q2_lat * error_ub, None, self.barrier_thr_lat)) - 1.
    )
    c_lat_lb = self.q1_lat * (
        jnp.exp(jnp.clip(self.q2_lat * error_lb, None, self.barrier_thr_lat)) - 1.
    )
    return c_lat_lb + c_lat_ub

  @partial(jit, static_argnames='self')
  def road_boundary_cost_stage(
      self, state: ArrayImpl, closest_pt: ArrayImpl, slope: ArrayImpl
  ) -> ArrayImpl:
    """
    Calculates the road boundary soft constraint cost.

    Args:
        state (ArrayImpl): (dim_x,) Joint system state
        closest_pt (ArrayImpl): (7,)
        slope (ArrayImpl): (1, num_players)

    Returns:
        ArrayImpl: cost (scalar)
    """
    state = self.LMx @ state
    sr = jnp.sin(slope[0, self.ego_subsys_idx])
    cr = jnp.cos(slope[0, self.ego_subsys_idx])
    dx = state[0] - closest_pt[0]
    dy = state[1] - closest_pt[1]
    dis = sr*dx - cr*dy

    # Right bound (outer edge).
    width_R = jnp.sqrt((closest_pt[0] - closest_pt[4])**2 + (closest_pt[1] - closest_pt[5])**2)
    b_r = dis - (width_R - self.r + self.road_thr)
    c_r = self.q1_road * jnp.exp(jnp.clip(self.q2_road * b_r, 0., self.barrier_thr_road))

    # Left bound (inner edge).
    width_L = jnp.sqrt((closest_pt[0] - closest_pt[2])**2 + (closest_pt[1] - closest_pt[3])**2)
    b_l = -dis - (width_L - self.r + self.road_thr)
    c_l = self.q1_road * jnp.exp(jnp.clip(self.q2_road * b_l, 0., self.barrier_thr_road))

    return c_l + c_r

  @partial(jit, static_argnames='self')
  def vel_bound_cost_stage(self, state: ArrayImpl) -> ArrayImpl:
    """
    Calculates the velocity bound soft constraint cost.

    Args:
        state (ArrayImpl): (dim_x,) Joint system state

    Returns:
        ArrayImpl: cost (scalar)
    """
    state = self.LMx @ state
    cons_v_min = self.v_min - state[2]
    cons_v_max = state[2] - self.v_max
    barrier_v_min = self.q1_v * jnp.exp(jnp.clip(self.q2_v * cons_v_min, None, self.barrier_thr))
    barrier_v_max = self.q1_v * jnp.exp(jnp.clip(self.q2_v * cons_v_max, None, self.barrier_thr))
    return barrier_v_min + barrier_v_max

  @partial(jit, static_argnames='self')
  def proximity_cost_stage_ellipse(self, state: ArrayImpl) -> ArrayImpl:
    """
    Calculates the proximity soft constraint cost.

    NOTE: Does not quite work <- math could be wrong.

    Args:
        state (ArrayImpl): (dim_x,) Joint system state

    Returns:
        ArrayImpl: cost (scalar)
    """
    _prox_x = self.config.PROX_X
    _prox_y = self.config.PROX_Y

    # Players' positional state indices.
    ego_x_idx, ego_y_idx, _ = self.ego_idx
    ado_x_idx, ado_y_idx, ado_psi_idx = self.ado_idx

    # Computes separation (in the rotated coordinate system).
    dx = (state[ego_x_idx] - state[ado_x_idx])
    dy = (state[ego_y_idx] - state[ado_y_idx])

    rot = state[ado_psi_idx]
    dx_trans = dx * jnp.cos(rot) - dy * jnp.sin(rot)
    dy_trans = dy * jnp.cos(rot) + dx * jnp.sin(rot)

    # Evaluates the cost.
    penalty_prox = -jnp.minimum(
        dx_trans**2 / _prox_x**2 + dy_trans**2 / _prox_y**2 - 1. * self.config.PENALTY_SCALE, 0.
    )
    c_prox = self.q1_prox * jnp.exp(
        jnp.clip(self.q2_prox * penalty_prox, None, self.barrier_thr_prox)
    )

    return c_prox

  @partial(jit, static_argnames='self')
  def proximity_cost_stage(self, state: ArrayImpl) -> ArrayImpl:
    """
    Calculates the proximity soft constraint cost.
    Covers each car footprint with two circles.

    Args:
        state (ArrayImpl): (dim_x,) Joint system state

    Returns:
        ArrayImpl: cost (scalar)
    """
    circ_r = 2.0  # [m]
    circ_sep = 2. * circ_r  # [m]

    # Players' positional state indices.
    ego_x_idx, ego_y_idx, ego_psi_idx = self.ego_idx
    ado_x_idx, ado_y_idx, ado_psi_idx = self.ado_idx

    ego_x, ego_y, ego_psi = state[ego_x_idx], state[ego_y_idx], state[ego_psi_idx]
    ado_x, ado_y, ado_psi = state[ado_x_idx], state[ado_y_idx], state[ado_psi_idx]

    # Circle centers
    centers_ego = jnp.vstack((
        jnp.array((ego_x, ego_y)),
        jnp.array((ego_x + circ_sep * jnp.cos(ego_psi), ego_y + circ_sep * jnp.sin(ego_psi))),
    ))

    centers_ado = jnp.vstack((
        jnp.array((ado_x, ado_y)),
        jnp.array((ado_x + circ_sep * jnp.cos(ado_psi), ado_y + circ_sep * jnp.sin(ado_psi))),
    ))

    _scale = self.config.PENALTY_SCALE
    cost_mat = jnp.zeros((2, 2))
    for i in range(2):
      for j in range(2):
        dx = centers_ego[i, 0] - centers_ado[j, 0]
        dy = centers_ego[i, 1] - centers_ado[j, 1]
        penalty_prox = -jnp.minimum(jnp.sqrt(dx**2 + dy**2) - circ_sep*_scale, 0.)
        cost_mat = cost_mat.at[i, j].set(
            self.q1_prox
            * jnp.exp(jnp.clip(self.q2_prox * penalty_prox, None, self.barrier_thr_prox))
        )

    return jnp.max(cost_mat)
