"""
Multiplayer dynamical systems.

Please contact the author(s) of this library if you have any questions.
Author: Haimin Hu (haiminh@princeton.edu)
"""
from typing import List, Tuple
from abc import abstractmethod

from functools import partial
from jax import jit, jacfwd, vmap
from jaxlib.xla_extension import ArrayImpl
import jax.numpy as jnp

from iLQR import Dynamics


class MultiPlayerDynamicalSystem(object):
  """
  Base class for all multiplayer continuous-time dynamical systems. Supports
  numrical integration and linearization.
  """

  def __init__(self, x_dim, u_dims, T=None):
    """
    Initialize with number of state/control dimensions.

    Args:
        x_dim (int): number of state dimensions
        u_dims ([int]): liset of number of control dimensions for each player
        T (float): time interval
    """
    self._x_dim = x_dim
    self._u_dims = u_dims
    self._T = T
    self._num_players = len(u_dims)

  @abstractmethod
  def cont_time_dyn(self, x: ArrayImpl, u_list: list, k: int = 0) -> list:
    """
    Computes the time derivative of state for a particular state/control.

    Args:
        x (ArrayImpl): joint state (nx,)
        u_list (list of ArrayImpl): list of controls [(nu_0,), (nu_1,), ...]

    Returns:
        list of ArrayImpl: list of next states [(nx_0,), (nx_1,), ...]
    """
    raise NotImplementedError

  @partial(jit, static_argnames='self')
  def disc_time_dyn(self, x: ArrayImpl, us: ArrayImpl, k: int = 0) -> ArrayImpl:
    """
    Computes the one-step evolution of the system in discrete time with Euler integration.

    Args:
        x (ArrayImpl): joint state (nx,)
        us (ArrayImpl): agent controls (nui, N_sys)

    Returns:
        ArrayImpl: next state (nx,)
    """
    x_dot = self.cont_time_dyn(x, us, k)
    return x + self._T * x_dot

  @abstractmethod
  def linearize_discrete_jitted(
      self,
      x: ArrayImpl,
      us: ArrayImpl,
      k: int = 0,
  ) -> Tuple[ArrayImpl, ArrayImpl]:
    """
    Compute the Jacobian linearization of the dynamics for a particular state x and control us.
    Outputs A and B matrices of a discrete-time linear system.

    Args:
        x (ArrayImpl): joint state (nx,)
        us (ArrayImpl): agent controls (nui, N_sys)

    Returns:
        ArrayImpl: the Jacobian of next state w.r.t. x (nx, nx)
        ArrayImpl: the Jacobians of next state w.r.t. us (nx, nui, N_sys)
    """
    raise NotImplementedError


class ProductMultiPlayerDynamicalSystem(MultiPlayerDynamicalSystem):

  def __init__(self, subsystem_list: List[Dynamics], T: float = 0.1):
    """
    Implements a multiplayer dynamical system who's dynamics decompose into a Cartesian product of
    single-player dynamical systems.

    Initialize with a list of dynamical systems.

    NOTE:
      - Assumes that all subsystems have the same state and control dimension.

    Args:
        subsystem_list ([DynamicalSystem]): single-player dynamical system
        T (float): discretization time interval
    """
    self._N_sys = len(subsystem_list)
    self._subsystem = subsystem_list[0]
    self._subsystems = subsystem_list

    self._x_dims = [subsys.dim_x for subsys in subsystem_list]
    self._x_dim = sum(self._x_dims)
    self._u_dims = [subsys.dim_u for subsys in subsystem_list]
    self._u_dim = sum(self._u_dims)

    super(ProductMultiPlayerDynamicalSystem, self).__init__(self._x_dim, self._u_dims, T)

    self.update_lifting_matrices()

    # Pre-computes Jacobian matrices.
    self.jac_f = jit(jacfwd(self.disc_time_dyn, argnums=[0, 1]))

  def update_lifting_matrices(self):
    """
    Updates the lifting matrices.
    """
    # Creates lifting matrices LMx_i for subsystem i such that LMx_i @ x = xi.
    _split_index = jnp.hstack((0, jnp.cumsum(jnp.asarray(self._x_dims))))
    self._LMx = jnp.zeros((self._subsystem.dim_x, self._x_dim, self._N_sys))
    _id_mat = jnp.eye(self._subsystem.dim_x)
    for i in range(self._N_sys):
      self._LMx = self._LMx.at[:, _split_index[i]:_split_index[i + 1], i].set(_id_mat)

    # Creates lifting matrices LMu_i for subsystem i such that LMu_i @ u = ui.
    _split_index = jnp.hstack((0, jnp.cumsum(jnp.asarray(self._u_dims))))
    self._LMu = jnp.zeros((self._subsystem.dim_u, self._u_dim, self._N_sys))
    _id_mat = jnp.eye(self._subsystem.dim_u)
    for i in range(self._N_sys):
      self._LMu = self._LMu.at[:, _split_index[i]:_split_index[i + 1], i].set(_id_mat)

  @partial(jit, static_argnames='self')
  def split_joint_state(self, x: ArrayImpl) -> ArrayImpl:
    """
    Splits the joint state.

    Args:
        x (ArrayImpl): joint state (nx,)

    Returns:
        ArrayImpl: states (nxi, N_sys)
    """
    _split = lambda LMx, x: LMx @ x

    _split_vmap = vmap(_split, in_axes=(2, None), out_axes=(1))
    return _split_vmap(self._LMx, x)

  @partial(jit, static_argnames='self')
  def disc_time_dyn(self, x: ArrayImpl, us: ArrayImpl, k: int = 0) -> ArrayImpl:
    """
    Computes the one-step time evolution of the system.

    Args:
        x (ArrayImpl): joint state (nx,)
        us (ArrayImpl): agent controls (nui, N_sys)

    Returns:
        ArrayImpl: next joint state (nx,)
    """
    xs = self.split_joint_state(x)

    # vmap over subsystems.
    _disc_time_dyn_vmap = vmap(self._subsystem.dct_time_dyn, in_axes=(1, 1), out_axes=(1))
    xs_next = _disc_time_dyn_vmap(xs, us)

    return xs_next.flatten('F')

  @partial(jit, static_argnames='self')
  def integrate_forward(self, x: ArrayImpl, us: ArrayImpl, k: int = 0) -> ArrayImpl:
    """
    Computes the one-step time evolution of the system.

    Args:
        x (ArrayImpl): joint state (nx,)
        us (ArrayImpl): agent controls (nui, N_sys)

    Returns:
        ArrayImpl: next joint state (nx,)
        ArrayImpl: next clipped controls (nui, num_players)
    """
    xs = self.split_joint_state(x)

    # vmap over subsystems.
    _integrate_forward_vmap = vmap(
        self._subsystem.integrate_forward_jax, in_axes=(1, 1), out_axes=(1, 1)
    )
    xs_next, us_next_clipped = _integrate_forward_vmap(xs, us)

    return xs_next.flatten('F'), us_next_clipped

  @partial(jit, static_argnames='self')
  def linearize_discrete_jitted(
      self,
      x: ArrayImpl,
      us: ArrayImpl,
      k: int = 0,
  ) -> Tuple[ArrayImpl, ArrayImpl]:
    """
    Compute the Jacobian linearization of the dynamics for a particular state x and control us.
    Outputs A and B matrices of a discrete-time linear system.

    Args:
        x (ArrayImpl): joint state (nx,)
        us (ArrayImpl): agent controls (nui, N_sys)

    Returns:
        ArrayImpl: the Jacobian of next state w.r.t. x (nx, nx)
        ArrayImpl: the Jacobians of next state w.r.t. us (nx, nui, N_sys)
    """
    A_disc, Bs_disc = self.jac_f(x, us, k)
    return A_disc, Bs_disc
