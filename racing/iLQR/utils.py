"""
Supporting functions.

Please contact the author(s) of this library if you have any questions.
Author: Haimin Hu (haiminh@princeton.edu)
"""

import yaml
import csv
import numpy as np
from .track import TrackVariableWidth


class Struct:
  """
  Struct for managing parameters.
  """

  def __init__(self, data):
    for key, value in data.items():
      setattr(self, key, value)


def load_config(file_path):
  """
  Loads the config file.

  Args:
      file_path (string): path to the parameter file.

  Returns:
      Struct: parameters.
  """
  with open(file_path) as f:
    data = yaml.safe_load(f)
  config = Struct(data)
  return config


def load_track_variable_width(track_name):
  filename = 'tracks/' + track_name + '.csv'

  x_ref = []
  y_ref = []
  z_ref = []
  v_ref = []
  x_inner = []
  y_inner = []
  x_outer = []
  y_outer = []

  with open(filename, newline='') as f:
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

  return track
