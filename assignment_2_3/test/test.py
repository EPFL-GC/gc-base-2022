import time
import pytest
import json
import sys
import igl
import numpy as np
sys.path.append('../')
sys.path.append('../src')


import utils, laplacian_utils, mean_curvature_flow, remesher_helper
from utils import parse_input_mesh, normalize_area, get_diverging_colors, remesh
from mean_curvature_flow import MCF

import copy
import numpy.linalg as la

epsilon1 = 5e-2
epsilon2 = 1e-3

eps = 1E-6

with open('test_data.json', 'r') as infile:
    homework_data = json.load(infile)

v = np.array(homework_data[0], dtype=float)
f = np.array(homework_data[1], dtype=int)
num_bdry_vx, num_intr_vx = int(homework_data[2]), int(homework_data[3])
curr_mcf = MCF(num_bdry_vx, num_intr_vx)
curr_mcf.bbox_diagonal = igl.bounding_box_diagonal(v)

@pytest.mark.timeout(0.5)
def test_average_mean_curvature():
    data = homework_data[4]
    laplace_v = copy.deepcopy(v)
    curr_mcf.update_system(laplace_v, f)
    assert np.linalg.norm(curr_mcf.average_mean_curvature - data) < eps

@pytest.mark.timeout(0.5)
def test_laplace_solution():
    data = homework_data[5]
    laplace_v = copy.deepcopy(v)
    curr_mcf.solve_laplace_equation(laplace_v, f)
    assert np.linalg.norm(laplace_v - np.array(data, dtype=float)) < eps

@pytest.mark.timeout(0.5)
def test_meet_sc():
    data = homework_data[6]
    student_sc = [curr_mcf.meet_stopping_criteria([0.001, 0.001], epsilon1, epsilon2), curr_mcf.meet_stopping_criteria([1, 0.5], epsilon1, epsilon2)]

    assert student_sc == data

@pytest.mark.timeout(0.5)
def test_mean_curvature_flow():
    data = np.array(homework_data[7], dtype=float)
    _, student_mean_curvature_flow = curr_mcf.run_mean_curvature_flow(v, f, 100, epsilon1, epsilon2)

    assert np.linalg.norm(student_mean_curvature_flow - data) < eps