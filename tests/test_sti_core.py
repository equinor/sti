import pytest
import numpy as np

from sti.sti_core import get_stand_translation_vector, translate_problem, get_stand_rotation_matrix

from random import random


def get_random_state():
    n = (-5+10*random()) * 1000
    e = (-5+10*random()) * 1000
    t = (-5+10*random()) * 1000
    
    inc = np.pi * random()
    azi = 2*np.pi * random()

    return np.array([n, e, t, inc, azi])


class TestStandardization():

    def test_get_stand_translation_vector(self):
        for i in range(1, 100):
            state = get_random_state()

            vec = get_stand_translation_vector(state)

            assert state[0] + vec[0] == 0.0
            assert state[1] + vec[1] == 0.0
            assert state[2] + vec[2] == 0.0
    

    def test_translate_problem(self):
        for i in range(1, 100):
            start_state = get_random_state()
            target_state = get_random_state()

            new_start, new_target = translate_problem(start_state, target_state)

            assert new_start[0] == 0.0
            assert new_start[1] == 0.0
            assert new_start[2] == 0.0
            assert new_start[3] == start_state[3]
            assert new_start[4] == start_state[4]

            assert new_target[0] != target_state[0]
            assert new_target[1] != target_state[1]
            assert new_target[2] != target_state[2]
            assert new_target[3] == target_state[3]
            assert new_target[4] == target_state[4]


    def test_get_stand_rotation_matrix(self):
        for i in range(1, 100):
            start_state = get_random_state()
            target_state = get_random_state()

            A = get_stand_rotation_matrix(start_state, target_state)

            # Definition of orthogonal matrix
            I = np.dot(A, A.T)

            assert sum(sum(abs(I - np.identity(3)))) == pytest.approx(0.0)