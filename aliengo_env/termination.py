from collections import OrderedDict

import numpy as np

from utils import DummyObstacle


class Termination:
    def __init__(self, termination_conditions, obstacle, quadruped, env):
        """
        There are four types of termination conditions

        1. Going out of bounds of a generated terrain (failure)
        2. termination_conditions arg conditions (failure or timeout)
        3. Reaching the end of the terrain (timeout)
        4. Reaching the last generated footstep (timeout)

        Failures should be checked before timeouts, because if we happen
        to fail and timeout on the same step, the RL algo should consider
        it a failure.

        The __call__ method stops checking conditions and returns once
        a termination condition is found.
        """
        self.env = env
        self.quadruped = quadruped
        self.termination_conditions = self._sort_term_conds(
            termination_conditions, obstacle)
        self.obstacle = obstacle
        self.type_two_conditions = {
            'height_bounds': self.height_bounds,
            'orientation_bounds': self.orientation_bounds,
            'timeout': self.timeout
        }
        assert all(part in self.type_two_conditions.keys()
                   for part in self.termination_conditions.keys())

    def __call__(self):
        termination_dict = {}

        # check type 1 conditions
        if not isinstance(self.obstacle, DummyObstacle):
            done, reason = self.obstacle.bounds_termination()
            if done:
                termination_dict['termination_reason'] = reason
                return True, termination_dict

        # check type 2 conditions
        for condition in self.termination_conditions.keys():
            if self.type_two_conditions[
                    condition](*self.termination_conditions[condition]):
                if condition == 'timeout':
                    termination_dict['TimeLimit.truncated'] = True
                else:
                    termination_dict['termination_reason'] = condition
                return True, termination_dict

        # check type 3 conditions (timeout)
        if not isinstance(self.obstacle, DummyObstacle):
            if self.obstacle.timeout_termination():
                termination_dict['TimeLimit.truncated'] = True
                return True, termination_dict

        # check type 4 conditions (timeout)
        if hasattr(self.quadruped, 'footstep_generator'):
            if self.quadruped.footstep_generator.is_timeout():
                termination_dict['TimeLimit.truncated'] = True
                return True, termination_dict

        return False, termination_dict

    def _sort_term_conds(self, termination_conditions, obstacle):
        """Sort termination conditions and return ordered dict,
        so that the conditions are checked
        in a determinisitic order (despite how they appear in the yaml),
        but with timeout checked last. Also,removes the height_bounds
        termination condition if an actual obstacle is passed.
        """

        contains_obstacle = not isinstance(obstacle, DummyObstacle)
        contains_timeout = 'timeout' in termination_conditions.keys()

        output = OrderedDict()
        keys = list(termination_conditions)
        if contains_timeout:
            keys.remove('timeout')
        if contains_obstacle:
            keys.remove('height_bounds')
        keys.sort()

        for key in keys:
            output[key] = termination_conditions[key]
        if contains_timeout:
            output['timeout'] = termination_conditions['timeout']
        return output

    def height_bounds(self, lb, ub):
        return (self.quadruped.base_position[2] <= lb) \
            or (self.quadruped.base_position[2] >= ub)

    def orientation_bounds(self, x, y, z):
        return (abs(self.quadruped.base_euler)
                > np.array([x, y, z]) * np.pi).any()

    def timeout(self, n_seconds):
        return (self.env.eps_step_counter
                >= 240.0/self.env.action_repeat * n_seconds)
