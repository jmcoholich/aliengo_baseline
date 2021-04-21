import numpy as np
from collections import OrderedDict
from utils import DummyObstacle

class Termination():
    def __init__(self, termination_conditions, obstacle, quadruped, env):
        """
        There are two types of termination conditions
        1. Ones passed through termination_conditions arg (included in self.all_conditions)
        2. Going out of bounds of a generated terrain
        3. Termination based on reaching the end of the terrain 
        4. Termination based on reaching the last generated footstep
        """
        self.env = env
        self.quadruped = quadruped
        self.termination_conditions = self._sort_term_conds(termination_conditions, obstacle)
        self.obstacle = obstacle
        self.type_one_conditions = {'height_bounds': self.height_bounds,
                                    'orientation_bounds': self.orientation_bounds,
                                    'timeout': self.timeout}
        assert all(part in self.type_one_conditions.keys() for part in self.termination_conditions.keys())


    def __call__(self):

        termination_dict = {}
        # check type 2 conditions
        if not isinstance(self.obstacle, DummyObstacle):
            done, reason = self.obstacle.bounds_termination():
            if done:
                termination_dict['termination_reason'] = reason
                return True, termination_dict

        # check type 1 conditions
        for condition in self.termination_conditions.keys():
            if self.type_one_conditions[condition](*self.termination_conditions[condition]):
                is_terminal = True
                if condition == 'timeout':
                    termination_dict['TimeLimit.truncated'] = True
                else:
                    termination_dict['termination_reason'] = condition
                return True, termination_dict # stop checking conditions once something is terminal

        # check type 3 conditions (these are all considered timeouts)
        if not isinstance(self.obstacle, DummyObstacle):
            if self.obstacle.timeout_termination():
                termination_dict['TimeLimit.truncated'] = True
                return True, termination_dict

        # check type 4 conditions (timeout)
        if self.quadruped.footstep_param is not None:
            if self.quadruped.footstep_generator.is_timeout():
                termination_dict['TimeLimit.truncated'] = True
                return True, termination_dict

        return False, termination_dict
    

    def _sort_term_conds(self, termination_conditions, obstacle):
        """ Sort termination conditions then add them to self.termination_conditions, so that the conditions are checked
        in a determinisitic order (despite how they appear in the yaml), but with timeout checked last. 
        
        Also,removes the height_bounds termination condition IF an actual obstacle is passed.
        """

        contains_obstacle = not isinstance(obstacle, DummyObstacle)

        output = OrderedDict()
        contains_timeout = 'timeout' in termination_conditions.keys()
        keys = list(termination_conditions)
        if contains_timeout: keys.pop('timeout')
        if contains_obstacle: keys.pop('height_bounds')
        keys.sort()
        for key in keys:
            output[key] = termination_conditions[key]
        if contains_timeout: output['timeout'] = termination_conditions['timeout']
        return output
    
    def height_bounds(self, lb, ub):
        return self.quadruped.base_position <= lb or self.quadruped.base_position >= lb

    def orientation_bounds(self, x, y, z):
        euler_angles = np.array(p.getEulerFromQuaternion(self.quadruped.base_orientation))
        return (abs(euler_angles) > [x, y, z]).any()

    def timeout(self, n_seconds):
        return self.env.eps_step_counter >= 240.0/self.env.n_hold_frames * n_seconds