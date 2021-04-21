"""This is a class for generating footstep targets and keeping track of current footstep."""

import numpy as np

class FootstepTargets:
    def __init__(self, params, quadruped):
        self.current_footstep = 0
        self.params = params
        self.quadruped = quadruped
        # self.generate_footsteps(self.params)


    def generate_footsteps(self, params):
            ''' This is just a straight path on flat ground for now '''
            
            self.footsteps = np.zeros((4, 3)) # each footstep is an x and y position
            step_len = params['step_length'] + (np.random.random_sample() - 0.5) * params['step_length_rand']
            width = params['step_width'] + (np.random.random_sample() - 0.5) * params['step_width_rand']
            length = params['base_length']
            len_offset = params['length_offset']

            if np.random.random_sample() > 0.5: # start with left side or right side
                self.footstep_idcs = [2,0,3,1]
                self.footsteps[0] = np.array([-length/2.0 + len_offset + step_len, -width/2.0, 0]) # RR
                self.footsteps[1] = np.array([length/2.0  + len_offset + step_len, -width/2.0, 0]) # FR
                self.footsteps[2] = np.array([-length/2.0 + len_offset + 2 * step_len, width/2.0, 0]) # RL
                self.footsteps[3] = np.array([length/2.0  + len_offset + 2 * step_len, width/2.0, 0]) # FL
            else:
                self.footstep_idcs = [3,1,2,0]
                self.footsteps[0] = np.array([-length/2.0 + len_offset + step_len, width/2.0, 0]) # RL
                self.footsteps[1] = np.array([length/2.0  + len_offset + step_len, width/2.0, 0]) # FL
                self.footsteps[2] = np.array([-length/2.0 + len_offset + 2 * step_len, -width/2.0, 0]) # RR
                self.footsteps[3] = np.array([length/2.0  + len_offset + 2 * step_len, -width/2.0, 0]) # FR
            
            self.footsteps = np.tile(self.footsteps, (params['n_cycles'], 1))
            self.footsteps[:, 0] += np.arange(params['n_cycles']).repeat(4) * step_len * 2
            self.footsteps[:, :-1] += (np.random.random_sample(self.footsteps[:, :-1].shape) - 0.5) *params['footstep_rand'] 

            # if self.vis: #TODO
            #     self.client.resetBasePositionAndOrientation(self.foot_step_marker,
            #                                                 self.footsteps[self.current_footstep], 
            #                                                 [0, 0, 0, 1])
    

    def get_current_foot_global_pos(self):
        foot = self.footstep_idcs[self.current_footstep%4]
        pos = np.array(self.quadruped.client.getLinkState(self.quadruped.quadruped, self.quadruped.foot_links[foot])[0])
        pos[2] -= 0.0265
        return pos


    def get_current_footstep_distance(self):
        """Returns xyz distance of current quadruped foot location to next footstep location."""

        pos = self.get_current_foot_global_pos()
        return self.footsteps[self.current_footstep] - pos
    

    def velocity_towards_footstep(self, max_vel): #TODO test this
        """Velocity of current foot towards footstep."""

        foot = self.footstep_idcs[self.current_footstep%4]
        # velocity vector
        vel = np.array(self.quadruped.client.getLinkState(self.quadruped.quadruped, 
                                                            self.quadruped.foot_links[foot], 
                                                            computeLinkVelocity=1)[6])
        # position unit vector
        pos = self.get_current_footstep_distance()
        pos_unit = pos/np.linalg.norm(pos)
        #dot product
        return (pos_unit * vel).sum()


    def footstep_reached(self, distance_threshold):
        """Returns 1 if the footstep has been reached (and increments current footstep), else 0."""
        
        dist = np.linalg.norm(self.get_current_footstep_distance())
        if dist <= distance_threshold:
            reached = 1.0
            self.current_footstep += 1
        else:
            reached = 0.0

        return reached 

    def is_timeout(self):
        assert self.current_footstep <= len(self.footsteps)
        return self.current_footstep == len(self.footsteps)

    def reset(self):
        self.current_footstep = 0
        self.generate_footsteps(self.params)
        
