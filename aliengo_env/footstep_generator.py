
import numpy as np
import pybullet as p


class FootstepGenerator:
    """This is a class for generating footstep targets
    and keeping track of current footstep.
    """
    def __init__(self, params, quadruped, vis=False):
        self.current_footstep = 0
        self.params = params
        self.quadruped = quadruped
        self.footsteps = None
        self.footstep_idcs = None
        self.vis = vis
        self.client = self.quadruped.client
        if self.vis:
            self.curr_step_body = None
            self.step_body_ids = []

    def generate_footsteps(self, params):
        """Stochastically generate footsteps in a straight line
        for a walking gait.
        """
        if params['gait'] != 'walk':
            raise NotImplementedError

        # each footstep is an x, y, z position
        self.footsteps = np.zeros((4, 3))
        step_len = params['step_length'] + ((np.random.random_sample() - 0.5)
                                            * params['step_length_rand'])
        width = params['step_width'] + ((np.random.random_sample() - 0.5)
                                        * params['step_width_rand'])
        length = params['base_length']
        len_offset = params['length_offset']

        # randomly chose right or left side of robot to start walking
        if np.random.random_sample() > 0.5:
            self.footstep_idcs = [2, 0, 3, 1]
            self.footsteps[0] = np.array([-length/2.0 + len_offset + step_len,
                                          -width/2.0, 0])  # RR
            self.footsteps[1] = np.array([length/2.0 + len_offset + step_len,
                                          -width/2.0, 0])  # FR
            self.footsteps[2] = np.array([-length/2.0 + len_offset + 2*step_len,
                                          width/2.0, 0])  # RL
            self.footsteps[3] = np.array([length/2.0 + len_offset + 2*step_len,
                                          width/2.0, 0])  # FL
        else:
            self.footstep_idcs = [3, 1, 2, 0]
            self.footsteps[0] = np.array([-length/2.0 + len_offset + step_len,
                                          width/2.0, 0])  # RL
            self.footsteps[1] = np.array([length/2.0 + len_offset + step_len,
                                          width/2.0, 0])  # FL
            self.footsteps[2] = np.array([-length/2.0 + len_offset + 2*step_len,
                                          -width/2.0, 0])  # RR
            self.footsteps[3] = np.array([length/2.0 + len_offset + 2*step_len,
                                          -width/2.0, 0])  # FR

        self.footsteps = np.tile(self.footsteps, (params['n_cycles'], 1))
        self.footsteps[:, 0] += (np.arange(params['n_cycles']).repeat(4)
                                 * step_len * 2)
        self.footsteps[:, :-1] += (np.random.random_sample(
            self.footsteps[:, :-1].shape) - 0.5) * params['footstep_rand']

        if self.vis:
            shape = self.client.createVisualShape(
                p.GEOM_CYLINDER,
                radius=.06,
                length=.001,
                rgbaColor=[191/255.0, 87/255.0, 0, 0.95])
            curr_step_shape = self.client.createVisualShape(
                p.GEOM_CYLINDER,
                radius=0.03,
                length=0.1,
                rgbaColor=[0, 1.0, 0, 1.0])

            self.step_body_ids = []
            for i in range(self.footsteps.shape[0]):
                id_ = self.client.createMultiBody(
                    baseVisualShapeIndex=shape,
                    basePosition=self.footsteps[i])
                self.step_body_ids.append(id_)
                self.client.addUserDebugText(str(i), self.footsteps[i])
            self.curr_step_body = self.client.createMultiBody(
                baseVisualShapeIndex=curr_step_shape,
                basePosition=self.footsteps[self.current_footstep])

    def get_current_foot_global_pos(self):
        foot = self.footstep_idcs[self.current_footstep % 4]
        pos = np.array(self.quadruped.client.getLinkState(
            self.quadruped.quadruped, self.quadruped.foot_links[foot])[0])
        pos[2] -= 0.0265
        return pos

    def get_current_footstep_distance(self):
        """Returns xyz distance of current quadruped foot location to next footstep location."""

        pos = self.get_current_foot_global_pos()
        return self.footsteps[self.current_footstep] - pos

    def velocity_towards_footstep(self):
        """Return velocity of current foot towards footstep.
        This method is called by the reward class.
        """
        foot = self.footstep_idcs[self.current_footstep % 4]
        # velocity vector
        vel = np.array(self.quadruped.client.getLinkState(
            self.quadruped.quadruped,
            self.quadruped.foot_links[foot],
            computeLinkVelocity=1)[6])
        # calculate unit vector in direction of footstep target
        pos = self.get_current_footstep_distance()
        pos_unit = pos/np.linalg.norm(pos)
        return (pos_unit * vel).sum()  # dot product

    def footstep_reached(self, distance_threshold):
        """Return 1 if the footstep has been reached
        (and increments current footstep), else 0.
        This method is called by the reward class.
        """
        dist = np.linalg.norm(self.get_current_footstep_distance())
        if dist <= distance_threshold:
            reached = 1.0
            self.current_footstep += 1
        else:
            reached = 0.0

        if reached and self.vis:
            self.client.resetBasePositionAndOrientation(
                self.curr_step_body,
                self.footsteps[self.current_footstep],
                [0, 0, 0, 1]
            )
        return reached

    def is_timeout(self):
        assert self.current_footstep <= len(self.footsteps)
        return self.current_footstep == len(self.footsteps)

    def reset(self):
        if self.vis:
            for i in range(len(self.step_body_ids)):
                self.client.removeBody(self.step_body_ids[i])
                # self.client.removeBody(self.text_ids[i])
            if self.curr_step_body is not None:
                self.client.removeBody(self.curr_step_body)
                self.client.removeAllUserDebugItems()
        self.current_footstep = 0
        self.generate_footsteps(self.params)


def main():
    from env import AliengoEnv
    import yaml
    import time
    import os

    path = os.path.join(os.path.dirname(__file__),
                        '../config/default_footstep.yaml')
    with open(path) as f:
        params = yaml.full_load(f)
    params = params['env_params']
    params['render'] = True
    params['fixed'] = False
    params['vis'] = True
    env = AliengoEnv(**params)
    env.reset()
    i = 0
    print()
    while True:
        i += 1
        time.sleep(1/240.0)
        env.client.stepSimulation()
        env.quadruped.footstep_generator.footstep_reached(
            params['reward_parts']['footstep_reached'][1])
        if i % 10 == 0:
            vel = env.quadruped.footstep_generator.velocity_towards_footstep()
            print("Velocity Towards current footstep: {:0.2f}".format(vel))


if __name__ == "__main__":
    main()
