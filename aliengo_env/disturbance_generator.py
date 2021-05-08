import numpy as np

class DisturbanceGenerator:
    def __init__(
            self,
            quadruped,
            avg_seconds_between_perturb=10.0,
            prob_foot_disturbance=0.5,
            max_impulse,
            max_angular_impulse,
            max_foot_inpulse,
            max_perturb_length
    ):
        self.quadruped = quadruped
        # This object is called for every stepSimulation(), so action repeat
        # has no effect.
        self.p = 1.0 / (avg_seconds_between_perturb * 240.0)
        self.max_impulse = max_impulse
        self.max_angular_impulse = max_angular_impulse
        self.max_foot_inpulse = max_foot_inpulse
        self.max_perturb_length = max_perturb_length
        self.foot_p = prob_foot_disturbance
        self.applying_disturbance = False

    def __call__(self):
        """Randomly choose whether to apply a disturbance, or continue
        to apply an ongoing disturbance. """

        if self.applying_disturbance:
            # continue applying disturbance or stop if its time to stop
            pass
        elif np.random.rand() < self.p:
            pass
            # Generate and apply a new disturbance


        if self.apply_perturb and (np.random.rand() < self.perturb_p):
            # TODO eventually make disturbance generating function that
            #  applies disturbances for multiple timesteps
            if np.random.rand() > 0.5:
                force, foot = self.quadruped.apply_foot_disturbance()
                return (force, foot)
            else:
                wrench = self.quadruped.apply_torso_disturbance()
                return wrench




    def apply_torso_disturbance(
            self, wrench=None, max_force_mag=5000 * 0, max_torque_mag=500 * 0):
        """Applies a given wrench to robot torso, or defaults to a random wrench. Only lasts for one timestep.
        Returns the wrench that was applied.

        NOTE: This function doesn't work properly when p.setRealTimeSimulation(True).
        """

        if wrench is None:
            max_force_component = (max_force_mag * max_force_mag/3.0)**0.5
            max_torque_component = (max_torque_mag * max_torque_mag/3.0)**0.5
            rand_force = (np.random.random_sample(3) - 0.5) * max_force_component * 2 # U[-max_force_component, +max...]
            rand_torque = (np.random.random_sample(3) - 0.5) * max_torque_component * 2
            wrench = np.concatenate((rand_force, rand_torque))
        self.client.applyExternalForce(self.quadruped, -1, wrench[:3], [0, 0, 0], p.LINK_FRAME)
        self.client.applyExternalTorque(self.quadruped, -1, wrench[3:], p.LINK_FRAME)
        self.last_torso_disturbance = wrench
        return wrench

    def apply_foot_disturbance(self, force=None, foot=None, max_force_mag=2500 * 0):
        '''Applies a given force to a given foot, or defaults to random force applied to random foot. Only lasts for
        one timestep. Returns force and foot applied to.

        NOTE: This function doesn't work properly when p.setRealTimeSimulation(True).
        '''

        if force is None:
            max_force_component = (max_force_mag * max_force_mag/3.0)**0.5
            force = (np.random.random_sample(3) - 0.5) * max_force_component * 2
        if foot is None:
            foot = np.random.randint(0, 4)
        self.client.applyExternalForce(self.quadruped, self.foot_links[foot], force, (0,0,0), p.LINK_FRAME)
        self.last_foot_disturbance = np.concatenate((force, np.array([foot])))
        return force, foot