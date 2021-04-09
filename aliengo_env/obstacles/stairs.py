import numpy as np
import pybullet as p


class Stairs():
    def __init__(self, 
                client, 
                fake_client,
                step_height=0.25, # [0.0, 0.5?] unknown how high we can really go Default = 0.25
                step_length=2.0): # [0.25, 3.0] Default = 2.0):


        self.client = client
        self.fake_client = fake_client

        # Stairs parameters, all units in meters. Stairs height is determined by step height and stairs length
        self.stairs_length = 50
        self.stairs_width = 10
        self.step_height = step_height # this is a mean
        self.step_length = step_length # this is a mean
        self.step_height_range = self.step_height/2. # this is total range, not just range in one direction
        self.step_length_range = self.step_length/2.
        self.ray_start_height = self.stairs_length / self.step_length * self.step_height * 2

        self.create_initial()


    def reset(self):
        total_len = 0
        total_height = 0
        i = 0
        while total_len < self.stairs_length:
            height = (np.random.rand() - 0.5) * self.step_height_range + self.step_height
            length = (np.random.rand() - 0.5) * self.step_length_range + self.step_length
            pos = [total_len + length/2. + 1.0, 0.0, total_height + height/2.]
            self.client.resetBasePositionAndOrientation(self.ids[i], posObj=pos, ornObj=[1.0,0,0,0])
            self.fake_client.resetBasePositionAndOrientation(self.fake_ids[i], posObj=pos, ornObj=[1.0,0,0,0])
            total_len += length
            total_height += height
            i += 1

        for j in range(i, self.max_steps): # just hid the unused steps under the ground
            pos = [0.0, 0.0, -3.0]
            self.client.resetBasePositionAndOrientation(self.ids[j], posObj=pos, ornObj=[1.0,0,0,0])
            self.fake_client.resetBasePositionAndOrientation(self.fake_ids[j], posObj=pos, ornObj=[1.0,0,0,0])



    def create_initial(self):
        """ Create maximum number of shapes and put them in simulation, then just reposition them upon reset """

        
        # use the same collision shape for every step
        halfExtents = [(self.step_length + self.step_length_range/2. + 0.01) / 2., 
                        self.stairs_width/2., 
                        (self.step_height + self.step_height_range/2. + 0.01 )/2.]
        _id = self.client.createCollisionShape(p.GEOM_BOX, halfExtents=halfExtents)
        fake_id = self.fake_client.createCollisionShape(p.GEOM_BOX, halfExtents=halfExtents)

        self.max_steps = int(self.stairs_length/(self.step_length - self.step_length_range/2.0)) + 1 # get ceiling
        self.ids = [-1] * self.max_steps
        self.fake_ids = [-1] * self.max_steps

        for i in range(self.max_steps):
            self.ids[i] = self.client.createMultiBody(baseCollisionShapeIndex=_id)
            self.fake_ids[i] = self.fake_client.createMultiBody(baseCollisionShapeIndex=fake_id)




    # def _create_stairs(self):
        


    # def _is_state_terminal(self): #TODO
    #     ''' Calculates whether to end current episode due to failure based on current state. '''

    #     quadruped_done, termination_dict = self.quadruped.is_state_terminal(flipping_bounds=[np.pi/2.0]*3, 
    #                                                                         height_ub=np.inf) # stairs can go very high 
    #     timeout = (self.eps_step_counter >= self.eps_timeout) or \
    #                 (self.quadruped.base_position[0] >= self.stairs_length - 2.0)
    #     y_out_of_bounds = not (-self.stairs_width/2. < self.quadruped.base_position[1] < self.stairs_width/2.)
    #     if timeout:
    #         termination_dict['TimeLimit.truncated'] = True
    #     elif y_out_of_bounds:
    #         # this overwrites previous termination reason in the rare case that more than one occurs at once.
    #         termination_dict['termination_reason'] = 'y_out_of_bounds'
    #     done = quadruped_done or timeout or y_out_of_bounds
    #     return done, termination_dict


if __name__ == '__main__':
    '''This test open the simulation in GUI mode for viewing the generated terrain, then saves a rendered image of each
    client for visual verification that the two are identical. Then the script just keeps generating random terrains 
    for viewing. '''

    env = gym.make('gym_aliengo:AliengoStairs-v0', render=True, realTime=True)
    imwrite('client_render.png', cvtColor(env.render(client=env.client, mode='rgb_array'), COLOR_RGB2BGR))
    imwrite('fake_client_render.png', cvtColor(env.render(client=env.fake_client, mode='rgb_array'), COLOR_RGB2BGR))

    
    while True:
        env.reset()
        time.sleep(5.0)