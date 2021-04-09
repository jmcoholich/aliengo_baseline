import numpy as np
import pybullet as p


# TODO see if there is a way to speed up the creation of multibodies (this is only done on initialization though)
# It takes about 0.05 seconds to create each multibody, so it really adds up.

class Steps():
    def __init__(self,
                client,
                fake_client,
                rows_per_m=2.0, # range from [1.0 to 5.0] (easy to hard) default=2.0
                terrain_height_range=0.25, # range from [0.0, 0.375] (easy to hard) default=0.25
                terrain_width=10.0, 
                terrain_length=20.0):


        self.client = client
        self.fake_client = fake_client

        # Terrain parameters, all units in meters
        assert rows_per_m > 0.0 and rows_per_m < 10.0
        self.row_width = 1.0/rows_per_m
        self.terrain_height_range = terrain_height_range # +/- half of this value to the height mean 
        self.terrain_length = terrain_length
        self.terrain_width = terrain_width
        self.terrain_height = terrain_height_range/2.0 + 0.01 # this is just the mean height of the blocks

        self.block_length_range = self.row_width/2. # the mean is set to the same as row_width. 
        self.ramp_distance = self.terrain_height * 10
        self.ray_start_height = self.terrain_height + self.terrain_height_range/2. + 1.
        self.create_initial()

    
    def reset(self):

        self.shuffle_shapes()
        height_values = np.linspace(-self.terrain_height_range/2., self.terrain_height_range/2., self.n_shapes)
        
        i = 0
        for row in range(int(self.terrain_width/self.row_width) + 1):
            total_len = 0.0
            while total_len < self.terrain_length:
                # i = np.random.randint(0, self.n_shapes)
                j = np.random.randint(0, self.n_shapes)
                if total_len < self.ramp_distance:
                    offset = self.terrain_height_range * (1 - float(total_len)/self.ramp_distance)
                else:
                    offset = 0
                pos = [total_len + self.shape_lengths[i]/2.0 + 0.5, # X
                        row * self.row_width - self.terrain_width/2. + self.row_width/2., # Y
                        height_values[j] - offset + 0.01] # Z
                self.client.resetBasePositionAndOrientation(self.ids[i], pos, [1,0,0,0])
                self.fake_client.resetBasePositionAndOrientation(self.fake_ids[i], pos, [1,0,0,0])
                total_len += self.shape_lengths[i]
                i += 1

        for k in range(i, self.n_bodies):
            self.client.resetBasePositionAndOrientation(self.ids[k], [0,0, -3], [1,0,0,0])
            self.fake_client.resetBasePositionAndOrientation(self.fake_ids[k], [0,0,-3], [1,0,0,0])


    def create_initial(self):
        '''Creates an identical steps terrain in client and fake client'''
        # x = self.client.createCollisionShape(p.GEOM_BOX, halfExtents=[1,1,1])
        # y = self.client.createVisualShape(p.GEOM_BOX, halfExtents=[1,1,1])
        # self.client.createMultiBody(x)
        # breakpoint()
        # return
        # pick a list of discrete values for heights to only generate a finite number of collision shapes
        self.n_shapes = 10
        length_lb = np.max((self.row_width - self.block_length_range/2., .05))
        length_values = np.linspace(length_lb,
                                    self.row_width + self.block_length_range/2.,
                                    self.n_shapes)      

        # generate n_shapes different block lengths                            
        shapeId = np.zeros(self.n_shapes, dtype=np.int)
        fake_shapeId = np.zeros(self.n_shapes, dtype=np.int)
        for i in range(len(length_values)):
            halfExtents=[length_values[i]/2., self.row_width/2., self.terrain_height_range/2.] 
            shapeId[i] = self.client.createCollisionShape(p.GEOM_BOX, halfExtents=halfExtents)
            fake_shapeId[i] = self.fake_client.createCollisionShape(p.GEOM_BOX, halfExtents=halfExtents)

        # create enough multibodies
        # average block length is just the row width (square blocks)
        # blocks per row * number of rows * 1.2 (as a buffer)
        self.n_bodies = int((int(self.terrain_length/self.row_width) + 1) * \
                                                                    (int(self.terrain_width/self.row_width) + 1) * 1.2)

        self.ids = np.zeros(self.n_bodies).astype(np.int32)
        self.fake_ids = np.zeros(self.n_bodies).astype(np.int32)
        self.shape_lengths = np.zeros(self.n_bodies)
        j = 0

        for i in range(self.n_bodies):
            self.ids[i] = self.client.createMultiBody(baseCollisionShapeIndex=shapeId[j%self.n_shapes])
            self.fake_ids[i] = self.fake_client.createMultiBody(baseCollisionShapeIndex=fake_shapeId[j%self.n_shapes])
            self.shape_lengths[i] = length_values[j%self.n_shapes]
            j += 1

    def shuffle_shapes(self):
        """Shuffles the order of the arrays containing the multibody Ids. 
        Applies same reordering to client and fake_client."""

        x = np.random.permutation(self.n_bodies)
        self.ids = self.ids[x]
        self.fake_ids = self.fake_ids[x]
        self.shape_lengths = self.shape_lengths[x]


    # def _is_state_terminal(self): #TODO
    #     ''' Calculates whether to end current episode due to failure based on current state. '''

    #     quadruped_done, termination_dict = self.quadruped.is_state_terminal(flipping_bounds=[np.pi/2.0]*3, 
    #                             height_ub=0.8 + self.terrain_height_range + 0.01) 
    #     timeout = (self.eps_step_counter >= self.eps_timeout) or \
    #                 (self.quadruped.base_position[0] >= self.terrain_length - 1.0)
    #     y_out_of_bounds = not (-self.terrain_width/2. + 0.25 < self.quadruped.base_position[1] < \
    #                                                                                     self.terrain_width/2. - 0.25)

    #     if timeout:
    #         termination_dict['TimeLimit.truncated'] = True
    #     elif y_out_of_bounds:
    #         # this overwrites previous termination reason in the rare case that more than one occurs at once.
    #         termination_dict['termination_reason'] = 'y_out_of_bounds'
    #     done = quadruped_done or timeout or y_out_of_bounds
        # return done, termination_dict


if __name__ == '__main__':
    '''This test open the simulation in GUI mode for viewing the generated terrain, then saves a rendered image of each
    client for visual verification that the two are identical. Then the script just keeps generating random terrains 
    for viewing. '''

    env = gym.make('gym_aliengo:AliengoSteps-v0', render=True, realTime=True)
    imwrite('client_render.png', cvtColor(env.render(client=env.client, mode='rgb_array'), COLOR_RGB2BGR))
    imwrite('fake_client_render.png', cvtColor(env.render(client=env.fake_client, mode='rgb_array'), COLOR_RGB2BGR))

    
    while True:
        env.reset()
        time.sleep(5.0)
