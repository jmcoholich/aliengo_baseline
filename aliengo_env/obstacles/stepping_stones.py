import numpy as np
import pybullet as p


class SteppingStones():
    def __init__(self, client, fake_client):
        self.client = client
        self.fake_client = fake_client
        
        # stepping stone parameters
        self.height = 1.0 # height of the heightfield
        self.course_length = 10.0 # total distance from edge of start block to edge of end block 
        self.course_width = 2.0 # widght of path of stepping stones 
        self.stone_length = 0.25 # side length of square stepping stones
        self.stone_density = 4.0 # stones per square meter 
        self.stone_height_range = 0.25 # heights of stones will be within [self.height - this/2, self.height + this/2 ]
        self.ray_start_height = self.height + self.stone_height_range
        self.n_stones = int(self.course_length * self.course_width * self.stone_density)
        self.create_initial()


    def generate_stone_pattern(self):
        """Randomly generate stone locations and heights.""" 
        
        stone_heights = (np.random.rand(self.n_stones) - 0.5) * self.stone_height_range + self.height/2.0 
        stone_x = np.random.rand(self.n_stones) * self.course_length + 1.0
        stone_y = (np.random.rand(self.n_stones) - 0.5) * self.course_width
        return stone_heights, stone_x, stone_y


    def create_initial(self):
        '''Creates an identical set of stepping stones in client and fake_client.'''

        stone_heights, stone_x, stone_y = self.generate_stone_pattern()

        self.client_ids = [-1] * self.n_stones
        self.fake_client_ids = [-1] * self.n_stones

        for item in [[self.client, self.client_ids], [self.fake_client, self.fake_client_ids]]:
            start_block = item[0].createCollisionShape(p.GEOM_BOX, 
                                                halfExtents=[1.0, self.course_width/2.0, self.height/2.0])
            stepping_stone = item[0].createCollisionShape(p.GEOM_BOX, 
                                                halfExtents=[self.stone_length/2.0, self.stone_length/2.0, self.height/2.0])

            start_body = item[0].createMultiBody(baseCollisionShapeIndex=start_block, 
                                            basePosition=[0,0,self.height/2.0])
            end_body = item[0].createMultiBody(baseCollisionShapeIndex=start_block, 
                                            basePosition=[self.course_length + 2.0, 0, self.height/2.],)
            
            for i in range(self.n_stones):
                item[1][i] = item[0].createMultiBody(baseCollisionShapeIndex=stepping_stone, 
                                        basePosition=[stone_x[i], stone_y[i], stone_heights[i]])


    def reset(self):
        stone_heights, stone_x, stone_y = self.generate_stone_pattern()

        for item in [[self.client, self.client_ids], [self.fake_client, self.fake_client_ids]]:
            for i in range(self.n_stones):
                item[0].resetBasePositionAndOrientation(bodyUniqueId=item[1][i], 
                                        posObj=[stone_x[i], stone_y[i], stone_heights[i]],
                                        ornObj=[1.0,0,0,0])
        
        return self.height


    # def _is_state_terminal(self): #TODO
    #     ''' Calculates whether to end current episode due to failure based on current state. '''

    #     quadruped_done, termination_dict = self.quadruped.is_state_terminal(flipping_bounds=[np.pi/2.0]*3,
    #                                                         height_lb=self.height - self.stone_height_range/2.0,
    #                                                         height_ub=self.height - self.stone_height_range/2.0 + 1.0) 
    #     timeout = (self.eps_step_counter >= self.eps_timeout) or \
    #                 (self.quadruped.base_position[0] >= self.course_length + 2.0)
    #     # the height termination condition should take care of y_out_of_bounds
    #     # y_out_of_bounds = not (-self.stairs_width/2. < self.base_position[1] < self.stairs_width/2.)
    #     if timeout:
    #         termination_dict['TimeLimit.truncated'] = True
    #     done = quadruped_done or timeout
    #     return done, termination_dict


if __name__ == '__main__':
    '''This test open the simulation in GUI mode for viewing the generated terrain, then saves a rendered image of each
    client for visual verification that the two are identical. There are two resets to ensure that the deletion and 
    addition of terrain elements is working properly. '''

    env = gym.make('gym_aliengo:AliengoSteppingStones-v0', render=True, realTime=True, env_mode='flat')
    env.reset()
    time.sleep(1e3)
    imwrite('client_render.png', cvtColor(env.render(client=env.client, mode='rgb_array'), COLOR_RGB2BGR))
    imwrite('fake_client_render.png', cvtColor(env.render(client=env.fake_client, mode='rgb_array'), COLOR_RGB2BGR))

    while True:
        env.reset()
        time.sleep(1e3)
