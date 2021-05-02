import numpy as np
from noise import pnoise2
import pybullet as p


class Hills():
    def __init__(self,
                client,
                fake_client,
                scale=1.0, # good values range from 5.0 (easy) to 0.5 (hard)
                amplitude=0.75): # try [0.1, 1.0]


        # Hills parameters, all units in meters
        self.hills_height = amplitude
        self.mesh_res = 15 # int, points/meter
        self.hills_length = 50
        self.hills_width = 5
        self.ramp_distance = 1.0
        self.ray_start_height = self.hills_height + 1.0

        # Perlin Noise parameters
        self.scale = self.mesh_res * scale
        self.octaves = 1
        self.persistence = 0.0 # roughness basically (assuming octaves > 1). I'm not using this.
        self.lacunarity = 2.0
        self.base = 0 # perlin noise base, to be randomized
        self.terrain = None # to be set later
        self.fake_terrain = None # to be set later

        if self.scale == 1.0: # this causes terrain heights of all zero to be returned, for some reason
            self.scale = 1.01

        self.client = client
        self.fake_client = fake_client
        self.client.setPhysicsEngineParameter(enableFileCaching=0) # load the newly generated terrain every reset()
        self.fake_client.setPhysicsEngineParameter(enableFileCaching=0)

        self.mesh_length = self.hills_length * self.mesh_res
        self.mesh_width = self.hills_width * self.mesh_res

        self.create_initial_mesh()

    def bounds_termination(self):
        raise NotImplementedError

    def timeout_termination(self):
        raise NotImplementedError

    def create_initial_mesh(self):
        vertices = self.generate_vertices()

        meshScale = [1.0/self.mesh_res, 1.0/self.mesh_res, self.hills_height]
        heightfieldTextureScaling = self.mesh_res/2.

        self.terrain = self.client.createCollisionShape(p.GEOM_HEIGHTFIELD,
                                                    meshScale=meshScale,
                                                    heightfieldTextureScaling=heightfieldTextureScaling,
                                                    heightfieldData=vertices.flatten(),
                                                    numHeightfieldRows=self.mesh_width + 1,
                                                    numHeightfieldColumns=self.mesh_length + 1)
        self.fake_terrain = self.fake_client.createCollisionShape(p.GEOM_HEIGHTFIELD,
                                                    meshScale=meshScale,
                                                    heightfieldTextureScaling=heightfieldTextureScaling,
                                                    heightfieldData=vertices.flatten(),
                                                    numHeightfieldRows=self.mesh_width + 1,
                                                    numHeightfieldColumns=self.mesh_length + 1)


        ori = self.client.getQuaternionFromEuler([0, 0, -np.pi/2.])
        pos = [self.hills_length/2. +0.5 , 0, self.hills_height/2.]
        self.client.createMultiBody(baseCollisionShapeIndex=self.terrain, baseOrientation=ori, basePosition=pos)
        self.fake_client.createMultiBody(baseCollisionShapeIndex=self.fake_terrain, baseOrientation=ori, basePosition=pos)


    def reset(self):
        '''Creates an identical hills mesh using Perlin noise. Added to client and fake client'''

        vertices = self.generate_vertices()
        meshScale = [1.0/self.mesh_res, 1.0/self.mesh_res, self.hills_height]
        heightfieldTextureScaling = self.mesh_res/2.


        self.client.createCollisionShape(p.GEOM_HEIGHTFIELD,
                                                    meshScale=meshScale,
                                                    heightfieldTextureScaling=heightfieldTextureScaling,
                                                    heightfieldData=vertices.flatten(),
                                                    numHeightfieldRows=self.mesh_width + 1,
                                                    numHeightfieldColumns=self.mesh_length + 1,
                                                    replaceHeightfieldIndex=self.terrain)
        self.fake_client.createCollisionShape(p.GEOM_HEIGHTFIELD,
                                                    meshScale=meshScale,
                                                    heightfieldTextureScaling=heightfieldTextureScaling,
                                                    heightfieldData=vertices.flatten(),
                                                    numHeightfieldRows=self.mesh_width + 1,
                                                    numHeightfieldColumns=self.mesh_length + 1,
                                                    replaceHeightfieldIndex=self.fake_terrain)


    def generate_vertices(self):
        """Generates a new set of random mesh vertices."""



        vertices = np.zeros((self.mesh_length + 1, self.mesh_width + 1))
        self.base = np.random.randint(300)
        for i in range(self.mesh_length + 1):
            for j in range(self.mesh_width + 1):
                vertices[i, j] = pnoise2(float(i)/(self.scale),
                                            float(j)/(self.scale),
                                            octaves=self.octaves,
                                            persistence=self.persistence,
                                            lacunarity=self.lacunarity,
                                            repeatx=self.mesh_length + 1,
                                            repeaty=self.mesh_width + 1,
                                            base=self.base) # base is the seed
        # Uncomment below to visualize image of terrain map
        # from PIL import Image
        # Image.fromarray(((np.interp(vertices, (vertices.min(), vertices.max()), (0, 255.0))>128)*255).astype('uint8'), 'L').show()
        vertices = np.interp(vertices, (vertices.min(), vertices.max()), (0, 1.0))

        # ramp down n meters, so the robot can walk up onto the hills terrain
        for i in range(int(self.ramp_distance * self.mesh_res)):
            vertices[i, :] *= i/(self.ramp_distance * self.mesh_res)
        return vertices



    # def _is_state_terminal(self): #TODO figure out how to pass terrain-specific termination conditions
    #     ''' Adds condition for running out of terrain.'''

    #     quadruped_done, termination_dict = self.quadruped.is_state_terminal(flipping_bounds=[np.pi/2.0]*3,
    #                                                                         height_ub=self.hills_height + 0.8)
    #     timeout = (self.eps_step_counter >= self.eps_timeout) or \
    #                 (self.quadruped.base_position[0] >= self.hills_length - 0.5) # don't want it to fall off the end.
    #     y_out_of_bounds = not (-self.hills_width/2. < self.quadruped.base_position[1] < self.hills_width/2.)
    #     if timeout:
    #         termination_dict['TimeLimit.truncated'] = True
    #     elif y_out_of_bounds:
    #         # this overwrites previous termination reason in the rare case that more than one occurs at once.
    #         termination_dict['termination_reason'] = 'y_out_of_bounds'
    #     done = quadruped_done or timeout or y_out_of_bounds
    #     return done, termination_dict

