from __future__ import print_function
import gym
from gym import spaces
import numpy as np

# for displaying the rendering
try:
    # for Python2
    import Tkinter as tk
except ImportError:
    # for Python3
    import tkinter as tk


from PIL import ImageTk

from home_platform.env import BasicEnvironment, Observation
from home_platform.suncg import data_dir, get_available_houses


class HomeEnv(gym.Env):
    """HoME basic starter environment
    TODO: explain env basics, reward, other important details

    Actions: (can do both move & look at the same time)

        1) Moving: Discrete 5 - NOOP[0],
                                UP[1], # forward
                                RIGHT[2], # strafe right
                                DOWN[3], # backward
                                LEFT[4] # strafe left
            - params: min: 0, max: 4
        2) Looking: Discrete 5 -NOOP[0],
                                UP[1],
                                RIGHT[2],
                                DOWN[3],
                                LEFT[4]
            - params: min: 0, max: 4

    """

    def __init__(self, turning_speed=0.1, moving_speed=100):
        self.turning_speed = turning_speed  # how fast is the camera turning
        self.moving_speed = moving_speed  # how fast is the agent moving
        # empty init for tests
        self.observation = Observation(None, None, np.zeros((100,100,3), dtype=np.uint8), None).as_dict()

        self.metadata = {'render.modes': ['human', 'rgb_array']}

        # Gym environments have no parameters, so we have to
        # make sure the user first creates a symlink
        # to their SUNCG dataset in ~/.suncg
        # so that there are folders ~/.suncg/[room|house|...]
        self.data_path = data_dir()
        print("DEBUG: SUNCG DATA DIRECTORY:", self.data_path)

        self.action_space = spaces.MultiDiscrete(5, 5)
        self.observation_space = spaces.Dict({
            # TODO what are the actual bounds of all possible houses?
            # position is x, y, z
            "position": spaces.Box(low=-100, high=100, shape=(3,), dtype='float32'),

            # TODO get actual box for HPR
            # orientation is HPR / heading, pitch, roll
            "orientation": spaces.Box(low=-100, high=100, shape=(3,), dtype='float32'),

            "image": spaces.Box(low=0, high=255, shape=(500, 500, 3)),

            # collision [0] - no collision, [1] - you bumped into sthg
            "collision": spaces.Discrete(2)
        })

        print("DEBUG: LOADING HOUSES... ")
        self.list_of_houses = get_available_houses()
        print("DEBUG: FOUND HOUSES: ", len(self.list_of_houses))

        # for determinism we have to load
        # the houses in specific order
        self.next_house = 0

        self.render_window = None  # if human rendering is on

        self._reset()

    def _seed(self, seed=0):
        """ Force loading a specific house

        :param seed: integer ID for house (not house ID)
        :return:
        """
        assert seed < len(self.list_of_houses)

        self.next_house = seed

        return [self.next_house]

    def executeLooking(self, action):
        new_orientation = self.observation["orientation"]

        if action == 0:
            pass  # NOOP
        elif action == 1:
            new_orientation[0] -= self.turning_speed
        elif action == 2:
            new_orientation[2] += self.turning_speed
        elif action == 3:
            new_orientation[0] += self.turning_speed
        elif action == 4:
            new_orientation[2] -= self.turning_speed
        else:
            raise Exception("Received unknown 'looking' action: {}. "
                            "Try an integer in the range between and including 0-4.".format(action))

        self.env.setAgentOrientation(new_orientation)

    def executeMoving(self, action):
        new_impulse = [0.0, 0.0, 0.0]  # impulse = force * time

        if action == 0:
            pass  # NOOP
        elif action == 1:
            new_impulse[1] += self.moving_speed
        elif action == 2:
            new_impulse[0] += self.moving_speed
        elif action == 3:
            new_impulse[1] -= self.moving_speed
        elif action == 4:
            new_impulse[0] -= self.moving_speed
        else:
            raise Exception("Received unknown 'moving' action: {}. "
                            "Try an integer in the range between and including 0-4.".format(action))

        print("new impulse:", new_impulse)
        self.env.applyImpulseToAgent(new_impulse)

    def _step(self, action):
        assert self.action_space.contains(action)

        self.executeMoving(action[0])
        self.executeLooking(action[1])

        self.env.step()

        self.observation = self.env.getObservation().as_dict()

        reward = 0
        done = False
        misc = None

        return self.observation, reward, done, misc

    def _reset(self):
        # TODO maybe in the future we can unload the old house/rooms/objects
        # so we don't have to recreate the Panda3dBulletPhysicWorld
        # and Panda3dRenderWorld on every reset

        # load a new house into the world
        houseId = self.list_of_houses[self.next_house]
        self.env = BasicEnvironment(houseId, suncgDatasetRoot=self.data_path)

        # TODO move agent to random pos and orientation
        self.env.setAgentPosition((42, -39, 1))
        self.env.setAgentOrientation((0.0, 0.0, 0.0))

        self.env.step()  # need a single step to create rendering in RAM
        self.observation = self.env.getObservation().as_dict()

        self.next_house += 1

        # if we went to the end of the house list, loop around
        if (self.next_house == len(self.list_of_houses)):
            self.next_house = 0

        return self.observation

    def _create_window(self):
        self.render_window = tk.Tk()
        image = np.zeros(self.env.size)
        img = ImageTk.Image.fromarray(image)
        imgTk = ImageTk.PhotoImage(img)

        self.render_panel = tk.Label(self.render_window, image=imgTk)
        self.render_panel.pack(side="bottom", fill="both", expand="yes")

    def _update_human_render(self):
        img = ImageTk.Image.fromarray(self.observation["image"])
        imgTk = ImageTk.PhotoImage(img)
        self.render_panel.configure(image=imgTk)
        self.render_panel.image = imgTk
        self.render_window.update()

    def _render(self, mode='human', close=False):
        """
        "human" render creates a Tk window, returns None
        "rgb_array" creates NO window,
                    returns ndarray containing the img
        """

        if mode == "rgb_array":
            return self.observation["image"]
        else:
            if self.render_window is None:
                self._create_window()

            self._update_human_render()

    def _close(self):
        if self.render_window is not None:
            self.render_window.destroy()
