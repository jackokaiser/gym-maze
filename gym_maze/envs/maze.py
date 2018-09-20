import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors
from skimage.draw import circle, line
from sklearn.metrics.pairwise import euclidean_distances
import gym
from gym import spaces
from gym.utils import seeding

def discretize(state):
    return [int(state[0]), int(state[1])]


class MazeEnv(gym.Env):
    """Configurable environment for maze. """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self,
                 maze_generator,
                 pob_size=1,
                 action_type='VonNeumann',
                 obs_type='full',
                 reward_type='discrete',
                 live_display=False,
                 render_trace=False,
                 title=""):
        """Initialize the maze. DType: list"""
        # Random seed with internal gym seeding
        self.seed()

        # Maze: 0: free space, 1: wall
        self.maze_generator = maze_generator
        self.maze = np.array(self.maze_generator.get_maze())
        self.maze_size = self.maze.shape
        self.init_state, self.goal_states = self.maze_generator.sample_state()

        self.render_trace = render_trace
        self.title = title
        self.traces = []
        self.action_type = action_type
        self.obs_type = obs_type
        self.reward_type = reward_type

        # If True, show the updated display each time render is called rather
        # than storing the frames and creating an animation at the end
        self.live_display = live_display

        self.state = None
        self.scale = self.maze.shape[0] / 20.
        self.speed = 1.

        # Action space: 0: Up, 1: Down, 2: Left, 3: Right
        if self.action_type == 'VonNeumann':  # Von Neumann neighborhood
            self.num_actions = 4
            self.action_space = spaces.Discrete(self.num_actions)
            self.all_actions = list(range(self.action_space.n))
        elif action_type == 'Moore':  # Moore neighborhood
            self.num_actions = 8
            self.action_space = spaces.Discrete(self.num_actions)
            self.all_actions = list(range(self.action_space.n))
        elif action_type.startswith('Continuous'):
            self.num_actions = 2
            self.action_space = spaces.Box(low=-1., high=1., shape=(2,))
            self.all_actions = [[-1, -1], [-1, 0], [-1, 1],
                                [0, -1], [0, 1],
                                [1, -1], [1, 0], [1, 1]]
        else:
            raise TypeError('Action type must be either \'VonNeumann\' or \'Moore\' or \'Continuous\'')

        # Size of the partial observable window
        self.pob_size = pob_size

        # Observation space
        low_obs = 0  # Lowest integer in observation
        high_obs = 6  # Highest integer in observation
        if self.obs_type == 'full':
            self.observation_space = spaces.Box(low=low_obs,
                                                high=high_obs,
                                                shape=self.maze_size)
        elif self.obs_type == 'partial':
            self.observation_space = spaces.Box(low=low_obs,
                                                high=high_obs,
                                                shape=(self.pob_size*2+1, self.pob_size*2+1))
        elif self.obs_type == 'laser':
            self.observation_space = spaces.Box(low=0.,
                                                high=self.maze_size,
                                                shape=(3))
        elif self.obs_type == 'state':
            self.observation_space = spaces.Box(low=0.,
                                                high=1.,
                                                shape=(3))

        else:
            raise TypeError('Observation type must be either \'full\' or \'partial\'')


        # Colormap: order of color is, free space, wall, agent, food, poison
        self.cmap = colors.ListedColormap(['white', 'black', 'blue', 'green', 'red', 'gray'])
        self.bounds = [0, 1, 2, 3, 4, 5, 6]  # values for each color
        self.norm = colors.BoundaryNorm(self.bounds, self.cmap.N)

        self.ax_imgs = []  # For generating videos

    def step(self, action):
        old_state = self.state
        # Update current state
        self.state, hit_wall = self._next_state(self.state, action)

        # Footprint: Record agent trajectory
        self.traces.append(discretize(self.state[0:2]))

        done = False
        if self.reward_type == 'discrete':
            if self._goal_test(discretize(self.state[0:2])):  # Goal check
                reward = +10
                done = True
            elif hit_wall:  # Hit wall
                reward = -1
            else:  # Moved, small negative reward to encourage shorest path
                reward = -0.01
        if self.reward_type == 'distance':
            if self._goal_test(discretize(self.state[0:2])):  # Goal chekc
                done = True
            reward = - euclidean_distances([discretize(self.state[0:2])], [self.goal_states[0]])[0,0] / 3000.

        # Additional info
        info = {}

        return self._get_obs(), reward, done, info

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

        return [seed]

    def reset(self, random_setup=True):
        # Reset maze
        self.maze_generator.reset()
        self.maze = np.array(self.maze_generator.get_maze())
        if random_setup:
            self.init_state, self.goal_states = self.maze_generator.sample_state()

        # Set current state be initial state
        self.state = self.init_state

        # Clean the list of ax_imgs, the buffer for generating videos
        self.ax_imgs = []
        # Clean the traces of the trajectory
        self.traces = [self.init_state]

        return self._get_obs()

    def render(self, mode='human', close=False):
        if close:
            plt.close()
            return

        obs = self._get_full_obs()
        partial_obs = self._get_partial_obs(self.pob_size)

        # For rendering traces: Only for visualization, does not affect the observation data
        if self.render_trace:
            obs[list(zip(*self.traces[:-1]))] = 6

        # Create Figure for rendering
        if not hasattr(self, 'fig'):  # initialize figure and plotting axes
            self.fig, self.ax_full = plt.subplots(nrows=1, ncols=1)
            # self.fig, (self.ax_full, self.ax_partial) = plt.subplots(nrows=1, ncols=2)
            self.ax_full.set_title(self.title)
        self.ax_full.axis('off')
        # self.ax_partial.axis('off')

        self.fig.show()
        if self.live_display:
            # Only create the image the first time
            if not hasattr(self, 'ax_full_img'):
                self.ax_full_img = self.ax_full.imshow(obs, cmap=self.cmap, norm=self.norm, animated=True)
            # if not hasattr(self, 'ax_partial_img'):
            #     self.ax_partial_img = self.ax_partial.imshow(partial_obs, cmap=self.cmap, norm=self.norm, animated=True)
            # Update the image data for efficient live video
            self.ax_full_img.set_data(obs)
            # self.ax_partial_img.set_data(partial_obs)
        else:
            # Create a new image each time to allow an animation to be created
            self.ax_full_img = self.ax_full.imshow(obs, cmap=self.cmap, norm=self.norm, animated=True)
            # self.ax_partial_img = self.ax_partial.imshow(partial_obs, cmap=self.cmap, norm=self.norm, animated=True)

        plt.draw()

        if self.live_display:
            # Update the figure display immediately
            self.fig.canvas.draw()
        else:
            # Put in AxesImage buffer for video generation
            self.ax_imgs.append(self.ax_full_img)  # List of axes to update figure frame
            # self.ax_imgs.append([self.ax_full_img, self.ax_partial_img])  # List of axes to update figure frame

            self.fig.set_dpi(100)

        return self.fig

    def _goal_test(self, state):
        """Return True if current state is a goal state."""
        if type(self.goal_states[0]) == list:
            return list(state) in self.goal_states
        elif type(self.goal_states[0]) == tuple:
            return tuple(state) in self.goal_states

    def _next_state(self, state, action):
        """Return the next state from a given state by taking a given action."""

        # Transition table to define movement for each action
        if self.action_type == 'VonNeumann':
            transitions = {0: [-1, 0], 1: [+1, 0], 2: [0, -1], 3: [0, +1]}
            move = transitions[action]
            new_state = [state[0] + move[0], state[1] + move[1]]

        elif self.action_type == 'Moore':
            transitions = {0: [-1, 0], 1: [+1, 0], 2: [0, -1], 3: [0, +1],
                           4: [-1, +1], 5: [+1, +1], 6: [-1, -1], 7: [+1, -1]}
            move = transitions[action]
            new_state = [state[0] + move[0], state[1] + move[1]]

        elif self.action_type == 'Continuous':
            # action is speed/angle
            move = [
                action[0] * np.cos(self.state[2]),
                action[0] * np.sin(self.state[2]),
                action[1] * 0.1
            ]
            new_state = [state[0] + self.speed * self.scale * move[0], state[1] + self.speed * self.scale * move[1], state[2] + move[2]]

        elif self.action_type == 'Continuous-diff':
            # differential drive with approximately pushbot dimension
            gridworld_scale = self.speed * self.scale * 2000. # distance between adjacent pixels
            delta_t = 0.03
            wheel_radius = 0.02
            robot_width = 0.05
            forward = action[0] + action[1]
            angle = action[0] - action[1]
            move = [
                gridworld_scale * delta_t * wheel_radius * 0.5 * forward * np.cos(self.state[2]),
                gridworld_scale * delta_t * wheel_radius * 0.5 * forward * np.sin(self.state[2]),
                delta_t * (wheel_radius / robot_width) * angle
            ]
            new_state = [state[0] + move[0], state[1] + move[1], state[2] + move[2]]
        elif self.action_type == 'Continuous-vector':
            # we disregard the heading, just apply the translation vector
            move = action
            new_state = [state[0] + self.speed * self.scale * move[0], state[1] + self.speed * self.scale * move[1], state[2]]

        if (not 0 <= int(new_state[0]) < self.maze.shape[0]) or (not 0 <= int(new_state[1]) < self.maze.shape[1]) or (self.maze[int(new_state[0])][int(new_state[1])] == 1):  # Hit wall, stay there
            if len(state) > 2:
                # apply rotation still
                state[2] = new_state[2]
            return (state, True)
        else:
            return (new_state, False)

    def _get_obs(self):
        if self.obs_type == 'full':
            return self._get_full_obs()
        elif self.obs_type == 'partial':
            return self._get_partial_obs(self.pob_size)
        elif self.obs_type == 'laser':
            return self._get_laser_obs()
        elif self.obs_type == 'state':
            return self._get_state_obs()

    def _get_full_obs(self):
        """Return a 2D array representation of maze."""
        obs = np.array(self.maze)
        # Set goal positions
        for goal in self.goal_states:
            obs[goal[0]][goal[1]] = 3  # 3: goal

        # Set current position
        # Come after painting goal positions, avoid invisible within multi-goal regions
        pos = np.array(self.state[0:2], dtype=int)
        obs[circle(pos[0], pos[1], radius=self.scale, shape=obs.shape)] = 2

        if len(self.state) > 2 and self.action_type != 'Continuous-vector':
            forward = [
                int(self.state[0] + self.scale * np.cos(self.state[2])),
                int(self.state[1] + self.scale * np.sin(self.state[2]))
            ]
            # forward_pixels = np.array(line(pos[0],pos[1],forward[0],forward[1]))
            # filter out pixels out of range
            # forward_pixels = np.array(list(filter(
            #     lambda (x,y): 0<=x<self.maze.shape[0] and 0<=y<self.maze.shape[1],
            #     forward_pixels.T
            # ))).T
            # obs[tuple(forward_pixels)] = 4
            forward_pixels = circle(forward[0], forward[1], radius=self.scale / 1.5, shape=obs.shape )
            obs[forward_pixels] = 4

        return obs

    def _get_state_obs(self):
        norm_state = self.state / np.array(self.maze.shape + (1.,))
        return norm_state

    def _get_laser_obs(self, maze):
        # intersect 3 rays with closest wall
        rays_angles = self.state[2] + np.array([-0.2, 0, 0.2])
        rays_forward = [
            [[int(self.state[0] + 6. * np.cos(a))],
             [int(self.state[1] + 6. * np.sin(a))]]
            for a in rays_angles
        ]

        return [0.], laser_maze

    def _get_partial_obs(self, size=1):
        """Get partial observable window according to Moore neighborhood"""
        # Get maze with indicated location of current position and goal positions
        maze = self._get_full_obs()
        pos = np.array(self.state[0:2]).astype(int)

        under_offset = np.min(pos - size)
        over_offset = np.min(len(maze) - (pos + size + 1))
        offset = np.min([under_offset, over_offset])

        if offset < 0:  # Need padding
            maze = np.pad(maze, np.abs(offset), 'constant', constant_values=1)
            pos += np.abs(offset)

        return maze[pos[0]-size : pos[0]+size+1, pos[1]-size : pos[1]+size+1]

    def _get_video(self, interval=200, gif_path=None):
        if self.live_display:
            # TODO: Find a way to create animations without slowing down the live display
            print('Warning: Generating an Animation when live_display=True not yet supported.')
        anim = animation.ArtistAnimation(self.fig, self.ax_imgs, interval=interval)

        if gif_path is not None:
            anim.save(gif_path, writer='imagemagick', fps=10)
        return anim


class SparseMazeEnv(MazeEnv):
    def step(self, action):
        obs, reward, done, info = super()._step(action)

        # Indicator reward function
        if reward != 1:
            reward = 0

        return obs, reward, done, info


##########################################
# TODO: Make Partial observable envs as OOP-style
###########################################
