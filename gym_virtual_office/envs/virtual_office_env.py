from gym_minigrid.minigrid import *

'''
TODO:
- Ensure the goals are invisible to the agent.
- Fix observations.
'''


class VirtualOfficeEnv(MiniGridEnv):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 10
    }

    class Actions(IntEnum):
        north = 0
        east = 1
        south = 2
        west = 3

    def __init__(self, grid_size=9, max_steps=100, see_through_walls=False, agent_view_size=3):
        super().__init__(grid_size=grid_size, max_steps=max_steps, see_through_walls=see_through_walls, agent_view_size=agent_view_size)

        # Re-define actions:
        self.actions = VirtualOfficeEnv.Actions
        self.action_space = spaces.Discrete(len(self.actions))

        # Define reward range:
        self.reward_range = (0, 1)

    def _gen_grid(self, width, height):
        # Create grid:
        self.grid = Grid(width, height)
        half_width = width // 2
        half_height = height // 2

        # Colour the hallway:
        for col in range(0, half_width):
            for row in range(0, height-1):
                self.grid.set(col, row, Floor(color='blue'))

        # Colour the rooms:
        for col in range(half_width, width-1):
            for row in range(0, height-1):
                self.grid.set(col, row, Floor(color='green'))

        # Set the hidden goal states:
        self.minor_goal_location = np.array([width-2, height-2])
        self.major_goal_location = np.array([width-2, 1])

        # Create surrounding walls:
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        # Create centre walls:
        self.grid.vert_wall(half_width, 0, 2)
        self.grid.vert_wall(half_width, 3, 3)
        self.grid.vert_wall(half_width, 7, 2)

        # Create the wall separating the rooms:
        self.grid.horz_wall(half_width, half_height)

        # Set the start state:
        self.agent_pos = (1, half_height)
        self.agent_dir = 0  # Pointing east.

        # Include a mission string:
        self.mission = 'Reach one of the hidden goals.'

    def step(self, action):
        self.step_count += 1
        done = self.step_count >= self.max_steps

        # Resolve the action:
        if action == self.actions.east:
            self.agent_dir = 0
        elif action == self.actions.south:
            self.agent_dir = 1
        elif action == self.actions.west:
            self.agent_dir = 2
        elif action == self.actions.north:
            self.agent_dir = 3
        else:
            assert False, "unknown action"

        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)
        if fwd_cell is None or fwd_cell.can_overlap():
            self.agent_pos = fwd_pos

        if (self.agent_pos == self.major_goal_location).all():
            reward = 1.
            done = True
        elif (self.agent_pos == self.minor_goal_location).all():
            reward = .5
            done = True
        else:
            reward = .0

        obs = self.gen_obs()
        return obs, reward, done, {}
