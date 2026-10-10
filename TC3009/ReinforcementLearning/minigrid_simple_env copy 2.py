from __future__ import annotations
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall, Lava, Floor
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
import matplotlib.pyplot as plt
from minigrid.wrappers import RGBImgObsWrapper,RGBImgPartialObsWrapper

class SimpleEnv(MiniGridEnv):
    def __init__(
        self,
        size=19,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.size = size
        self.key_positions = []
        self.lava_positions = []

        self.start_agent_pos=(1,1)
        
        mission_space = MissionSpace(mission_func=self._gen_mission)
        
        if max_steps is None:
            max_steps = 4 * size**2
        
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "Reach the goal"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        
        # Place walls in straight lines
        # Vertical walls
        for y in range(1, height-1):
            self.put_obj(Wall(), width // 2, y)
        
        # Horizontal walls
        for x in range(1, width-1):
            self.put_obj(Wall(), x, height//2)
        
        # Create openings in the walls
        openings = [(width//2,5),(width//2,15),(5,height//2),(15,height//2),]
        
        for x, y in openings:
            self.grid.set(x, y, None)

        
        # Place a goal square in the bottom-right corner
        self.goal_pos = (width - 2, height - 2)
        self.put_obj(Goal(), *self.goal_pos)
        
        self._place_agent()
        
        self.mission = "Reach the goal"

    def _place_agent(self):
        print("Executing place agent")
        # self.agent_pos=(5,1)
        # self.agent_dir=0

        # return
        while True:
            print("placing agent")
            x = random.randint(1, self.size - 2)
            y = random.randint(1, self.size - 2)
            # x=17
            # y=2
            pos = (x, y)

            if(x < self.size//2 + 1 or y < self.size//2 + 1):
                continue
            
            print(pos)
            # Check if the position is empty (not wall, lava, floor, or goal)
            if (self.grid.get(*pos) is None and
                pos != self.goal_pos):
                self.agent_pos = pos
                self.agent_dir = random.randint(0, 3)  # Random direction
                break

    def reset(self, **kwargs):
        print("resetting")
        self.stepped_floors = set()
        obs = super().reset(**kwargs)
        # self._place_agent()  # Place the agent in a new random position
        return obs

    def step(self, action):
        prev_pos=self.agent_pos
        prev_dir=self.agent_dir
        obs, reward, terminated, truncated, info = super().step(action)

        if(self.grid.get(*self.agent_pos) is None):
            # print("this is normal floor. i.e., None")
            reward=-0.2
        
        if(self.agent_pos[0]>17//2 ):
            reward = -0.1

        if(self.agent_pos[1]>17//2 ):
            reward = -0.1

        if(self.agent_pos[0]>17//2 and self.agent_pos[1]>17//2):
            reward = 0.1

        if(prev_dir==self.agent_dir and prev_pos ==  self.agent_pos):
            print("BUMP!!!")
            reward=-0.3
        
        if isinstance(self.grid.get(*self.agent_pos), Goal):
            reward = 10
            terminated = True
        self.agent_pos=(1,1)
        return obs, reward, terminated, truncated, info