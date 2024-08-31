import numpy as np
import random
from monte_carlo_tree_search import MCTS

class SimpleGame:
    def __init__(self, grid_size=8):
        self.grid_size = grid_size
        self.grid = np.full((grid_size, grid_size), '0', dtype=str)
        self.actions = 7  # Starting with 8 "taper" actions
        self.super_actions = 0  # Starting with 0 "super taper" actions
        self.points = 0
        self.super_actions_used_correctly =0

        # Placing the stone block in the center
        self.grid[grid_size // 2, grid_size // 2] = 'S'
        
        # Placing tofus diagonally around the stone block
        self.grid[grid_size // 2 - 1, grid_size // 2 - 1] = 'T'
        self.grid[grid_size // 2 - 1, grid_size // 2 + 1] = 'T'
        self.grid[grid_size // 2 + 1, grid_size // 2 - 1] = 'T'
        self.grid[grid_size // 2 + 1, grid_size // 2 + 1] = 'T'
    
    def display_grid(self):
        for row in self.grid:
            print(' '.join(row))

    def is_valid_position(self, x, y):
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def get_possible_adjacent_positions(self, x, y):
        possible_positions = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_x, new_y = x + dx, y + dy
            if self.is_valid_position(new_x, new_y) and self.grid[new_x, new_y] == '0':
                possible_positions.append((new_x, new_y))
        return possible_positions

    def hit(self, x, y):
        if not self.is_valid_position(x, y) or self.grid[x, y] not in ['S', 'T']:
            return False  # Invalid hit

        if self.grid[x, y] == 'S':
            # Hit a stone block
            self.super_actions += 1
            new_blocks = self.get_possible_adjacent_positions(x, y)
            for pos in random.sample(new_blocks, min(3, len(new_blocks))):
                self.grid[pos] = 'S'
        
        elif self.grid[x, y] == 'T':
            # Hit a tofu
            self.points += 400
            new_tofus = self.get_possible_adjacent_positions(x, y)
            for pos in random.sample(new_tofus, min(3, len(new_tofus))):
                self.grid[pos] = 'T'

        self.grid[x, y] = '0'  # Remove the hit item
        self.actions -= 1
        return True

    def super_hit(self, x, y):
        if self.super_actions <= 0:
            return False  # No super actions available
        
        positions_to_hit = [(x, y), (x - 1, y), (x, y - 1), (x + 1, y), (x, y + 1)]
        hit_count = 0

        for pos in positions_to_hit:
            if self.is_valid_position(*pos) and self.grid[pos] in ['S', 'T']:
                if(self.grid[pos] == 'S'):
                    self.points += 100
                self.grid[pos] = '0'
                hit_count += 1
        
        if hit_count == 5:
            self.actions += 2
            self.super_actions_used_correctly +=1
        
        self.super_actions -= 1
        return True

    def is_game_over(self):
        return (self.actions <= 0 and self.super_actions <= 0) or not np.any(np.isin(self.grid, ['S', 'T']))
    
    def find_children(self):
        "All possible successors of this board state"
        children = []
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.grid[x, y] in ['S', 'T']:
                    if self.actions > 0:
                        children.append(self.copy(x,y))
                    if self.super_actions > 0:
                        children.append(self.copy(x,y,"SA"))
        return children

    def find_random_child(self):
        "Random successor of this board state (for more efficient simulation)"
        children = self.find_children()
        return random.choice(children) if children else None

    def is_terminal(self):
        "Returns True if the node has no children"
        return self.is_game_over()

    def reward(self):
        SA_Reward=0
        if self.super_actions_used_correctly > 0:
            SA_Reward = self.super_actions_used_correctly * 10000
        base_reward = self.points 
        return base_reward + SA_Reward

    def copy(self,x,y,action="A"):
        new_game = SimpleGame(self.grid_size)
        new_game.grid = np.copy(self.grid)
        new_game.actions = self.actions
        new_game.super_actions = self.super_actions
        new_game.points = self.points
        new_game.super_actions_used_correctly = self.super_actions_used_correctly
        if action=="A":
            new_game.hit(x,y)
        elif action =="SA":
            new_game.super_hit(x,y)
        return new_game

def play_game():
    tree = MCTS()
    game = SimpleGame()
    round = 0
    while (game.actions>0 or game.super_actions>0) and np.any(np.isin(game.grid, ['S', 'T'])):
        round+=1
        for _ in range(2000):
            tree.do_rollout(game)
        print("Round num :",round,"\n Points  : ",game.points,' || Actions : ',game.actions," || Super Action : ",game.super_actions, " || Super Action used corretly : ",game.super_actions_used_correctly)
        game = tree.choose(game)
    print("Total points : ",game.points)

play_game() 

# Real Waven game x = 9 and y = 7 