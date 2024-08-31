import math
import random
import numpy as np

id = 0
class MCTSNode:
    def __init__(self, state,env,id, parent=None):
        self.id = id
        self.env = env
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

    def is_fully_expanded(self):
        return len(self.children) == self.env.action_space.n

#score enfant
    def best_child(self, c_param=1.4):
        choices_weights = [
            (child.value / child.visits) + c_param * math.sqrt((2 * math.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self):
        return random.choice(range(self.env.action_space.n))

    def expand(self):
        global id
        id=id+1
        untried_actions = [i for i in range(self.env.action_space.n) if not any(child.state == i for child in self.children)]
        action = random.choice(untried_actions)
        next_state, _, _, _ = self.env.step(action)
        child_node = MCTSNode(next_state, parent=self,env=self.env,id=id)
        self.children.append(child_node)
        return child_node

    def backpropagate(self, reward):
        self.visits += 1
        self.value += reward
        if self.parent:
            self.parent.backpropagate(reward)

def mcts(env, iterations=1000):
    """
    Execute a Monte-Carlo Tree Search (MCTS) algorithm on a given environment.

    Parameters
    ----------
    env : gym.Env
        The environment to execute the algorithm on.
    iterations : int
        The number of iterations to run the algorithm for.

    Returns
    -------
    best_child : MCTSNode
        The best child node of the root node, which is the best action to take in the current state of the environment.
    """
    root = MCTSNode(state=env.reset(),env=env,id=id)
    
    for _ in range(iterations):
        node = root
        env.reset()
        state = env.state
        
        # Sélection
        while node.is_fully_expanded() and node.children:
            node = node.best_child()
            state, _, _, _ = env.step(node.state)
        
        # Expansion
        if not node.is_fully_expanded():
            node = node.expand()
            state, _, _, _ = env.step(node.state)
        
        # Simulation
        while True:
            action = node.rollout_policy()
            state, reward, done, _ = env.step(action)
            if done:
                 break
        
        # Rétropropagation
        node.backpropagate(reward)
    
    return root.best_child(c_param=0),root

