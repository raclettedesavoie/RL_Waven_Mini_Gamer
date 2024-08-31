from MCTSNode import mcts
import test.SimpleBoardGameEnv as SimpleBoardGameEnv
from drawNodes import draw_mcts_tree

env = SimpleBoardGameEnv.SimpleBoardGameEnv()

env.reset()
env.render()

best_node,root = mcts(env, iterations=1000)
env.step(best_node.state)
env.render()

draw_mcts_tree(root)
print(f"Meilleure action: {best_node.state}")