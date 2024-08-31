import gym
from gym import spaces

class SimpleBoardGameEnv(gym.Env):
    def __init__(self):
        super(SimpleBoardGameEnv, self).__init__()
        self.observation_space = spaces.Discrete(10)  # Plateau de 10 cases
        self.action_space = spaces.Discrete(2)  # 0: Reculer, 1: Avancer
        self.state = 0
        self.reward=1

    def step(self, action):
        self.reward = 1
        if action == 1:  # Avancer
            self.state = min(self.state + 1, 9)
        else:  # Reculer
            self.reward -= 0.1
            self.state = max(self.state - 1, 0)
        
        # Récompense et Terminaison
        if self.state == 9:
            done = True
        else:
            done = False
        
        return self.state, self.reward, done, {}

    def reset(self):
        self.state = 0
        return self.state

    def render(self, mode='human'):
        print(f"Position actuelle: {self.state}")

# # Initialisation de l'environnement
# env = SimpleBoardGameEnv()

# env.reset()        # Réinitialiser l'environnement
# env.render()       # Afficher l'état initial

# for _ in range(10):
#     action = env.action_space.sample()  # Choisir une action aléatoire
#     state, reward, done, _ = env.step(action)  # Prendre l'action
#     env.render()   # Afficher l'état après l'action
#     if done:
#         break 
