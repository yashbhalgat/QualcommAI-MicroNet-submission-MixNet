import numpy as np

class CEM(object):
    def __init__(self, num_action, num_pop, elite_fraction, bias, init):
        self.num_action = num_action
        self.num_pop = num_pop
        self.bias = bias
        self.n_elite = int(num_pop * elite_fraction)        
        self.theta_mean = np.zeros(num_action)
        self.init = np.array(init, dtype=np.float)
        self.theta_mean = np.copy(self.init)
        self.theta_mean += self.bias
        self.theta_std = np.ones(num_action)
        self.thetas = np.zeros((num_pop, self.num_action), dtype=float)
        self.rewards = np.zeros((num_pop,), dtype=float)
        
    def make_and_act_policy(self, popNum):
        self.thetas[popNum] = np.random.randn(1, self.num_action) * self.theta_std + self.theta_mean
        action = np.rint(self.thetas[popNum])
        action = np.minimum(action, self.init)
        action[action < 2] = 2
        action[action > 7] = 7 
        return action
    
    def reward(self, popNum, reward):
        self.rewards[popNum] = reward
        
    def learn(self, genNum):
        # Get elite thetas
        elite_idxs = self.rewards.argsort()[-self.n_elite:]
        elite_thetas = self.thetas[elite_idxs]
        
        # Update theta_mean, theta_std
        self.theta_mean = elite_thetas.mean(axis=0)
        self.theta_std = elite_thetas.std(axis=0)
        
        # add noise after iteration
        noise = 0.02 * max(1 - genNum/20.0, 0)
        self.theta_std += noise
        
        if (True):
            print(" Learn: {}, {:.4f}, {:.4f}, {:.3f}".format(genNum, np.mean(self.rewards), np.max(self.rewards), np.mean(self.theta_std)))
            print(self.make_and_act_policy(0).astype(int).tolist())

        return np.mean(self.rewards), np.max(self.rewards)
