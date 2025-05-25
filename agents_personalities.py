class AgentsPersonalities:
    def __init__(self, lr, gamma, epsilon, epsilon_decay, min_epsilon):
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        
        