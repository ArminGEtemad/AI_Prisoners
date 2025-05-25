import torch
import matplotlib.pyplot as plt

class OneChanceGame:
    def __init__(self, game_rules, personalities, device=torch.device('cuda')):
        self.game_rules = game_rules
        self.actions = game_rules.get_action()
        self.states = game_rules.get_state()
        self.state_idx = game_rules.STATE_IDX

        self.personalities = personalities

        # making table
        self.action_size = len(self.actions)
        self.state_size = len(self.states)

        self.Q_agent_1 = torch.zeros((self.state_size, self.action_size),
                                     device=device)
        self.Q_agent_2 = torch.zeros((self.state_size,self. action_size),
                                     device=device)
        
        self.prob_dist_init_state = torch.ones(self.state_size) / self.state_size

    def epsilon_greedy(self, Q, state_idx, epsilon):
        if torch.rand(1).item() < epsilon:
            # there is a possiblity that agents pick a random action to explore
            return torch.randint(self.action_size, (1,)).item()
        # if no exploration then the agents pick the best learned action
        return torch.argmax(Q[state_idx]).item()
    
    def q_update(self, s_next, Q, s, a, lr, r, gamma):
        """
        s_next --> next state
        Q --> table
        s --> state
        a --> action
        lr --> learning rate
        r --> reward
        gamma --> importance of future rewards
        """
        # this is the Bellman equation
        best_next_move = torch.max(Q[s_next])
        Q[s, a] = Q[s, a] + lr * (r + gamma * best_next_move - Q[s, a])
    
    def training(self, num_episodes=10000, track_window=100):
        loyal_list_agent_1 = []
        loyal_list_agent_2 = []
        loyal_counts_agent_1 = 0
        loyal_counts_agent_2 = 0
        for episodes in range(num_episodes):
            current_state = self.states[torch.multinomial(self.prob_dist_init_state, 1).item()] # initial state
            state_idx = self.state_idx[current_state]

            # best action index
            agent_1_action_idx = self.epsilon_greedy(self.Q_agent_1, state_idx, self.personalities.epsilon[0])
            agent_2_action_idx = self.epsilon_greedy(self.Q_agent_2, state_idx, self.personalities.epsilon[1])

            # translate index to the action
            agent_1_action = self.actions[agent_1_action_idx]
            agent_2_action = self.actions[agent_2_action_idx]

            # give their rewards (or punishments)
            agent_1_reward, agent_2_reward = self.game_rules.game_rewards(agent_1_action, agent_2_action)

            # what is the new state?
            new_state = (agent_1_action, agent_2_action)
            new_state_idx = self.state_idx[new_state]

            # update the table
            self.q_update(new_state_idx, self.Q_agent_1, state_idx, agent_1_action_idx, 
                    self.personalities.lr[0], agent_1_reward, self.personalities.gamma[0])
            self.q_update(new_state_idx, self.Q_agent_2, state_idx, agent_2_action_idx, 
                    self.personalities.lr[1], agent_2_reward, self.personalities.gamma[1])
            
            self.personalities.epsilon[0] = max(self.personalities.min_epsilon[0], 
                             self.personalities.epsilon[0] * self.personalities.epsilon_decay[0])
            self.personalities.epsilon[1] = max(self.personalities.min_epsilon[1], 
                             self.personalities.epsilon[1] * self.personalities.epsilon_decay[1])

            # Loyalty count
            loyal_counts_agent_1 += (agent_1_action == 'L')
            loyal_counts_agent_2 += (agent_2_action == 'L')

            # keep record to study
            if (episodes + 1) % track_window == 0:
                loyal_list_agent_1.append(loyal_counts_agent_1 / track_window)
                loyal_list_agent_2.append(loyal_counts_agent_2 / track_window)
                loyal_counts_agent_1 = 0
                loyal_counts_agent_2 = 0

        self.plot_loyalty(loyal_list_agent_1, loyal_list_agent_2, track_window, num_episodes)
            
    def plot_loyalty(self, agent_1, agent_2, window, episodes):
        plt.plot(agent_1, label='Agent 1 Loyalty')
        plt.plot(agent_2, label='Agent 2 Loyalty')
        plt.xlabel(f'Window ({window} episodes)')
        plt.ylabel('Loyal Moves')
        plt.legend()
        plt.title(f'Loyalty percentage Over {episodes} episodes of game')
        plt.show()