# import packages 
import torch
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # gpu accelaration if possible

# define actions, possible states
# R --> ratting out your partner, L --> stay loyal to your partner 
ACTIONS = ['R', 'L']
STATES = [(a, b) for a in ACTIONS for b in ACTIONS]

STATE_IDX = {state: idx for idx, state in enumerate(STATES)}
IDX_STATE = {idx: state for state, idx in STATE_IDX.items()}

# making table
action_size = len(ACTIONS)
state_size = len(STATES)

Q_agent_1 = torch.zeros((state_size, action_size), device=device)
Q_agent_2 = torch.zeros((state_size, action_size), device=device)

# defining the game rules
def game_rewards(agent_1_action, agent_2_action):
    """
    For each round both agents need to decide
    and eigher get rewarded or punished.
    R --> rating out your partner
    L --> stay loyal to your partner 
    """

    actions = {
        ('R', 'R'): (1, 1), # being a rat costs but not as high as being loyal
        ('R', 'L'): (5, 0), # reward being a rat
        ('L', 'R'): (0, 5), # reward being a rat
        ('L', 'L'): (3, 3), # being loyal is costly
    }

    return actions[(agent_1_action, agent_2_action)]

def epsilon_greedy(Q, state_idx, epsilon):
    if torch.rand(1).item() < epsilon:
        # there is a possiblity that agents pick a random action to explore
        return torch.randint(action_size, (1,)).item()
    
    # if no exploration then the agents pick the best learned action
    return torch.argmax(Q[state_idx]).item()

def q_update(s_next, Q, s, a, lr, r, gamma):
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

prob_dist_init_state = torch.ones(state_size) / state_size # agents are as loyal as selfish

num_episodes = 10000

# agents have different personalities!
# idx 0 --> agent 1, idx 1 --> agent 2
lr = [0.1, 0.1]
gamma = [0.99, 0]
epsilon = [1.0, 1.0]
epsilon_decay = [0.9995, 0.9995]
min_epsilon = [0.01, 0.01]

window = 100
loyal_list_agent_1 = []
loyal_list_agent_2 = []
loyal_counts_agent_1 = 0
loyal_counts_agent_2 = 0

for episodes in range(num_episodes):
    current_state = STATES[torch.multinomial(prob_dist_init_state, 1).item()] # initial state
    state_idx = STATE_IDX[current_state]

    # best action index
    agent_1_action_idx = epsilon_greedy(Q_agent_1, state_idx, epsilon[0])
    agent_2_action_idx = epsilon_greedy(Q_agent_2, state_idx, epsilon[1])

    # translate index to the action
    agent_1_action = ACTIONS[agent_1_action_idx]
    agent_2_action = ACTIONS[agent_2_action_idx]

    # give their rewards (or punishments)
    agent_1_reward, agent_2_reward = game_rewards(agent_1_action, agent_2_action)

    # what is the new state?
    new_state = (agent_1_action, agent_2_action)
    new_state_idx = STATE_IDX[new_state]

    # update the table
    q_update(new_state_idx, Q_agent_1, state_idx, agent_1_action_idx, 
             lr[0], agent_1_reward, gamma[0])
    q_update(new_state_idx, Q_agent_2, state_idx, agent_2_action_idx, 
             lr[1], agent_2_reward, gamma[1])
    
    epsilon[0] = max(min_epsilon[0], epsilon[0] * epsilon_decay[0])
    epsilon[1] = max(min_epsilon[1], epsilon[1] * epsilon_decay[1])

    # Loyalty count
    loyal_counts_agent_1 += (agent_1_action == 'L')
    loyal_counts_agent_2 += (agent_2_action == 'L')

    # keep record to study
    if (episodes + 1) % window == 0:
        loyal_list_agent_1.append(loyal_counts_agent_1 / window)
        loyal_list_agent_2.append(loyal_counts_agent_2 / window)
        loyal_counts_agent_1 = 0
        loyal_counts_agent_2 = 0
    
plt.plot(loyal_list_agent_1, label='Agent 1 Loyalty')
plt.plot(loyal_list_agent_2, label='Agent 2 Loyalty')
plt.xlabel(f'Window ({window} episodes)')
plt.ylabel('Loyal Moves')
plt.legend()
plt.title(f'Loyalty percentage Over {num_episodes} episodes of game')
plt.show()