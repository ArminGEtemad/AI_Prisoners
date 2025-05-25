# Q-Learning

Q-learning is a model free reinforcement learning algorithm.
The agent learns the quality of taking action $a$ in state $s$, which is called $Q(s, a)$

$Q$ values are to be updated using Bellman equation:

```math
Q(s, a) \leftarrow Q(s, a) + \alpha \Big[r + \gamma\, \mathrm{max}Q(s^\prime, a^\prime)  -  Q(s, a)\Big]
```

- $s$ is the current state
- $a$ action taken
- $r$ is the reward received
- $s^\prime$ is the next state
- $lr$ is the learning rate
- $\gamma$ is the discount factor (how importat future rewards are)

If prisoners are pure bandits they do not care about the future rewards. They want immediate advantage. i.e., $\gamma=0$

In python
```python
def update_q(Q, s, a, r, s_next, alpha, gamma):
    best_next = torch.max(Q[s_next])
    Q[s, a] = (1 - alpha) * Q[s, a] + alpha * (r + gamma * best_next)
```

# Greedy Policy
The agents must exploit the best known action but should be curious enough to explore other options. In this way, the agents will not lock in on one path and explore other options too!
```python
def epsilon_greedy(Q, state_idx, epsilon):
    if torch.rand(1).item() < epsilon:
        return torch.randint(action_size, (1,)).item()
    return torch.argmax(Q[state_idx]).item()
```
