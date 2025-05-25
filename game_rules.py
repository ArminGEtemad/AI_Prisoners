class GameRules:
    def __init__(self):
        # define actions, possible states
        # R --> ratting out your partner, L --> stay loyal to your partner 
        self.ACTIONS = ['R', 'L']
        self.STATES = [(a, b) for a in self.ACTIONS for b in self.ACTIONS]

        self.STATE_IDX = {state: idx for idx, state in enumerate(self.STATES)}
        self.IDX_STATE = {idx: state for state, idx in self.STATE_IDX.items()}

    # defining the game rules
    def game_rewards(self, agent_1_action, agent_2_action):
        """
        For each round both agents need to decide
        and eigher get rewarded or punished.
        R --> ratting out your partner
        L --> stay loyal to your partner 
        """

        actions = {
            ('R', 'R'): (1, 1), # being a rat costs but not as high as being loyal
            ('R', 'L'): (5, 0), # reward being a rat
            ('L', 'R'): (0, 5), # reward being a rat
            ('L', 'L'): (3, 3), # being loyal is costly
        }

        return actions[(agent_1_action, agent_2_action)]
    
    def get_action(self):
        return self.ACTIONS
    
    def get_state(self):
        return self.STATES
    
    def get_idx_from_state(self, state):
        return self.STATE_IDX[state]
    
    def get_state_from_idx(self, idx):
        return self.IDX_STATE[idx]