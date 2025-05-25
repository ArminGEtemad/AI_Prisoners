from agents_personalities import AgentsPersonalities
from game_rules import GameRules
from one_chance_game import OneChanceGame

personalities = AgentsPersonalities(lr = [0.1, 0.1],
                                    gamma = [0.99, 0.99],
                                    epsilon = [1.0, 1.0],
                                    epsilon_decay = [0.9995, 0.9995],
                                    min_epsilon = [0.01, 0.01])
game_rules = GameRules()

one_chance_game = OneChanceGame(game_rules=game_rules, personalities=personalities)
one_chance_game.training(num_episodes=10000, track_window=100)