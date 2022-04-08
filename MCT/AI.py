import SimTree

class AI():
    def __init__(self, env, limit):
        self.env = env
        self.limit = limit # This could be a time limit (module) or a limit on # of moves

    # This is where the MCTS is run by the AI, before it selects a move
    def monte_carlo_search(self):
        mcts = SimTree.BattleshipsMonteCarloTreeSearch(self.env.opp_board, guess_limit=self.limit)
        best_action = mcts.best_action()
        return best_action

    def move(self):
        # Make a move based on running the MCTS
        best_action = self.monte_carlo_search()
        return self.env.step(best_action)
    