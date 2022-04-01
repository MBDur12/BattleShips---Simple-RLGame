import copy
import MCTSNode
import MCTS

class AI():
    def __init__(self, env, limit):
        self.env = env
        self.limit = limit # This could be a time limit (module) or a limit on # of moves

    # This is where the MCTS is run by the AI, before it selects a move
    def monte_carlo_search(self):
        node = MCTSNode.BattleshipsMonteCarloTreeSearchNode(self.env.opp_board)
        mcts = MCTS.MonteCarloTreeSearch(node)
        best_child = mcts.best_action(simulations_number=5)
        return best_child
    


    def move(self):
        # Make a move based on running the MCTS
        best_child = self.monte_carlo_search()
        return self.env.step(best_child.action)
    