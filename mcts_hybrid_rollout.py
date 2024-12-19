from utils4e import MCT_Node, ucb
import random
import numpy as np


def alpha_beta_cutoff_eval(state, game, d=4):
    """Search game to evaluate state; use alpha-beta pruning.
    This version cuts off search, returning 0 if no terminal states were found."""

    player = game.to_move(state)

    # Functions used by alpha_beta
    def max_value(state, alpha, beta, depth):
        if cutoff_test(state, depth):
            return eval_fn(state)
        v = -np.inf
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a), alpha, beta, depth + 1))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(state, alpha, beta, depth):
        if cutoff_test(state, depth):
            return eval_fn(state)
        v = np.inf
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a), alpha, beta, depth + 1))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    # Body of alpha_beta_cutoff_search starts here:
    # The default test cuts off at depth d or at a terminal state
    cutoff_test = lambda state, depth: depth > d or game.terminal_test(state)
    eval_fn = lambda state: 0
    if player == 1:
        best_score = -np.inf
        beta = np.inf
        for a in game.actions(state):
            v = min_value(game.result(state, a), best_score, beta, 1)
            if v > best_score:
                best_score = v
        return best_score
    else:
        best_score = np.inf
        alpha = -np.inf
        for a in game.actions(state):
            v = max_value(game.result(state, a), alpha, best_score, 1)
            if v < best_score:
                best_score = v
        return best_score

def mcts_hybrid_rollout_search(state, game, N=1000, minimax_depth=2):
    def select(n):
        """select a leaf node in the tree"""
        if n.children:
            return select(max(n.children.keys(), key=ucb))
        else:
            return n

    def expand(n):
        """expand the leaf node by adding all its children states"""
        if not n.children and not game.terminal_test(n.state):
            n.children = {MCT_Node(state=game.result(n.state, action), parent=n): action
                          for action in game.actions(n.state)}
        return select(n)

    def simulate(game, state):
        """simulate the utility of current state by random picking a step"""
        player = game.to_move(state)
        while not game.terminal_test(state):
            mm_eval = alpha_beta_cutoff_eval(state, game, d=minimax_depth)
            if mm_eval != 0:
                return mm_eval
            action = random.choice(list(game.actions(state)))
            state = game.result(state, action)
        v = game.utility(state, player)
        return -v

    def backprop(n, utility):
        """passing the utility back to all parent nodes"""
        if utility > 0:
            n.U += utility
        # if utility == 0:
        #     n.U += 0.5
        n.N += 1
        if n.parent:
            backprop(n.parent, -utility)

    root = MCT_Node(state=state)

    for _ in range(N):
        leaf = select(root)
        child = expand(leaf)
        result = simulate(game, child.state)
        backprop(child, result)

    max_state = max(root.children, key=lambda p: p.N)

    return root.children.get(max_state)