import copy
import itertools
import random
from collections import namedtuple
from games4e import *

import numpy as np

from utils4e import vector_add, MCT_Node, ucb

GameState = namedtuple('GameState', 'to_move, utility, board, moves')
StochasticGameState = namedtuple('StochasticGameState', 'to_move, utility, board, moves, chance')

def good_eval(state):
    board = state.board

    if state.to_move == 1:
        add1 = 100
        add2 = 1
        add3 = 10000
    else:
        add1 = 1
        add2 = 100
        add3 = 10000

    eval = 0
    for x in range(4):
        for y in range(4):
            sum = 0
            for z in range(4):
                sum += board[x][y][z]
            eval += quicksum(sum, add1, add2, add3)
    for x in range(4):
        for z in range(4):
            sum = 0
            for y in range(4):
                sum += board[x][y][z]
            eval += quicksum(sum, add1, add2, add3)
    for y in range(4):
        for z in range(4):
            sum = 0
            for x in range(4):
                sum += board[x][y][z]
            eval += quicksum(sum, add1, add2, add3)
    for x in range(4):
        for orient in (-1, 1):
            sum = 0
            for which in range(4):
                sum += board[x][which][calcDiag(orient, which)]
            eval += quicksum(sum, add1, add2, add3)
    for y in range(4):
        for orient in (-1, 1):
            sum = 0
            for which in range(4):
                sum += board[which][y][calcDiag(orient, which)]
            eval += quicksum(sum, add1, add2, add3)
    for z in range(4):
        for orient in (-1, 1):
            sum = 0
            for which in range(4):
                sum += board[calcDiag(orient, which)][calcDiag(orient, which)][z]
            eval += quicksum(sum, add1, add2, add3)
    for orient1 in (-1, 1):
        for orient2 in (-1, 1):
            sum = 0
            for which in range(4):
                sum += board[which][calcDiag(orient1, which)][calcDiag(orient2, which)]
            eval += quicksum(sum, add1, add2, add3)
    return eval

def calcDiag(orient, which):
    if orient == 1:
        return which
    else:
        return 3-which

def quicksum(val, add1, add2, add3):
    if val == 4:
        return add3
    if val == 3:
        return add2
    if val == -4:
        return -add3
    if val == -3:
        return -add1
    return 0

def ab_dep1_player1(state, game):
    return alpha_beta_cutoff_search(state, game, d=1, eval_fn=good_eval)

def ab_dep1_player2(game, state):
    return alpha_beta_cutoff_search(state, game, d=1, eval_fn=lambda x: -good_eval(x))

# ______________________________________________________________________________
# Monte Carlo Tree Search Hybrid


def monte_carlo_hybrid_tree_search(state, game, N=1000):
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
        if state.to_move == 1:
            player = ab_dep1_player1
        else:
            player = ab_dep1_player2
        move = player(game, state)
        state2 = game.result(state, move)
        if game.terminal_test(state2):
            return -game.utility(state2, player)

        """simulate the utility of current state by random picking a step"""
        player = game.to_move(state)
        while not game.terminal_test(state):
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