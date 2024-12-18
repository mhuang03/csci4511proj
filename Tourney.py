import sys
sys.path.append('aima-python')
from games4e import *
import math

def good_eval(state):
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
                sum += state[x][y][z]
            eval += quicksum(sum, add1, add2, add3)
    for x in range(4):
        for z in range(4):
            sum = 0
            for y in range(4):
                sum += state[x][y][z]
            eval += quicksum(sum, add1, add2, add3)
    for y in range(4):
        for z in range(4):
            sum = 0
            for x in range(4):
                sum += state[x][y][z]
            eval += quicksum(sum, add1, add2, add3)
    for x in range(4):
        for orient in (-1, 1):
            sum = 0
            for which in range(4):
                sum += state[x][which][(orient*which) % 4]
            eval += quicksum(sum, add1, add2, add3)
    for y in range(4):
        for orient in (-1, 1):
            sum = 0
            for which in range(4):
                sum += state[which][y][(orient*which) % 4]
            eval += quicksum(sum, add1, add2, add3)
    for z in range(4):
        for orient in (-1, 1):
            sum = 0
            for which in range(4):
                sum += state[which][(orient*which) % 4][z]
            eval += quicksum(sum, add1, add2, add3)
    for orient1 in (-1, 1):
        for orient2 in (-1, 1):
            sum = 0
            for which in range(4):
                sum += state[which][(orient1*which) % 4][(orient2*which) % 4]
            eval += quicksum(sum, add1, add2, add3)
    return eval

def quicksum(val, add1, add2, add3):
    if val == 4:
        return add3
    if sum == 3:
        return add2
    if sum == -4:
        return -add3
    if sum == -3:
        return -add1
    return 0



def ab_custom_player3(game, state):
    return alpha_beta_cutoff_search(state, game, d=3, eval_fn=good_eval)

class Tourney1:
    def __init__(self):
        pass

def main():
    tourney = Tourney1()