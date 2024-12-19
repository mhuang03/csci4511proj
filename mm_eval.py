
def mm_eval(state):
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
