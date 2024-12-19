from typing import *
from copy import deepcopy
from games4e import Game, GameState


class TTT3D(Game):
    """
    Implements 3D tic-tac-toe.
    """

    n: int
    turn: int

    def __init__(self, n: int = 4):
        self.n = n

        # X is 1, O is -1, X goes first
        board = [[[0 for _ in range(n)] for _ in range(n)] for _ in range(n)]
        moves = [(i, j, k) for i in range(n) for j in range(n) for k in range(n)]
        self.initial = GameState(to_move=1, utility=0, board=board, moves=moves)

    def display(self, state) -> None:
        """
        Prints out the board and turn states.
        """
        board = state.board

        col_sep = "|"
        row_sep = "---"
        jun_sep = "+"
        board_sep = "   "
        hor_line = (row_sep + jun_sep) * (self.n - 1) + row_sep
        between = (hor_line + board_sep) * self.n

        rows = [[] for _ in range(self.n)]
        for b in range(self.n):
            for r in range(self.n):
                rows[r].append(
                    col_sep.join([" X " if i == 1 else " O " if i == -1 else "   " for i in board[b][r]]))
        strs = [board_sep.join(row) for row in rows]

        label = "".join(
            [(f"Board {i}:" + " " * (len(hor_line) + len(board_sep) - len(str(i)) - 7)) for i in range(self.n)])

        if state.utility == 0:
            print("X" if state.to_move == 1 else "O", "to move")
        else:
            print("Winner:", "X" if state.utility == 1 else "O")
        print(label)
        print(("\n" + between + "\n").join(strs))
        print()

    def result(self, state, move: Tuple[int, int, int]) -> GameState:
        """
        Makes current player play at pos.

        :param state: the current game state
        :param move: an int 3-tuple indicating the next position to play in
        """

        b, r, c = move
        board = deepcopy(state.board)
        if board[b][r][c] != 0:
            raise RuntimeError("illegal move")
        board[b][r][c] = state.to_move

        moves = state.moves.copy()
        moves.remove(move)

        return GameState(
            to_move= 1 if state.to_move == -1 else -1,
            utility=self.detect_win(board),
            board=board,
            moves=moves
        )

    def actions(self, state) -> list[Tuple[int, int, int]]:
        """
        Gets the list of available moves.

        :return: a list of int 3-tuples indicating legal move positions
        """

        board = state.board
        moves = []
        for b in range(self.n):
            for r in range(self.n):
                for c in range(self.n):
                    if board[b][r][c] == 0:
                        moves.append((b, r, c))
        return moves

    def utility(self, state, player):
        return state.utility

    def detect_win(self, board) -> int:
        """
        Detects whether a player has won.

        :return: 0 if no win, 1 if X won, -1 if O won.
        """

        def has_win(*args):
            top, bot = max(*args), min(*args)
            return 1 if top == self.n else -1 if bot == -self.n else 0

        # space diags
        main_space_f, main_space_b, off_space_f, off_space_b = 0, 0, 0, 0
        for i in range(self.n):
            main_space_f += board[i][i][i]
            main_space_b += board[i][i][self.n-1-i]
            off_space_f += board[i][self.n-1-i][i]
            off_space_b += board[i][self.n-1-i][self.n-1-i]

            # diags
            main_sum_b, off_sum_b, main_sum_r, off_sum_r, main_sum_c, off_sum_c = 0, 0, 0, 0, 0, 0
            for j in range(self.n):
                main_sum_b += board[i][j][j]
                off_sum_b += board[i][j][self.n-1-j]
                main_sum_r += board[j][i][j]
                off_sum_r += board[j][i][self.n-1-j]
                main_sum_c += board[j][j][i]
                off_sum_c += board[j][self.n-1-j][i]

                # rows/cols/verts
                row_sum, col_sum, vert_sum = 0, 0, 0
                for k in range(self.n):
                    row_sum += board[i][j][k]
                    col_sum += board[i][k][j]
                    vert_sum += board[k][i][j]

                w = has_win(row_sum, col_sum, vert_sum)
                if w != 0:
                    return w

            w = has_win(main_sum_b, off_sum_b, main_sum_r, off_sum_r, main_sum_c, off_sum_c)
            if w != 0:
                return w

        w = has_win(main_space_f, main_space_b, off_space_f, off_space_b)
        if w != 0:
            return w

        return 0

    def terminal_test(self, state) -> bool:
        return self.detect_win(state.board) != 0 or len(state.moves) == 0

    def play_game(self, *players ,verbose=False):
        state = self.initial
        while True:
            for player in players:
                move = player(self, state)
                state = self.result(state, move)
                if verbose:
                    self.display(state)
                if self.terminal_test(state):
                    return self.utility(state, self.to_move(self.initial))