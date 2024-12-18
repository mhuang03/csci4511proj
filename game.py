from typing import *


class TTT3D:
    """
    Implements 3D tic-tac-toe.
    """

    n: int
    turn: int
    board: list[list[list[int]]]

    def __init__(self, n: int = 4):
        self.n = n
        self.turn = 1  # X is 1, O is -1, X goes first
        self.board = [[[0 for _ in range(n)] for _ in range(n)] for _ in range(n)]

    def display(self) -> None:
        """
        Prints out the board and turn states.
        """

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
                    col_sep.join([" X " if i == 1 else " O " if i == -1 else "   " for i in self.board[b][r]]))
        strs = [board_sep.join(row) for row in rows]

        label = "".join(
            [(f"Board {i}:" + " " * (len(hor_line) + len(board_sep) - len(str(i)) - 7)) for i in range(self.n)])

        print("X" if self.turn == 1 else "O", "to move")
        print()
        print(label)
        print(("\n" + between + "\n").join(strs))

    def move(self, pos: Tuple[int, int, int]) -> None:
        """
        Makes player ``self.turn`` play at pos.

        :param pos: an int 3-tuple indicating position
        """

        b, r, c = pos
        if self.board[b][r][c] != 0:
            raise RuntimeError("illegal move")

        self.board[b][r][c] = self.turn
        self.turn = -self.turn

    def get_moves(self) -> list[Tuple[int, int, int]]:
        """
        Gets the list of available moves.

        :return: a list of int 3-tuples indicating legal move positions
        """

        moves = []
        for b in range(self.n):
            for r in range(self.n):
                for c in range(self.n):
                    if self.board[b][r][c] == 0:
                        moves.append((b, r, c))
        return moves

    def detect_win(self) -> int:
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
            main_space_f += self.board[i][i][i]
            main_space_b += self.board[i][i][self.n-1-i]
            off_space_f += self.board[i][self.n-1-i][i]
            off_space_b += self.board[i][self.n-1-i][self.n-1-i]

            # diags
            main_sum_b, off_sum_b, main_sum_r, off_sum_r, main_sum_c, off_sum_c = 0, 0, 0, 0, 0, 0
            for j in range(self.n):
                main_sum_b += self.board[i][j][j]
                off_sum_b += self.board[i][j][self.n-1-j]
                main_sum_r += self.board[j][i][j]
                off_sum_r += self.board[j][i][self.n-1-j]
                main_sum_c += self.board[j][j][i]
                off_sum_c += self.board[j][self.n-1-j][i]

                # rows/cols/verts
                row_sum, col_sum, vert_sum = 0, 0, 0
                for k in range(self.n):
                    row_sum += self.board[i][j][k]
                    col_sum += self.board[i][k][j]
                    vert_sum += self.board[k][i][j]

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

    def wins_on(self, player, pos) -> bool:
        """
        Calculates whether ``player`` would win after playing at ``pos``.

        :param player: 1 or -1
        :param pos: an int 3-tuple
        :return: whether the player wins
        """

        b, r, c = pos
        if player != self.turn:
            raise RuntimeError("not the correct turn")
        if self.board[b][r][c] != 0:
            raise RuntimeError("illegal move")

        self.board[b][r][c] = player
        w = self.detect_win()
        self.board[b][r][c] = 0
        return w == player


if __name__ == "__main__":
    game = TTT3D(4)
    game.board[0][0][0] = 1
    game.board[1][1][0] = 1
    game.board[2][2][0] = 1
    game.board[3][3][0] = 1

    game.display()
    print(game.get_moves)
    print(game.detect_win())
