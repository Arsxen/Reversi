import numpy as np
import gym
import boardgame2 as bg2

_ENV = gym.make('Reversi-v0')
_ENV.reset()


def transition(board, player, action):
    """Return a new board if the action is valid, otherwise None."""
    if _ENV.is_valid((board, player), action):
        new_board, __ = _ENV.get_next_state((board, player), action)
        return new_board
    return None


def get_move(board, player):
    valid_actions = _ENV.get_valid((board, player))
    valid_actions = np.array(list(zip(*valid_actions.nonzero())))
    return valid_actions


class ZobristHashing:
    def __init__(self):
        board_size = (8, 8)
        uint64max = np.iinfo(np.uint64).max
        self._table = {bg2.WHITE: np.random.randint(uint64max, size=board_size, dtype=np.uint64),
                       bg2.EMPTY: np.random.randint(uint64max, size=board_size, dtype=np.uint64),
                       bg2.BLACK: np.random.randint(uint64max, size=board_size, dtype=np.uint64)}

    def hash_board(self, board):
        h = np.uint64(0)
        for i in range(8):
            for j in range(8):
                h = h ^ self._table.get(board[i, j])[i, j]
        return int(h)


class OthelloBoard:
    zobrist_hash = ZobristHashing()

    def __init__(self, board):
        self.board = board

    def __hash__(self):
        return self.zobrist_hash.hash_board(self.board)

    def __eq__(self, other):
        return self.board == other.board


if __name__ == '__main__':
    env = gym.make('Reversi-v0')
    env_board, turn = env.reset()
    d = {}
    state = OthelloBoard(env_board)
    m = get_move(env_board, 1)[0]
    new_board = transition(env_board, 1, m)
    n_state = OthelloBoard(new_board)
    d[(state, 0)] = 0
    d[(state, 0)] = 1
    print(d)
