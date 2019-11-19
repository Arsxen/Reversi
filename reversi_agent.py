"""
This module contains agents that play reversi.

Version 3.0
"""

import abc
import random
import asyncio
import traceback
import time
from multiprocessing import Process, Value
from math import inf
import sys

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


class ReversiAgent(abc.ABC):
    """Reversi Agent."""

    def __init__(self, color):
        """
        Create an agent.

        Parameters
        -------------
        color : int
            BLACK is 1 and WHITE is -1. We can get these constants
            from bg2.BLACK and bg2.WHITE.

        """
        super().__init__()
        self._move = None
        self._color = color

    @property
    def player(self):
        """Return the color of this agent."""
        return self._color

    @property
    def pass_move(self):
        """Return move that skips the turn."""
        return np.array([-1, 0])

    @property
    def best_move(self):
        """Return move after the thinking.

        Returns
        ------------
        move : np.array
            The array contains an index x, y.

        """
        if self._move is not None:
            return self._move
        else:
            return self.pass_move

    async def move(self, board, valid_actions):
        """Return a move. The returned is also availabel at self._move."""
        self._move = None
        output_move_row = Value('d', -1)
        output_move_column = Value('d', 0)
        try:
            # await self.search(board, valid_actions)
            p = Process(
                target=self.search,
                args=(
                    self._color, board, valid_actions,
                    output_move_row, output_move_column))
            p.start()
            while p.is_alive():
                await asyncio.sleep(0.1)
        except asyncio.CancelledError as e:
            print('The previous player is interrupted by a user or a timer.')
        except Exception as e:
            print(type(e).__name__)
            print('move() Traceback (most recent call last): ')
            traceback.print_tb(e.__traceback__)
        finally:
            p.kill()
            self._move = np.array(
                [output_move_row.value, output_move_column.value],
                dtype=np.int32)
        return self.best_move

    @abc.abstractmethod
    def search(
            self, color, board, valid_actions,
            output_move_row, output_move_column):
        """
        Set the intended move to self._move.

        The intended move is a np.array([r, c]) where r is the row index
        and c is the column index on the board. [r, c] must be one of the
        valid_actions, otherwise the game will skip your turn.

        Parameters
        -------------------
        board : np.array
            An 8x8 array that contains
        valid_actions : np.array
            An array of shape (n, 2) where n is the number of valid move.

        Returns
        -------------------
        None
            This method should set value for
            `output_move_row.value` and `output_move_column.value`
            as a way to return.

        """
        raise NotImplementedError('You will have to implement this.')


class RandomAgent(ReversiAgent):
    """An agent that move randomly."""

    def search(
            self, color, board, valid_actions,
            output_move_row, output_move_column):
        """Set the intended move to the value of output_moves."""
        # If you want to "simulate a move", you can call the following function:
        # transition(board, self.player, valid_actions[0])

        # To prevent your agent to fail silently we should an
        # explicit trackback printout.
        try:
            # while True:
            #     pass
            # time.sleep(1)
            randidx = random.randint(0, len(valid_actions) - 1)
            random_action = valid_actions[randidx]
            output_move_row.value = random_action[0]
            output_move_column.value = random_action[1]
        except Exception as e:
            print(type(e).__name__, ':', e)
            print('search() Traceback (most recent call last): ')
            traceback.print_tb(e.__traceback__)


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
        return np.array_equal(self.board, other.board)


class TonAgent(ReversiAgent):
    """An agent that move follow by Ton Rule"""
    depth_limit = 4

    def __init__(self, color):
        super().__init__(color)
        # (board,depth) -> (minimax value, depth)
        self.max_transposition = {}
        self.min_transposition = {}
        self.position_weight = np.array([[1.00, -0.25, 0.10, 0.05, 0.5, 0.10, -0.25, 1.00],
                                         [-0.25, -0.25, 0.01, 0.01, 0.01, 0.01, -0.25, -0.25],
                                         [0.10, 0.01, 0.05, 0.02, 0.02, 0.05, 0.01, 0.10],
                                         [0.05, 0.01, 0.02, 0.01, 0.01, 0.02, 0.01, 0.05],
                                         [0.05, 0.01, 0.02, 0.01, 0.01, 0.02, 0.01, 0.05],
                                         [0.10, 0.01, 0.05, 0.02, 0.02, 0.05, 0.01, 0.10],
                                         [-0.25, -0.25, 0.01, 0.01, 0.01, 0.01, -0.25, -0.25],
                                         [1.00, -0.25, 0.10, 0.05, 0.5, 0.10, -0.25, 1.00]])

    def utility(self, winner):
        if winner == self.player:
            return 100000000
        elif winner == 0:
            return 0
        else:
            return -100000000

    def weighted_piece_counter(self, board):
        score = 0
        for i in range(8):
            for j in range(8):
                if board[i, j] == 0:
                    b = 0
                elif board[i, j] == self.player:
                    b = 1
                else:
                    b = -1
                score += self.position_weight[i, j] * b
        return score

    def coin_parity(self, board):
        return (np.sum(board) * self.player)/np.sum(board != 0)

    def mobility(self, board):
        max_move = len(get_move(board, self.player))
        min_move = len(get_move(board, -self.player))
        return (max_move-min_move)/(max_move+min_move)

    def evaluate(self, board):
        weights = np.array([10, 1])
        scores = np.array([self.weighted_piece_counter(board), self.coin_parity(board)])
        return np.sum(weights*scores)
        # return random.randint(0, 1000)

    def alpha_beta_search(self, board, valid_actions):
        v, action = self.root_max_value(board, -inf, inf, valid_actions)
        return action

    def root_max_value(self, board, alpha, beta, valid_actions):
        v = -inf
        state = OthelloBoard(board)
        max_action = valid_actions[0]
        for action in valid_actions:
            next_board = transition(board, self.player, action)
            next_state = OthelloBoard(next_board)
            if (next_state, 1) not in self.max_transposition:
                min_node_val = self.min_value(next_board, alpha, beta, 1)
            else:
                min_node_val = self.max_transposition.get((next_state, 1))
            if v < min_node_val:
                v = min_node_val
                max_action = action
            if v >= beta:
                self.max_transposition[(state, 0)] = v
                return v, max_action
            alpha = max(alpha, v)
        self.max_transposition[(state, 0)] = v
        return v, max_action

    def max_value(self, board, alpha, beta, depth):
        winner = _ENV.get_winner((board, self.player))
        if winner is not None:
            return self.utility(winner)
        if depth == self.depth_limit:
            return self.evaluate(board)
        v = -inf
        state = OthelloBoard(board)
        valid_actions = get_move(board, self.player)
        for action in valid_actions:
            next_board = transition(board, self.player, action)
            next_state = OthelloBoard(next_board)
            if (next_state, depth + 1) not in self.max_transposition:
                v = max(v, self.min_value(next_board, alpha, beta, depth + 1))
            else:
                v = max(v, self.max_transposition.get((next_state, depth + 1)))
            if v >= beta:
                self.max_transposition[(state, depth)] = v
                return v
            alpha = max(alpha, v)
        self.max_transposition[(state, depth)] = v
        return v

    def min_value(self, board, alpha, beta, depth):
        winner = _ENV.get_winner((board, -self.player))
        if winner is not None:
            return self.utility(winner)
        if depth == self.depth_limit:
            return self.evaluate(board)
        v = inf
        state = OthelloBoard(board)
        valid_actions = get_move(board, -self.player)
        for action in valid_actions:
            next_board = transition(board, -self.player, action)
            next_state = OthelloBoard(next_board)
            if (next_state, depth + 1) not in self.min_transposition:
                v = min(v, self.max_value(next_board, alpha, beta, depth + 1))
            else:
                v = min(v, self.min_transposition.get((next_state, depth + 1)))
            if v <= alpha:
                self.min_transposition[(state, depth)] = v
                return v
            beta = min(beta, v)
        self.min_transposition[(state, depth)] = v
        return v

    def search(
            self, color, board, valid_actions,
            output_move_row, output_move_column):
        try:
            # while True:
            #     pass
            move = self.alpha_beta_search(board, valid_actions)
            output_move_row.value = move[0]
            output_move_column.value = move[1]
        except Exception as e:
            print(type(e).__name__, ':', e)
            print('search() Traceback (most recent call last): ')
            traceback.print_tb(e.__traceback__)
