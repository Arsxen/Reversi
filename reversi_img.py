import cv2 as cv
import numpy as np
import os
import shutil


class ReversiImg:
    def __init__(self, show=True, save=False, folder=None, delay=0):
        self.black_cell = cv.imread('BlackCell.png')
        self.emp_cell = cv.imread('EmptyCell.png')
        self.white_cell = cv.imread('WhiteCell.png')
        self.img_board = None
        self.show_img = show
        self.save_img = save
        self.save_path = folder
        self.img_num = 1
        self.delay = delay
        if self.save_img and os.path.exists(self.save_path):
            shutil.rmtree(self.save_path)

    def set_img_board(self, board):
        rows = []
        for i in range(8):
            row = [self.player_to_img(board[i, j]) for j in range(8)]
            rows.append(np.concatenate(row, axis=1))
        self.img_board = np.concatenate(rows, axis=0)

    def show_board_img(self, board):
        self.set_img_board(board)
        if self.img_board is None:
            raise Exception("No Board Set!")
        if self.save_img:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            file = os.path.join(self.save_path, f"{self.img_num}.png")
            cv.imwrite(file, self.img_board)
            self.img_num += 1
        if self.show_img:
            cv.imshow('Reversi Board', self.img_board)

    def player_to_img(self, player: int):
        if player == 0:
            return self.emp_cell
        elif player == 1:
            return self.black_cell
        else:
            return self.white_cell

    def wait(self, delay=None):
        if self.show_img:
            if delay is None:
                delay = self.delay
            cv.waitKey(delay)

    def close_board(self):
        if self.show_img:
            cv.destroyAllWindows()
