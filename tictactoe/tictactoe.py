"""
Tic Tac Toe Player
"""

import math
import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    # since x gets the first move
    count_X = 0
    count_O = 0
    for row in board:
        for i in row:
            if i == X:
                count_X += 1
            elif i == O:
                count_O += 1
            else:
                continue
    if count_X > count_O :
        return O
    else:
        return X


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    actions = set()
    for row in range(3):
        for col in range(3):
            if board[row][col] == EMPTY:
                actions.add((row,col))
                
    return actions



def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    if action not in actions(board):
        raise Exception("INVALID ACTION")

    new_board = copy.deepcopy(board)
    new_board[action[0]][action[1]] = player(board)
    return new_board
  

def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    for i in range(3):
        if board[i][0] ==  board[i][1]  and board[i][2] == board[i][1]:
            return board[i][1]
        elif board[0][i] ==  board[1][i] and board[2][i] == board[1][i]:
            return board[1][i]
    if board[0][0] ==  board[1][1] and board[2][2] == board[1][1]:
        return board[1][1]
    elif board[0][2] == board[1][1] and board[2][0] == board[1][1]:
        return board[1][1]
    else:
        return None

    


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    actions_list = actions(board)
    if len(actions_list) == 0 or winner(board) is not None:
        return True
    else:
        return False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    win = winner(board)
    if terminal(board):
        if win == X:
            return 1
        elif win == O:
            return -1
        else:
            return 0

def max_value(board):
    actions_list = actions(board)
    if terminal(board):
        return utility(board), None
    else:
        maxEval = -100
        move = None
        for action in actions_list:
            Eval , act = min_value(result(board,action))
            if Eval > maxEval:
                maxEval = Eval
                move =  action
                if maxEval == 1:
                    return maxEval,move 
        return maxEval,move

def min_value(board):
    actions_list = actions(board)
    if terminal(board):
        return utility(board), None
    else:
        minEval = +100
        move = None
        for action in actions_list:
            Eval , act = max_value(result(board,action))
            if Eval < minEval:
                minEval = Eval
                move =  action
                if minEval == -1:
                    return minEval,move 
        return minEval,move

def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """ 
    if terminal(board):
        return None
    else:
        if player(board) == X:
            value,action = max_value(board)
            return action
        else:
            value,action = min_value(board)
            return action
        




