import numpy as np
import random

class ImprovedHeuristicGomokuAI:
    def __init__(self, board_size, win_size):
        self.board_size = board_size
        self.win_size = win_size
        self.directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        self.pattern_scores = {
            (5, 0): 1000000,  # Win
            (4, 2): 100000,   # Open four
            (4, 1): 10000,    # Semi-open four
            (3, 2): 5000,     # Open three
            (3, 1): 1000,     # Semi-open three
            (2, 2): 500,      # Open two
            (2, 1): 100,      # Semi-open two
            (1, 2): 50,       # Open one
        }

    def evaluate_board(self, board, player_index):
        score = 0
        opponent_index = 3 - player_index
        for x in range(self.board_size):
            for y in range(self.board_size):
                if board[0, x, y] == 1:  # Empty cell
                    score += self.evaluate_position(board, x, y, player_index)
                    score -= self.evaluate_position(board, x, y, opponent_index) * 1.1  # Slightly prioritize defense
        return score

    def evaluate_position(self, board, x, y, player_index):
        position_score = 0
        for dx, dy in self.directions:
            pattern = self.get_pattern(board, x, y, dx, dy, player_index)
            if pattern in self.pattern_scores:
                position_score += self.pattern_scores[pattern]
        return position_score

    def get_pattern(self, board, x, y, dx, dy, player_index):
        line = []
        open_ends = 0
        for i in range(-4, 5):  # Check 9 cells (4 on each side + center)
            nx, ny = x + dx * i, y + dy * i
            if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                if board[player_index, nx, ny] == 1:
                    line.append(1)
                elif board[0, nx, ny] == 1:
                    if len(line) == 0 or (len(line) > 0 and line[-1] != 0):
                        open_ends += 1
                    line.append(0)
                else:
                    line.append(2)  # Opponent's piece
            else:
                line.append(2)  # Treat out of bounds as opponent's piece

        # Find the longest consecutive sequence of player's pieces
        max_length = 0
        current_length = 0
        for cell in line:
            if cell == 1:
                current_length += 1
                max_length = max(max_length, current_length)
            else:
                current_length = 0

        return (max_length, open_ends)

    def generate_moves(self, board, valid_actions):
        num_pieces = np.sum(board[1,:,:] + board[2,:,:])
        
        if num_pieces == 0:
            return [(self.board_size // 2, self.board_size // 2)]
        
        if num_pieces < 4:
            return self.moves_near_center(board, valid_actions)
        
        return self.get_strategic_moves(board, valid_actions)

    def moves_near_center(self, board, valid_actions):
        center = self.board_size // 2
        moves = [(x, y) for x in range(center-2, center+3) 
                 for y in range(center-2, center+3) 
                 if (x, y) in valid_actions]
        return moves if moves else list(valid_actions)

    def get_strategic_moves(self, board, valid_actions):
        moves = []
        for x, y in valid_actions:
            score = self.evaluate_position(board, x, y, 1) + self.evaluate_position(board, x, y, 2)
            moves.append(((x, y), score))
        moves.sort(key=lambda x: x[1], reverse=True)
        return [move for move, _ in moves[:min(20, len(moves))]]  # Return top 20 moves

    def select_best_move(self, board, player, possible_moves):
        player_index = 2 if player == 'x' else 1
        opponent_index = 3 - player_index
        best_move = None
        best_score = -float('inf')
        
        for move in possible_moves:
            new_board = board.copy()
            new_board[0, move[0], move[1]] = 0
            new_board[player_index, move[0], move[1]] = 1
            
            score = self.evaluate_board(new_board, player_index)
            
            # Check if this move blocks opponent's winning move
            if self.is_winning_move(board, move, opponent_index):
                score += 500000  # High priority for blocking opponent's win
            
            # Check if this move creates a winning position
            if self.is_winning_move(new_board, move, player_index):
                score += 1000000  # Highest priority for winning move
            
            if score > best_score:
                best_move = move
                best_score = score
        
        return best_move

    def is_winning_move(self, board, move, player_index):
        x, y = move
        for dx, dy in self.directions:
            count = 1  # Count the piece at (x, y)
            # Check in positive direction
            for i in range(1, self.win_size):
                nx, ny = x + dx * i, y + dy * i
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size and board[player_index, nx, ny] == 1:
                    count += 1
                else:
                    break
            # Check in negative direction
            for i in range(1, self.win_size):
                nx, ny = x - dx * i, y - dy * i
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size and board[player_index, nx, ny] == 1:
                    count += 1
                else:
                    break
            if count >= self.win_size:
                return True
        return False

class Submission:
    def __init__(self, board_size, win_size):
        self.ai = ImprovedHeuristicGomokuAI(board_size, win_size)

    def __call__(self, state):
        current_player = state.current_player()
        valid_moves = state.valid_actions()
        return self.ai.select_best_move(state.board, current_player, self.ai.generate_moves(state.board, valid_moves))
