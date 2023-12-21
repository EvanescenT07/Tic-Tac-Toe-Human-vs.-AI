import tkinter as tk
import random
from tkinter import messagebox
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

BLANK = ' '
AI_PLAYER = 'X'
HUMAN_PLAYER = 'O'

REWARD_WIN = 1
REWARD_TIE = 0
REWARD_LOSE = 0

class Player:
    @staticmethod
    def show_board(board):
        print('|'.join(board[0:3]))
        print('|'.join(board[3:6]))
        print('|'.join(board[6:9]))

class HumanPlayer(Player):
    def reward(self, value, board):
        pass

    def make_move(self, board):
        while True:
            try:
                self.show_board(board)
                move = input('Your next move (cell index 1-9):')
                move = int(move)
                if not (move - 1 in range(9)):
                    raise ValueError
            except ValueError:
                print('Invalid move; try again:\n')
            else:
                return move - 1

class AIPlayer(Player):
    def __init__(self, epsilon=0.4, alpha=0.3, gamma=0.9):
        self.EPSILON = epsilon
        self.ALPHA = alpha
        self.GAMMA = gamma

        self.q = Sequential()
        self.q.add(Dense(32, input_dim=36, activation='relu'))
        self.q.add(Dense(1, activation='relu'))
        self.q.compile(optimizer='adam', loss='mean_squared_error')
        self.move = None
        self.board = (' ',) * 9

    def available_moves(self, board):
        return [i for i in range(9) if board[i] == ' ']

    def encode_input(self, board, action):
        vector_representation = []

        for cell in board:
            for ticker in ['X', 'O', ' ']:
                if cell == ticker:
                    vector_representation.append(1)
                else:
                    vector_representation.append(0)

        for move in range(9):
            if action == move:
                vector_representation.append(1)
            else:
                vector_representation.append(0)

        return np.array([vector_representation])

    def make_move(self, board):
        self.board = tuple(board)
        actions = self.available_moves(board)

        if random.random() < self.EPSILON:
            self.move = random.choice(actions)
            return self.move

        q_values = [self.get_q(self.board, a) for a in actions]
        max_q_value = max(q_values)

        if q_values.count(max_q_value) > 1:
            best_actions = [i for i in range(len(actions)) if q_values[i] == max_q_value]
            best_move = actions[random.choice(best_actions)]
        else:
            best_move = actions[q_values.index(max_q_value)]

        self.move = best_move
        return self.move

    def get_q(self, state, action):
        return self.q.predict([self.encode_input(state, action)], batch_size=1)

    def reward(self, reward, board):
        if self.move:
            prev_q = self.get_q(self.board, self.move)
            max_q_new = max([self.get_q(tuple(board), a) for a in self.available_moves(self.board)])
            self.q.fit(self.encode_input(self.board, self.move),
                       prev_q + self.ALPHA * ((reward + self.GAMMA * max_q_new) - prev_q),
                       epochs=3, verbose=0)

        self.move = None
        self.board = None

class TicTacToe:
    def __init__(self, player1, player2):
        self.player1 = player1
        self.player2 = player2
        self.first_player_turn = random.choice([True, False])
        self.board = [' '] * 9

    def play(self):
        while True:
            if self.first_player_turn:
                player = self.player1
                other_player = self.player2
            else:
                player = self.player2
                other_player = self.player1

            game_over, winner = self.is_game_over()

            if game_over:
                if winner == AI_PLAYER:
                    print('\n AI Player won!')
                    player.reward(REWARD_WIN, self.board[:])
                    other_player.reward(REWARD_LOSE, self.board[:])
                elif winner == HUMAN_PLAYER:
                    print('\n Human Player won!')
                    other_player.reward(REWARD_WIN, self.board[:])
                    player.reward(REWARD_LOSE, self.board[:])
                else:
                    print('Tie!')
                    player.reward(REWARD_TIE, self.board[:])
                    other_player.reward(REWARD_TIE, self.board[:])
                break

            self.first_player_turn = not self.first_player_turn
            move = player.make_move(self.board)
            self.board[move] = HUMAN_PLAYER if player == self.player1 else AI_PLAYER

    def is_game_over(self):
        for player_ticker in [AI_PLAYER, HUMAN_PLAYER]:
            for i in range(3):
                if self.board[3 * i + 0] == player_ticker and \
                        self.board[3 * i + 1] == player_ticker and \
                        self.board[3 * i + 2] == player_ticker:
                    return True, player_ticker

            for j in range(3):
                if self.board[j + 0] == player_ticker and \
                        self.board[j + 3] == player_ticker and \
                        self.board[j + 6] == player_ticker:
                    return True, player_ticker

            if self.board[0] == player_ticker and self.board[4] == player_ticker and self.board[8] == player_ticker:
                return True, player_ticker

            if self.board[2] == player_ticker and self.board[4] == player_ticker and self.board[6] == player_ticker:
                return True, player_ticker

        if BLANK not in self.board:
            return True, None
        else:
            return False, None


class TicTacToeGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Tic-Tac-Toe")
        self.board = [' '] * 9
        self.current_player = None
        self.ai_score = 0
        self.human_score = 0
        self.create_board_buttons()

        loaded_model = load_model('model.h5')
        self.ai_player = AIPlayer()
        self.ai_player.q = loaded_model
        self.ai_player.EPSILON = 0

        self.human_player = HumanPlayer()

        self.start_game()

    def create_board_buttons(self):
        self.buttons = []
        for i in range(3):
            for j in range(3):
                button = tk.Button(self.master, text='', font=('normal', 20), width=6, height=2,
                                   command=lambda row=i, col=j: self.make_move(row, col))
                button.grid(row=i, column=j)
                self.buttons.append(button)

        # Display scores
        self.score_label = tk.Label(self.master, text=f'Scores: AI {self.ai_score} - Human {self.human_score}')
        self.score_label.grid(row=3, column=0, columnspan=3)

    def start_game(self):
        self.current_player = random.choice([self.ai_player, self.human_player])

        if self.current_player == self.ai_player:
            self.ai_make_move()

    def make_move(self, row, col):
        index = 3 * row + col

        if self.board[index] == ' ':
            self.board[index] = HUMAN_PLAYER
            self.update_button(row, col)

            if self.check_winner() or BLANK not in self.board:
                self.end_game()
                return

            self.ai_make_move()

            if self.check_winner() or BLANK not in self.board:
                self.end_game()

    def ai_make_move(self):
        move = self.ai_player.make_move(self.board)
        row, col = divmod(move, 3)
        self.board[move] = AI_PLAYER
        self.update_button(row, col)

        if self.check_winner() or BLANK not in self.board:
            self.end_game()

    def update_button(self, row, col):
        index = 3 * row + col
        self.buttons[index].config(text=self.board[index], state=tk.DISABLED)

    def check_winner(self):
        for i in range(3):
            if self.board[3 * i] == self.board[3 * i + 1] == self.board[3 * i + 2] != ' ':
                return self.board[3 * i]
            if self.board[i] == self.board[i + 3] == self.board[i + 6] != ' ':
                return self.board[i]

        if self.board[0] == self.board[4] == self.board[8] != ' ':
            return self.board[0]

        if self.board[2] == self.board[4] == self.board[6] != ' ':
            return self.board[2]

        return None

    def end_game(self):
        winner = self.check_winner()

        if winner:
            if winner == 'X':
                self.ai_score += 1
            elif winner == 'O':
                self.human_score += 1

            self.show_winner_message(winner)
        else:
            self.show_winner_message("It's a tie!")

        for button in self.buttons:
            button.config(state=tk.DISABLED)

        self.reset_game()

    def show_winner_message(self, winner):
        messagebox.showinfo("Game Over", f"{winner} wins!")
        self.score_label.config(text=f'Scores: AI {self.ai_score} - Human {self.human_score}')

    def reset_game(self):
        self.board = [' '] * 9
        for button in self.buttons:
            button.config(text='', state=tk.NORMAL)

        self.start_game()

# Create the main window
root = tk.Tk()
app = TicTacToeGUI(root)
root.mainloop()
