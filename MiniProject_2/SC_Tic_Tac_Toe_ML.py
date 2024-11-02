# Authors(Team): Sainath Chettupally, Sai Satwik Yarapothini, Srikar Gowrishetty
# Date: 10-30-2024
# Overview: This script simulates a Tic-Tac-Toe game where a human player competes against an RandomForestClassifier ML model.
# The human player and the model take turns placing their respective marks ('X' or 'O') on a 3x3 grid.
# The model is trained using a Intermediate boards optimal play dataset 'tictac_single.txt' of Tic-Tac-Toe board states to predict optimal moves.
# The program validates each move, updates the board, and checks for a winning condition or a tie after every turn. The game continues until there is a winner or the board is full. Players can choose to play again or exit after each game.

#Importing Machine Learning Libraries.
import numpy as np
import pandas as pd
import random
import joblib  # To save/load model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Load and Prepare the Dataset
def load_dataset(filename):
    #Load the Tic-Tac-Toe dataset from a text file.
    data = pd.read_csv(filename, header=None, delim_whitespace=True)
    # Remove rows with missing values
    data.dropna(inplace=True)
    # Ensure the dataset has 9 features representing the board state and 1 target column
    if data.shape[1] != 10:
        raise ValueError(
            f"Dataset does not have the expected 10 columns (9 features + 1 target). Found {data.shape[1]} columns.")
    X = data.iloc[:, :9].values   # Board state features
    y = data.iloc[:, 9].values   # Target value (winning player or optimal move)
    return X, y


# Train the Model
def train_model(X, y):
    #Train a RandomForestClassifier model to predict optimal moves
    # Check if the dataset is empty
    if X.size == 0:
        raise ValueError("Dataset is empty. Please check the dataset.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model trained with accuracy: {accuracy * 100:.2f}%")
    joblib.dump(model, 'tic_tac_toe_model.pkl')
    return model


#Load or Train the Model
def load_or_train_model(dataset_filename):
    #Load a pre-trained model or train a new one if not available
    try:
        model = joblib.load('tic_tac_toe_model.pkl')
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Model file not found. Training a new model...")
        X, y = load_dataset(dataset_filename)
        model = train_model(X, y)
    return model


def get_model_move(board_state, model):
    #Predict the next optimal move for player O.
    board_state = np.array(board_state).reshape(1, -1)
    return model.predict(board_state)[0]


def check_win(board):
    #Check for a win condition.
    win_conditions = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Horizontal wins
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Vertical wins
        [0, 4, 8], [2, 4, 6]  # Diagonal wins
    ]

    for condition in win_conditions:
        if board[condition[0]] == board[condition[1]] == board[condition[2]] != 0:
            return board[condition[0]]  # Return the winner (1 for X, -1 for O)
    return None


def play_game(model):
    #Play a Tic-Tac-Toe game where a human player competes against the ML model.
    while True:
        board = [0] * 9  # Initialize an empty board
        current_player = 1  # Player X starts first

        while True:
            print_board(board)
            if current_player == 1:
                # Player's turn (X)
                try:
                    move_input = input("Enter your move as row,col (e.g., 0,0 for top-left): ")
                    move_coords = tuple(map(int, move_input.split(',')))
                    move = move_coords[0] * 3 + move_coords[1]
                    if move < 0 or move > 8 or board[move] != 0:
                        print("Invalid move. Try again.")
                        continue
                    board[move] = 1
                except (ValueError, IndexError):
                    print("Invalid input. Please enter the move as row,col (e.g., 0,0 for top-left).")
                    continue
            else:
                # Model's turn (O)
                print("Model is making a move...")
                model_move = get_model_move(board, model)
                if board[model_move] != 0:
                    print("Model attempted an invalid move. Adjusting...")
                    available_moves = [i for i in range(9) if board[i] == 0]
                    model_move = random.choice(available_moves)
                board[model_move] = -1

            # Check for win condition or draw
            winner = check_win(board)
            if winner:
                print_board(board)
                if winner == 1:
                    print("Player X wins!")
                else:
                    print("Player O (model) wins!")
                break

            if 0 not in board:
                print_board(board)
                print("It's a draw!")
                break

            # Switch players
            current_player *= -1

        # Ask if the player wants to play again
        play_again = input("Do you want to play another game? (yes/no): ").strip().lower()
        if play_again != 'yes':
            print("Thanks for playing!")
            break


def print_board(board):
    #Display the current board state with row and column labels
    symbols = {1: 'X', -1: 'O', 0: ' '}
    print("-----------------")
    print("|R\\C| 0 | 1 | 2 |")
    print("-----------------")
    for i in range(3):
        print(f"| {i} | {' | '.join([symbols[board[j]] for j in range(i * 3, (i + 1) * 3)])} |")
        print("-----------------")
    print()


#Calling Main Function
if __name__ == "__main__":
    dataset_filename = 'tictac_single.txt'
    model = load_or_train_model(dataset_filename)
    play_game(model)
