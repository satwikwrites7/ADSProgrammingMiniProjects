# Author: Sai Satwik Yarapothini
# Date: 2024-10-05
# Description: This program implements a two-player Tic-Tac-Toe game sharing same keyboard.
# Players have thier turns to place their marks (X or O) on a 3x3 grid.The game checks for wins, ties, and handles invalid inputs. It also allows players to start a new game.

# Importing Colorama Library which allows to print colored text in the terminal.
from colorama import init, Fore


# TicTacToe Class with '8' Functions : __init__(), printBoard(), resetBoard(), validateEntry(), checkFull(),
# checkWin(), checkEnd(), play()
class TicTacToe:
    def __init__(self):
        # Initialize the board as a 3x3 grid filled with spaces
        self.board = [[" " for _ in range(3)] for _ in range(3)]
        # Set the initial turn to 'X'
        self.current_turn = "X"

    def printBoard(self):
        # Display the current state of the game board
        print("-----------------")
        print("|R\\C| 0 | 1 | 2 |")
        print("-----------------")
        for i, row in enumerate(self.board):
            print(f"| {i} | " + " | ".join(row) + " |")
            print("-----------------")

    def resetBoard(self):
        # Reset the board and set the first turn to 'X'
        self.board = [[" " for _ in range(3)] for _ in range(3)]
        self.current_turn = "X"
        print("\nNew Game: X goes first.\n")

    def validateEntry(self, row, col):
        # Validate if the entered row and column are within bounds and not already taken
        if row < 0 or row > 2 or col < 0 or col > 2:  # Check for within bounds
            print(f"You have entered row #{row}")
            print(f"          and column #{col}")
            print("Invalid entry: try again.")
            print("Row & column numbers must be either 0, 1, or 2.\n")
            return False
        if self.board[row][col] != " ":  # Check whether cell is already taken
            print(f"You have entered row #{row}")
            print(f"          and column #{col}")
            print("That cell is already taken.\nPlease make another selection.\n")
            return False
        return True

    def checkFull(self):
        # Check if the board is completely filled
        return all(self.board[i][j] != " " for i in range(3) for j in range(3))

    def checkWin(self):
        # Check all rows, columns, and diagonals for a win condition
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] != " ":
                return True
            if self.board[0][i] == self.board[1][i] == self.board[2][i] != " ":
                return True
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != " ":
            return True
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != " ":
            return True
        return False

    def checkEnd(self):
        # Determine if the game has ended in a win or draw
        if self.checkWin():  # Check for Win
            print(f"\n{self.current_turn} IS THE WINNER!!!")
            self.printBoard()
            return True
        elif self.checkFull():  # Check for Draw
            print("DRAW! NOBODY WINS!")
            self.printBoard()
            return True
        return False

    def play(self):
        init()  # Initialize colorama for colored text output
        print("New Game: X goes first.\n")

        while True:
            self.printBoard()

            while True:
                print(f"\n{self.current_turn}'s turn.")
                move = input(
                    f"Where do you want your {self.current_turn} placed?\nPlease enter row number and column number separated by a comma.\n{Fore.RED}"
                )  # For Highlighting the User Input with Red Color
                row_col = move.split(",")
                print(Fore.RESET)  # Reset color after input

                try:
                    row, col = map(int, row_col)
                    if self.validateEntry(row, col):
                        break
                except ValueError:
                    print(f"\nInvalid input. Please enter row,col (e.g., 0,0)\n")

            # Place the current player's mark on the board
            self.board[row][col] = self.current_turn
            print(f"\nYou have entered row #{row}")
            print(f"          and column #{col}")
            print("Thank you for your selection.\n")

            if self.checkEnd():
                choice = input("Another game? Enter Y or y for yes: ").lower()
                if choice != "y":
                    print("Thanks you for playing.")
                    break
                self.resetBoard()
            else:
                # Switch turns between 'X' and 'O'
                self.current_turn = "O" if self.current_turn == "X" else "X"


# Main Function for GamePlay (User Interaction)
if __name__ == "__main__":
    game = TicTacToe()
    game.play()
