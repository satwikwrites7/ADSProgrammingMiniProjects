# Author: Sai Satwik Yarapothini
# Date: 2024-10-29
# Description: This program implements a two-player Tic-Tac-Toe game sharing same keyboard using the OOPS concepts : Classes and Objects.
# Players have thier turns to place their marks (X or O) on a 3x3 grid.The game checks for wins, ties, and handles invalid inputs. It also allows players to start a new game.

# Importing Colorama Library which allows to print colored text in the terminal.
from colorama import init, Fore


# Defining Board class to build the Game Board.
class Board:
    # Constructor initiates the board with empty cells using c as attribute name
    def __init__(self):
        self.c = [[" ", " ", " "], [" ", " ", " "], [" ", " ", " "]]

    # Method to print the board
    def printBoard(self):
        BOARD_HEADER = "-----------------\n|R\\C| 0 | 1 | 2 |\n-----------------"
        print(BOARD_HEADER)
        for i in range(3):
            print(f"| {i} | " + " | ".join(self.c[i]) + " |")
            print("-----------------")


# Defining Game class to implement the Game Logic.
class Game:
    # Constructor which Initializes the board and set the first turn to 'X'
    def __init__(self):
        self.board = Board()
        self.turn = "X"

    # Method to switch players
    def switchPlayer(self):
        self.turn = "O" if self.turn == "X" else "X"

    # Method to validate the user's entry
    def validateEntry(self, row, col):
        if (
            row < 0 or row > 2 or col < 0 or col > 2
        ):  # Check if the entered row and column are within bounds
            print(f"You have entered row #{row}")
            print(f"          and column #{col}")
            print("Invalid entry: try again.")
            print("Row & column numbers must be either 0, 1, or 2.\n")
            return False
        if self.board.c[row][col] != " ":  # Check if the cell is already taken
            print(f"You have entered row #{row}")
            print(f"          and column #{col}")
            print("That cell is already taken.\nPlease make another selection.\n")
            return False
        return True

    # Method to check if the board is full
    def checkFull(self):
        return all(self.board.c[i][j] != " " for i in range(3) for j in range(3))

    # Method to check for a winner
    def checkWin(self):
        for i in range(3):
            if self.board.c[i][0] == self.board.c[i][1] == self.board.c[i][2] != " ":
                return True
            if self.board.c[0][i] == self.board.c[1][i] == self.board.c[2][i] != " ":
                return True
        if self.board.c[0][0] == self.board.c[1][1] == self.board.c[2][2] != " ":
            return True
        if self.board.c[0][2] == self.board.c[1][1] == self.board.c[2][0] != " ":
            return True
        return False

    # Method to check if the game has met an END condition by calling checkFull() and checkWin() class methods
    def checkEnd(self):
        if self.checkWin():  # Check for a Win
            print(f"\n{self.turn} IS THE WINNER!!!")
            self.board.printBoard()
            return True
        elif self.checkFull():  # Check for a Draw
            print("DRAW! NOBODY WINS!")
            self.board.printBoard()
            return True
        return False

    # Method to run the tic-tac-toe game
    def playGame(self):
        init()  # Initialize colorama for colored text output
        print("New Game: X goes first.\n")

        while True:
            self.board.printBoard()

            while True:
                # Asking the Current Player for their move.
                print(f"\n{self.turn}'s turn.")
                move = input(
                    f"Where do you want your {self.turn} placed?\nPlease enter row number and column number separated by a comma.\n{Fore.RED}"
                )  # Fore.RED for having Input as Red Color for User Input on Terminal
                row_col = move.split(",")
                print(Fore.RESET)

                try:
                    row, col = map(int, row_col)
                    if self.validateEntry(row, col):
                        break
                except ValueError:
                    print(f"\nInvalid input. Please enter row,col (e.g., 0,0)\n")

            # Placing the player's mark on the board as per their specified move.
            self.board.c[row][col] = self.turn
            print(f"\nYou have entered row #{row}")
            print(f"          and column #{col}")
            print("Thank you for your selection.\n")

            # Check if the game has ended
            if self.checkEnd():
                return
            # Switch to the other player
            self.switchPlayer()


# Main Function for USER INTERACTION
def main():
    play_again = True  # Initializing a variable to repeat the game
    while (
        play_again
    ):  # Using while-loop that runs until the user says no for another game
        game = Game()
        game.playGame()
        choice = input("Another game? Enter Y or y for yes: ").lower()
        play_again = choice == "y"

    print("Thank you for playing!")  # Goodbye Message


# Calling the Main Function
if __name__ == "__main__":
    main()
