# Author: Sai Satwik Yarapothini
# Date: 2024-10-05
# Description: This program implements a two-player Connect Four game with the help of 8 functions. Players take turns to drop their tokens into a 6x7 grid.
# The game checks for wins, ties, and handles invalid inputs. It also allows players to start a new game.


def printBoard(board):
    # Display the current state of the Connect Four game board
    for row in range(5, -1, -1):
        print(f"| {row + 1} |", end=" ")
        for col in board[row]:
            print(col, "|", end=" ")
        print()
        print("----------------------------------")
    print("|R/C| a | b | c | d | e | f | g |")
    print("----------------------------------")


def resetBoard():
    # Create a new 6x7 board for the game
    return [[" " for _ in range(7)] for _ in range(6)]


def validateEntry(board, col):
    # Check if the column entered by the user is valid
    if col < 0 or col >= 7:
        return False  # Out of bounds
    return any(
        board[row][col] == " " for row in range(6)
    )  # Check if there's space in the column


def checkFull(board):
    # Check if the board is fully occupied
    return all(all(cell != " " for cell in row) for row in board)


def availablePosition(board):
    # Return a list of available positions where players can drop their tokens
    positions = []
    for col in range(7):  # Check each column
        for row in range(6):  # Check each row from bottom up
            if board[row][col] == " ":  # Find the first empty space
                positions.append(
                    chr(col + 97) + str(row + 1)
                )  # Append as column letter + row number
                break  # Only need the lowest position
    return positions


def checkWin(board, turn):
    # Check if the current player has won the game
    for row in range(6):
        for col in range(7):
            if board[row][col] == turn:
                # Check horizontally
                if col + 3 < 7 and all(board[row][col + i] == turn for i in range(4)):
                    return True
                # Check vertically
                if row + 3 < 6 and all(board[row + i][col] == turn for i in range(4)):
                    return True
                # Check diagonal (bottom-left to top-right)
                if (
                    row + 3 < 6
                    and col + 3 < 7
                    and all(board[row + i][col + i] == turn for i in range(4))
                ):
                    return True
                # Check diagonal (top-left to bottom-right)
                if (
                    row - 3 >= 0
                    and col + 3 < 7
                    and all(board[row - i][col + i] == turn for i in range(4))
                ):
                    return True
    return False


def checkEnd(board, turn):
    # Determine if the game is over due to a win or a full board
    if checkWin(board, turn):
        return True  # Game over due to a win
    if checkFull(board):
        return True  # Game over due to a draw
    return False


def play():
    while True:
        board = resetBoard()
        current_turn = "X"

        print("New game: X goes first.\n")

        while True:
            printBoard(board)
            print(f"\n{current_turn}'s turn.")

            while True:
                available_positions = availablePosition(board)
                print(f"Available positions are: {available_positions}")
                move = (
                    input("Please enter column-letter and row-number (e.g., a1): ")
                    .strip()
                    .lower()
                )

                try:
                    col_letter = move[0]
                    col = ord(col_letter) - ord("a")

                    if validateEntry(board, col):
                        break

                    print("Invalid entry: try again.")

                except (IndexError, ValueError):
                    print(
                        "Invalid input. Please enter column-letter and row-number (e.g., a1)."
                    )

            # Place token at the lowest available position in the column
            for row in range(6):
                if board[row][col] == " ":
                    board[row][col] = current_turn
                    break

            print("Thank you for your selection.\n")

            if checkEnd(board, current_turn):
                printBoard(board)
                print(f"\n{current_turn} IS THE WINNER!!!")

                choice = input("Another game (y/n)? ").strip().lower()
                if choice != "y":
                    print("Thank you for playing!")
                    return

            current_turn = "O" if current_turn == "X" else "X"


# Main Function for User Interaction
if __name__ == "__main__":
    play()
