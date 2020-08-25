class Sudoku:
    def __init__(self, board, n_rows, n_cols):
        self.board = board
        self.n_rows = n_rows
        self.n_cols = n_cols
    
    
    def find_empty(self):
        """Find an empty cell in the board"""
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                if self.board[i][j] == 0:
                    return (i, j)
        
        return None
    
    
    def isValid(self, num, row, col):
        """Check if the position is valid to put the given number"""
        # check row
        for i in range(self.n_rows):
            if i != col and self.board[row][i] == num:
                return False
        
        # check column
        for i in range(self.n_cols):
            if i != row and self.board[i][col] == num:
                return False
        
        # check the 3x3 box
        box_x = col // 3
        box_y = row // 3

        for i in range(box_y*3, box_y*3 + 3):
            for j in range(box_x*3, box_x*3 + 3):
                if i != row and j != col and self.board[i][j] == num:
                    return False
        
        # all checks passes, return true
        return True
    

    def print_board(self):
        """Prints the board to the screen"""
        for i in range(self.n_rows):
            if i % 3 == 0:
                print("+-----------------------+")
            
            for j in range(self.n_cols):
                if j % 3 == 0:
                    print("| ", end="")
                if j == self.n_cols-1:
                    if self.board[i][j] == 0:
                        print(" ", '|')
                    else:
                        print(self.board[i][j], '|')
                else:
                    if self.board[i][j] == 0:
                        print(" ", end=" ")
                    else:
                        print(self.board[i][j], end=" ")
            
            if i == self.n_rows-1:
                print("+-----------------------+")
    

    def solve(self):
        """Solves the sudoku puzzle using Backtracking Algorithm"""
        # search for an empty cell
        empty_cell = self.find_empty()

        # if there is no empty cell, puzzle is solved, return
        if empty_cell is None:
            return True
        
        # get the position of the empty cell
        row, col = empty_cell

        # try to add a number from 1 to 9 in the empty cell
        for num in range(1, 10):
            if self.isValid(num, row, col):
                self.board[row][col] = num
                if self.solve():
                    return True
                self.board[row][col] = 0
        
        return False
