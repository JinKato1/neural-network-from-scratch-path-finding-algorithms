import numpy as np


#Returns 3 tuples. Each tuples contains a 2d nparray containg only 0s or 9s with the specificed number of rows and column
# and a list containing the correct coordinates (in tuple form) for those grids
#First grid has 0s for the first row and 0s at the end of each row
#second grid has 0s at the beginning of each row and all zeroes the last row
#the third grid has a zig zag pattern of zeroes and the number of rows and columns must be the same for it to work
#you can also set the low value and high value (need to change to low=1 and high=10 for ant colony)
def test_grids(num_rows, num_cols, low=0, high=9):
    test_grids = []
    nines_grid = np.full((num_rows, num_cols), high)

    top_right_correct_grid = np.copy(nines_grid)
    coordinates = []
    for i in range(num_rows):
        if i == 0:
            top_right_correct_grid[0] = np.full((num_cols), low)
            for i in range(num_cols):
                coordinates.append((0, i))
        else:
            top_right_correct_grid[i][-1] = low
            coordinates.append((i, len(top_right_correct_grid[i]) - 1))
    test_grids.append((top_right_correct_grid, coordinates))
    coordinates = []
    bottom_left_correct_grid = np.copy(nines_grid)
    for i in range(num_rows):
        if i == num_rows - 1:
            bottom_left_correct_grid[i] = np.full((num_cols), low)
            for i in range(num_cols):
                coordinates.append((num_rows - 1, i))
        else:
            bottom_left_correct_grid[i][0] = low
            coordinates.append((i, 0))
    test_grids.append((bottom_left_correct_grid, coordinates))


    zig_zag_grid = np.copy(nines_grid)
    col = 0
    row = 0
    coordinates = [(0, 0)]
    while row != num_rows - 1 and col != col - 1:
        new_row = row + 1
        new_col = col + 1

        coordinates.append((row, new_col))
        coordinates.append((new_row, new_col))
        row = new_row
        col = new_col

    for coord in coordinates:
        zig_zag_grid[coord] = low

    test_grids.append((zig_zag_grid, coordinates))

    return test_grids

