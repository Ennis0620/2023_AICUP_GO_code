import numpy as np

def remove_dead_stones(board, move):
    board_size = board.shape[0]
    visited = np.zeros((board_size, board_size), dtype=bool)
    move_color = board[move]

    for i in range(board_size):
        for j in range(board_size):
            stone = board[i, j]
            if stone == -move_color and not visited[i, j]:
                group = []
                group, visited = find_connected_group(board, (i, j), visited, group)
                if is_dead_group(board, group):
                    for stone_position in group:
                        board[stone_position] = 0

def find_connected_group(board, position, visited, group):
    board_size = board.shape[0]
    stone_color = board[position]
    stack = [position]
    while stack:
        i, j = stack.pop()
        if visited[i, j]:
            continue
        visited[i, j] = True
        group.append((i, j))
        neighbors = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
        for neighbor in neighbors:
            ni, nj = neighbor
            if 0 <= ni < board_size and 0 <= nj < board_size and board[ni, nj] == stone_color:
                stack.append((ni, nj))
    return group, visited

def is_dead_group(board, group):
    board_size = board.shape[0]
    for i, j in group:
        neighbors = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
        for ni, nj in neighbors:
            if 0 <= ni < board_size and 0 <= nj < board_size:
                neighbor_color = board[ni, nj]
                if neighbor_color == 0:
                    return False
    return True

def calculate_liberties(board, position):
    board_size = board.shape[0]
    stone_color = board[position]

    liberties = set()
    visited = set()

    def get_adjacent(position):
        i, j = position
        return [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]

    def dfs(current):
        visited.add(current)
        for adj in get_adjacent(current):
            if adj in visited:
                continue
            if 0 <= adj[0] < board_size and 0 <= adj[1] < board_size:
                adj_color = board[adj]
                if adj_color == 0:
                    liberties.add(adj)
                elif adj_color == stone_color:
                    dfs(adj)

    dfs(position)
    return len(liberties)

def calculate_all_liberties(board):
    board_size = board.shape[0]
    liberties = {}
    visited = set()

    def get_adjacent(position):
        i, j = position
        return [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]

    def dfs(current, stone_color):
        visited.add(current)
        for adj in get_adjacent(current):
            if adj in visited:
                continue
            if 0 <= adj[0] < board_size and 0 <= adj[1] < board_size:
                adj_color = board[adj]
                if adj_color == 0:
                    liberties[stone_color] = liberties.get(stone_color, set())
                    liberties[stone_color].add(adj)
                elif adj_color == stone_color:
                    dfs(adj, stone_color)

    for i in range(board_size):
        for j in range(board_size):
            if board[i, j] != 0 and (i, j) not in visited:
                visited = set()
                dfs((i, j), board[i, j])

    liberties_count = {color: len(liberties[color]) for color in liberties}
    return liberties_count

def calculate_all_liberties_19x19(board):
    board_size = board.shape[0]
    liberties = np.zeros((board_size, board_size))
    visited = set()
    def get_adjacent(position):
        i, j = position
        return [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]

    def dfs(current, stone_color):
        visited.add(current)
        for adj in get_adjacent(current):
            if adj in visited:
                continue
            if 0 <= adj[0] < board_size and 0 <= adj[1] < board_size:
                adj_color = board[adj]
                if adj_color == 0:
                    liberties[adj] += 1
                elif adj_color == stone_color:
                    dfs(adj, stone_color)

    for i in range(board_size):
        for j in range(board_size):
            if board[i, j] != 0:
                visited = set()
                dfs((i, j), board[i, j])

    return liberties

def calculate_liberties_for_each_position(board):
    board_size = board.shape[0]
    liberties_board = np.zeros((board_size, board_size), dtype=int)

    def get_adjacent(position):
        i, j = position
        return [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]

    def count_liberties(current):
        visited = set()
        stack = [current]
        liberties = 0

        while stack:
            current = stack.pop()
            visited.add(current)

            for adj in get_adjacent(current):
                x, y = adj
                if 0 <= x < board_size and 0 <= y < board_size and adj not in visited:
                    adj_color = board[x, y]
                    if adj_color == 0:
                        liberties += 1
                    elif adj_color == stone_color:
                        stack.append(adj)

        return liberties

    for i in range(board_size):
        for j in range(board_size):
            if board[i, j] == 1:  # Assuming 1 represents black stones
                stone_color = 1
                liberties_board[i, j] = count_liberties((i, j))
            elif board[i, j] == -1:  # Assuming -1 represents white stones
                stone_color = -1
                liberties_board[i, j] = count_liberties((i, j))

    return liberties_board

class board_pick:
    def __init__(self, board, move) -> None:
        self.board = board
        self.board_size = board.shape[0]
        self.visited = np.zeros((self.board_size, self.board_size), dtype=bool)
        self.move = move

    def remove_dead_stones(self):
        self.visited = np.zeros((self.board_size, self.board_size), dtype=bool)#初始化visited
        move_color = self.board[self.move]
        for i in range(self.board_size):
            for j in range(self.board_size):
                stone = self.board[i, j]
                if stone == -move_color and not self.visited[i, j]:
                    group = []
                    group, self.visited = self.find_connected_group((i, j), group)
                    if self.is_dead_group(group):
                        for stone_position in group:
                            self.board[stone_position] = 0
        return self.board

    def find_connected_group(self, position, group):
        stone_color = self.board[position]
        stack = [position]
        while stack:
            i, j = stack.pop()
            if self.visited[i, j]:
                continue
            self.visited[i, j] = True
            group.append((i, j))
            neighbors = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
            for neighbor in neighbors:
                ni, nj = neighbor
                if 0 <= ni < self.board_size and 0 <= nj < self.board_size and self.board[ni, nj] == stone_color:
                    stack.append((ni, nj))
        return group, self.visited

    def is_dead_group(self, group):
        for i, j in group:
            neighbors = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
            for ni, nj in neighbors:
                if 0 <= ni < self.board_size and 0 <= nj < self.board_size:
                    neighbor_color = self.board[ni, nj]
                    if neighbor_color == 0:
                        return False
        return True

if __name__ == '__main__':

    board_size = 19
    move = (0,2)
    board = np.array([
        [ 0,  -1,   1, -1,   1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 1,  -1,  -1,  1,  -1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,   1,   0,  1,  -1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,   1,   1, -1,   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  -1,  1,  -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  -1,  1,  -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
    ])


    print("提子後(用class)：")
    print(board_pick(board,move).remove_dead_stones())

    remove_dead_stones(board ,move)
    print("提子後：")
    
    print(board)
    print("計算氣：")
    print(calculate_liberties_for_each_position(board))
    