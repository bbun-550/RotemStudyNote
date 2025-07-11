import random

def make_map(cols, rows,mines):
    
    
    grid = []
    for c in range(cols):
        for r in range(rows):
            grid.append([c,r])
            
    mines_loc = random.sample(grid, mines) # ex) [[1,2],[7,3]...]
    
    game_map = [] # [0,0,0,0...],[0,0,0,0...] ...    
    for c in range(cols):
        map_row = []
        for r in range(rows):
            map_row.append(0)
        game_map.append(map_row)
    
    for c,r in mines_loc:
        game_map[c][r] = -1
        # 지뢰 주변 +1 상하 좌우 대각선4곳
        for d_c in [-1,0,1]: 
            for d_r in [-1,0,1]: # 9칸 격자
                non_c = c+d_c
                non_r = r+d_r
                if (0<= non_c < cols) and (0 <= non_r < rows) and game_map[non_c][non_r] != -1:
                    game_map[non_c][non_r] +=1

    print(game_map)


make_map(9,9,1)