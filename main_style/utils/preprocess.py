import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .board_up import remove_dead_stones, calculate_liberties_for_each_position

chars = 'abcdefghijklmnopqrst'
coordinates = {k:v for v,k in enumerate(chars)}

def augmentation(prepare_input_board):
    rotated_90 = np.rot90(prepare_input_board, k=1, axes=(1, 2))
    rotated_90_h = np.flip(rotated_90, axis=2)
    rotated_90_v = np.flip(rotated_90, axis=1)
    rotated_180 = np.rot90(prepare_input_board, k=2, axes=(1, 2))
    rotated_270 = np.rot90(prepare_input_board, k=3, axes=(1, 2))
    horizontal_flip = np.flip(prepare_input_board, axis=2)
    vertical_flip = np.flip(prepare_input_board, axis=1)
    combined_array = np.stack([prepare_input_board, rotated_90, rotated_90_h, rotated_90_v, rotated_180, rotated_270, horizontal_flip, vertical_flip], axis=0)
    return combined_array

def augmentation_crop_aug(prepare_input_board):

    combined_all_array = []
    for augment_time in range(len(prepare_input_board)):
        origin = prepare_input_board[augment_time]
        rotated_90 = np.rot90(prepare_input_board[augment_time], k=1, axes=(1, 2))
        rotated_90_h = np.flip(rotated_90, axis=2)
        rotated_90_v = np.flip(rotated_90, axis=1)
        rotated_180 = np.rot90(prepare_input_board[augment_time], k=2, axes=(1, 2))
        rotated_270 = np.rot90(prepare_input_board[augment_time], k=3, axes=(1, 2))
        horizontal_flip = np.flip(prepare_input_board[augment_time], axis=2)
        vertical_flip = np.flip(prepare_input_board[augment_time], axis=1)
        
        combined_all_array.append(origin)
        combined_all_array.append(rotated_90)
        combined_all_array.append(rotated_90_h)
        combined_all_array.append(rotated_90_v)
        combined_all_array.append(rotated_180)
        combined_all_array.append(rotated_270)
        combined_all_array.append(horizontal_flip)
        combined_all_array.append(vertical_flip)
    combined_all_array = np.array(combined_all_array)

    return combined_all_array

def rand_augmentation(prepare_input_board):
    p = np.random.random(1)
    if 0 <= p < (1/8):
        trans_board = np.rot90(prepare_input_board, k=1, axes=(1, 2))
    elif (1/8) <= p < (2/8): 
        trans_board = np.rot90(prepare_input_board, k=1, axes=(1, 2))
        trans_board = np.flip(trans_board, axis=1)
    elif (2/8) <= p < (3/8):
        trans_board = np.rot90(prepare_input_board, k=1, axes=(1, 2))
        trans_board = np.flip(trans_board, axis=2)
    elif (3/8) <= p < (4/8):
        trans_board = np.rot90(prepare_input_board, k=2, axes=(1, 2))
    elif (4/8) <= p < (5/8): 
        trans_board = np.rot90(prepare_input_board, k=3, axes=(1, 2))
    elif (5/8) <= p < (6/8): 
        trans_board = np.flip(prepare_input_board, axis=2)
    elif (6/8) <= p < (7/8): 
        trans_board = np.flip(prepare_input_board, axis=1)
    elif (7/8) <= p <= 1:
        trans_board = prepare_input_board
    return trans_board

def rand_augmentation_crop_aug(prepare_input_board):
    
    for augment_time in range(len(prepare_input_board)):
        p = np.random.random(1)
        if 0 <= p < (1/8):
            prepare_input_board[augment_time] = np.rot90(prepare_input_board[augment_time], k=1, axes=(1, 2))
        elif (1/8) <= p < (2/8): 
            prepare_input_board[augment_time] = np.rot90(prepare_input_board[augment_time], k=1, axes=(1, 2))
            prepare_input_board[augment_time] = np.flip(prepare_input_board[augment_time], axis=2)#水平
        elif (2/8) <= p < (3/8): 
            prepare_input_board[augment_time] = np.rot90(prepare_input_board[augment_time], k=1, axes=(1, 2))
            prepare_input_board[augment_time] = np.flip(prepare_input_board[augment_time], axis=1)#垂直
        elif (3/8) <= p < (4/8): 
            prepare_input_board[augment_time] = np.rot90(prepare_input_board[augment_time], k=2, axes=(1, 2))
        elif (4/8) <= p < (5/8): 
            prepare_input_board[augment_time] = np.rot90(prepare_input_board[augment_time], k=3, axes=(1, 2))
        elif (5/8) <= p < (6/8): 
            prepare_input_board[augment_time] = np.flip(prepare_input_board[augment_time], axis=2)#水平
        elif (6/8) <= p < (7/8): 
            prepare_input_board[augment_time] = np.flip(prepare_input_board[augment_time], axis=1)#垂直
        elif (7/8) <= p <= 1:
            prepare_input_board[augment_time] = prepare_input_board[augment_time]
    
    return prepare_input_board

def rand_augmentation_2path(prepare_input_board,prepare_input_board2):
    p = np.random.random(1)
    if 0 <= p < (1/8):
        trans_board = np.rot90(prepare_input_board, k=1, axes=(1, 2))
        trans_board2 = np.rot90(prepare_input_board2, k=1, axes=(1, 2))
    elif (1/8) <= p < (2/8): 
        trans_board = np.rot90(prepare_input_board, k=1, axes=(1, 2))
        trans_board = np.flip(trans_board, axis=2)
        trans_board2 = np.rot90(prepare_input_board2, k=1, axes=(1, 2))
        trans_board2 = np.flip(trans_board2, axis=2)
    elif (2/8) <= p < (3/8): 
        trans_board = np.rot90(prepare_input_board, k=1, axes=(1, 2))
        trans_board = np.flip(trans_board, axis=1)
        trans_board2 = np.rot90(prepare_input_board2, k=1, axes=(1, 2))
        trans_board2 = np.flip(trans_board2, axis=1)
    elif (3/8) <= p < (4/8): 
        trans_board = np.rot90(prepare_input_board, k=2, axes=(1, 2))
        trans_board2 = np.rot90(prepare_input_board2, k=2, axes=(1, 2))
    elif (4/8) <= p < (5/8): 
        trans_board = np.rot90(prepare_input_board, k=3, axes=(1, 2))
        trans_board2 = np.rot90(prepare_input_board2, k=3, axes=(1, 2))
    elif (5/8) <= p < (6/8): 
        trans_board = np.flip(prepare_input_board, axis=1)
        trans_board2 = np.flip(prepare_input_board2, axis=1)
    elif (7/8) <= p < (5/6): 
        trans_board = np.flip(prepare_input_board, axis=2)
        trans_board2 = np.flip(prepare_input_board2, axis=2)
    elif (5/6) <= p <= 1:
        trans_board = prepare_input_board
        trans_board2 = prepare_input_board2

    return trans_board, trans_board2

#crop board
def prepare_input_crop(moves, size=9):
    x = np.zeros((2,19,19))
    last_move_column = 0
    last_move_row = 0
    for idx,move in enumerate(moves):
        color = move[0] 
        column = coordinates[move[2]]
        row = coordinates[move[3]]
        x[0,row,column] = 1
    if moves:
        last_move_column = coordinates[moves[-1][2]]
        last_move_row = coordinates[moves[-1][3]]
        x[1,last_move_row,last_move_column] = 1
    
    if size == 19:
        return x
    else:
        half_size = size // 2
        start_row = max(0, last_move_row - half_size)
        end_row = min(start_row + size, x.shape[1])
        
        start_col = max(0, last_move_column - half_size)
        end_col = min(start_col + size, x.shape[2])

        while end_row - start_row < size:
            if end_row >= x.shape[1]-1:
                if start_row>=0:
                    start_row -= 1
            elif end_row <= x.shape[1]-1:
                end_row += 1

        while end_col - start_col < size:
            if end_col >= x.shape[2]-1:
                if start_col>=0:
                    start_col -= 1
            elif end_col <= x.shape[2]-1:
                end_col += 1

        result = x[:, start_row:end_row, start_col:end_col]

        return result
#occupied、last move、B、W
def prepare_input_channel4_crop(moves, size=9):
    x = np.zeros((4,19,19))
    last_move_column = 0
    last_move_row = 0
    for idx,move in enumerate(moves):
        color = move[0] 
        column = coordinates[move[2]]
        row = coordinates[move[3]]
        x[0,row,column] = 1
        if color == 'B':
            x[2,row,column] = 1
        elif color == 'W':
            x[3,row,column] = 1
    if moves:
        last_move_column = coordinates[moves[-1][2]]
        last_move_row = coordinates[moves[-1][3]]
        x[1,last_move_row,last_move_column] = 1
    
    if size == 19:
        return x
    else:
        half_size = size // 2
        start_row = max(0, last_move_row - half_size)
        end_row = min(start_row + size, x.shape[1])
        
        start_col = max(0, last_move_column - half_size)
        end_col = min(start_col + size, x.shape[2])

        while end_row - start_row < size:
            if end_row >= x.shape[1]-1:
                if start_row>=0:
                    start_row -= 1
            elif end_row <= x.shape[1]-1:
                end_row += 1

        while end_col - start_col < size:
            if end_col >= x.shape[2]-1:
                if start_col>=0:
                    start_col -= 1
            elif end_col <= x.shape[2]-1:
                end_col += 1

        result = x[:, start_row:end_row, start_col:end_col]

        return result
#traget color seqence
def prepare_input_channel8_crop(moves, target_color, size=9):
    x = np.zeros((8,19,19))
    last_move_column = 0
    last_move_row = 0
    counter = 0 #紀錄 target_color 第幾手
    for idx,move in enumerate(moves):
        color = move[0]
        column = coordinates[move[2]]
        row = coordinates[move[3]]
        if color == target_color:
            counter+= 1
            #紀錄下棋順序 idx=0:channl0、1:chennel1、2:channel0+1
            #因為只紀錄 黑or白的第幾手  所以會減半
            binary_idx = bin(counter)[2:]       
            binary_idx_inverse = binary_idx[::-1]#反轉字串 迴圈比較好做
            for channel,carry in enumerate(binary_idx_inverse):
                if carry == '1':
                    try:
                        x[channel, row,column] = 1
                    except:
                        print(binary_idx, idx+1)
    
    half_size = size // 2
    start_row = max(0, last_move_row - half_size)
    end_row = min(start_row + size, x.shape[1])
    
    start_col = max(0, last_move_column - half_size)
    end_col = min(start_col + size, x.shape[2])

    while end_row - start_row < size:
        if end_row >= x.shape[1]-1:
            if start_row>=0:
                start_row -= 1
        elif end_row <= x.shape[1]-1:
            end_row += 1

    while end_col - start_col < size:
        if end_col >= x.shape[2]-1:
            if start_col>=0:
                start_col -= 1
        elif end_col <= x.shape[2]-1:
            end_col += 1

    result = x[:, start_row:end_row, start_col:end_col]

    return result
#target sequence、enemy sequence
def prepare_input_channel16_crop(moves, target_color, size=9):
    x = np.zeros((16,19,19))
    last_move_column = 0
    last_move_row = 0
    counter_main = 0 #紀錄 target_color 第幾手
    counter_empty = 0#紀錄 對手順序
    for idx,move in enumerate(moves):
        color = move[0]
        column = coordinates[move[2]]
        row = coordinates[move[3]]
        if color == 'B':
            counter_main+= 1
            #紀錄下棋順序 idx=0:channl0、1:chennel1、2:channel0+1
            #因為只紀錄 黑or白的第幾手  所以會減半
            binary_idx = bin(counter_main)[2:]       
            binary_idx_inverse = binary_idx[::-1]#反轉字串 迴圈比較好做
            for channel,carry in enumerate(binary_idx_inverse):
                if carry == '1':
                    try:
                        x[channel, row,column] = 1
                    except:
                        print(binary_idx, idx+1)
        else:
            counter_empty+= 1
            binary_idx = bin(counter_main)[2:]       
            binary_idx_inverse = binary_idx[::-1]
            for channel,carry in enumerate(binary_idx_inverse):
                if carry == '1':
                    x[channel+8, row,column] = 1

    
    half_size = size // 2
    start_row = max(0, last_move_row - half_size)
    end_row = min(start_row + size, x.shape[1])
    
    start_col = max(0, last_move_column - half_size)
    end_col = min(start_col + size, x.shape[2])

    while end_row - start_row < size:
        if end_row >= x.shape[1]-1:
            if start_row>=0:
                start_row -= 1
        elif end_row <= x.shape[1]-1:
            end_row += 1

    while end_col - start_col < size:
        if end_col >= x.shape[2]-1:
            if start_col>=0:
                start_col -= 1
        elif end_col <= x.shape[2]-1:
            end_col += 1

    result = x[:, start_row:end_row, start_col:end_col]

    return result

#剪裁board大小 input 3維
def crop_board(x, crop_size, last_move_row,last_move_column):

    if len(x.shape) == 2:
        x = np.expand_dims(x,axis=0)
    if crop_size == 19:
        return x
    else:
        half_size = crop_size // 2
        start_row = max(0, last_move_row - half_size)
        end_row = min(start_row + crop_size, x.shape[1])
        start_col = max(0, last_move_column - half_size)
        end_col = min(start_col + crop_size, x.shape[2])
        while end_row - start_row < crop_size:
            if end_row >= x.shape[1]-1:
                if start_row>=0:
                    start_row -= 1
            elif end_row <= x.shape[1]-1:
                end_row += 1
        while end_col - start_col < crop_size:
            if end_col >= x.shape[2]-1:
                if start_col>=0:
                    start_col -= 1
            elif end_col <= x.shape[2]-1:
                end_col += 1
        result = x[:, start_row:end_row, start_col:end_col]

    return result

#剪裁board大小 input 4維
def crop_board_BW_inverse(x, crop_size, last_move_row,last_move_column):
    if crop_size == 19:
        return x
    else:
        half_size = crop_size // 2
        start_row = max(0, last_move_row - half_size)
        end_row = min(start_row + crop_size, x.shape[2])
        start_col = max(0, last_move_column - half_size)
        end_col = min(start_col + crop_size, x.shape[3])
        while end_row - start_row < crop_size:
            if end_row >= x.shape[2]-1:
                if start_row>=0:
                    start_row -= 1
            elif end_row <= x.shape[2]-1:
                end_row += 1
        while end_col - start_col < crop_size:
            if end_col >= x.shape[3]-1:
                if start_col>=0:
                    start_col -= 1
            elif end_col <= x.shape[3]-1:
                end_col += 1
        result = x[:, :, start_row:end_row, start_col:end_col]

    return result


#剪裁board大小 並做augmentation 去附近搜尋 不一定要在正中間
def crop_board_augmentation(x, crop_size, last_move_row,last_move_column, augment_times = 5):

    if len(x.shape) == 2:
        x = np.expand_dims(x,axis=0)
    
    if crop_size == 19:
        x = np.stack([x]*augment_times)
        return x
    else:
        half_size = crop_size // 2
        start_row = max(0, last_move_row - half_size)
        end_row = min(start_row + crop_size, x.shape[1])
        start_col = max(0, last_move_column - half_size)
        end_col = min(start_col + crop_size, x.shape[2])

        while end_row - start_row < crop_size:
            if end_row >= x.shape[1]-1:
                if start_row>=0:
                    start_row -= 1
            elif end_row <= x.shape[1]-1:
                end_row += 1
        while end_col - start_col < crop_size:
            if end_col >= x.shape[2]-1:
                if start_col>=0:
                    start_col -= 1
            elif end_col <= x.shape[2]-1:
                end_col += 1

        tmp_result = x[:, start_row:end_row, start_col:end_col]
        
        #如果在邊角 直接乘上要aug的次數
        if (last_move_row == 0 and last_move_column==0) or (last_move_row == 18 and last_move_column==0) or (last_move_row == 0 and last_move_column==18) or (last_move_row == 18 and last_move_column==18):
            result = np.stack([tmp_result]*augment_times)
            return result
        else:
            augment_times = augment_times-1
            start_row_down = max(0, last_move_row-crop_size+1)
            start_row_up = min(18, last_move_row+1)

            start_col_up = max(0, last_move_column-crop_size+1)
            end_col_down = min(18, last_move_column+1)
            aug_counter = 0
            tmp = []
            for row in range(start_row_down, start_row_up-1):
                if row+crop_size > 19:
                    break
                for col in range(start_col_up, end_col_down-1):
                    if col+crop_size > 19:
                        break
                    tmp_board = x[:,row:row+crop_size,col:col+crop_size]
                    aug_counter += 1
                    tmp.append(tmp_board)
            
            tmp = np.array(tmp)
            idx = [x for x in range(aug_counter)]
            np.random.shuffle(idx)
            result = tmp[idx]
            
            #如果數量不足 要copy補足
            if aug_counter < augment_times:
                diff = augment_times - aug_counter
                for i in range(diff):
                    result = np.append(result, np.expand_dims(result[0],axis=0) ,axis=0)
            else:
                result = result[:augment_times]

            result = np.append(result, np.expand_dims(tmp_result,axis=0),axis=0)
            return result

#剪裁board大小 並做augmentation 去附近搜尋 不一定要在正中間
def crop_board_around_augmentation(x, crop_size, last_move_row,last_move_column, augment_times = 9):
    
    if len(x.shape) == 2:
        x = np.expand_dims(x,axis=0)
    
    if crop_size == 19:
        x = np.stack([x]*augment_times)
        return x
    else:
        half_size = crop_size // 2
        start_row = max(0, last_move_row - half_size)
        end_row = min(start_row + crop_size, x.shape[1])
        start_col = max(0, last_move_column - half_size)
        end_col = min(start_col + crop_size, x.shape[2])

        while end_row - start_row < crop_size:
            if end_row >= x.shape[1]-1:
                if start_row>=0:
                    start_row -= 1
            elif end_row <= x.shape[1]-1:
                end_row += 1
        while end_col - start_col < crop_size:
            if end_col >= x.shape[2]-1:
                if start_col>=0:
                    start_col -= 1
            elif end_col <= x.shape[2]-1:
                end_col += 1

        tmp_result = x[:, start_row:end_row, start_col:end_col]
        
        #如果在邊角 直接乘上要aug的次數
        if (last_move_row == 0 and last_move_column==0) or (last_move_row == 18 and last_move_column==0) or (last_move_row == 0 and last_move_column==18) or (last_move_row == 18 and last_move_column==18):
            result = np.stack([tmp_result]*augment_times)
            return result
        else:
            tmp = []
            tmp.append(tmp_result)
            aug_counter = 1
            for direction in [(-1,1),(-1,0),(-1,1),(0,1),(0,-1),(1,1),(1,0),(1,-1)]:
                new_start_row, new_end_row, new_start_col, new_end_col = 0,0,0,0
                new_row_diff, new_col_diff = 0, 0

                if last_move_row==0 or last_move_row==18 or last_move_column==0 or last_move_column==18:
                    if last_move_row==0 or last_move_row==18:
                        if direction==(0,1) or  direction==(0,-1):
                            new_start_row = start_row + direction[0]
                            new_end_row = end_row + direction[0]
                            new_start_col = start_col + direction[1]
                            new_end_col = end_col + direction[1]
                    elif last_move_column==0 or last_move_column==18: 
                        if direction==(1,0) or direction==(-1,0):
                            new_start_row = start_row + direction[0]
                            new_end_row = end_row + direction[0]
                            new_start_col = start_col + direction[1]
                            new_end_col = end_col + direction[1]
                else:
                    new_start_row = start_row + direction[0]
                    new_end_row = end_row + direction[0]
                    new_start_col = start_col + direction[1]
                    new_end_col = end_col + direction[1]

                new_row_diff = new_end_row - new_start_row
                new_col_diff = new_end_col - new_start_col

                if ( new_row_diff == crop_size) and ( new_col_diff == crop_size) and (new_start_row>0) and (new_end_row<=19) and (new_start_col>0) and (new_end_col<=19):
                    # print(direction)
                    # print('new_start_row', new_start_row)
                    # print('new_end_row', new_end_row)
                    # print('new_start_col', new_start_col)
                    # print('new_end_col', new_end_col)
                    new_result = x[:, new_start_row:new_end_row, new_start_col:new_end_col]
                    aug_counter+=1
                    tmp.append(new_result)
                    # print('new_result shape', np.shape(new_result))

            
            # #如果數量不足 要copy補足
            if aug_counter < augment_times:
                diff = augment_times - aug_counter
                p = np.random.randint(0,aug_counter)
                for i in range(diff):
                    tmp.append(tmp[p])
            tmp = np.array(tmp)
            # print(np.shape(tmp))
            return tmp

#當前盤面表示 黑:1、白:-1、空:0  可以看是否需要提子
def prepare_board_status(moves,
                         remove_dead=True,
                         sperate_posotion=False,
                         empty_is_one = False,
                         last_move_ = False,
                         crop_size=19):
    '''
    `moves`:格式 [B[pq],W[qd],B[cp],W[dd],..]\n
    `remove_dead`:是否要提子\n
    `sperate_posotion`:是否回傳黑、白獨立的所在盤面\n
    `empty_is_one`:回傳的empty位置給予的值,True:代表為1(被佔領的位置=0),False:代表為0(被佔領的位置=1)\n
    `last_move`:是否為最後一手的位置\n
    `crop_size`:要切盤面的大小
    `Return`\n 
    `x`:最終盤面的狀況(黑=1、白=-1、空=0)
    `B_W_postition`:額外的資訊(sperate_posotion、empty_is_one、last_move)組成
    `B_W_ration`:黑白棋數量比例
    '''
    x = np.zeros((19,19)).astype(np.int8)
    last_move = np.zeros((19,19)).astype(np.int8)
    last_move_row = 0
    last_move_column = 0
    for idx,move in enumerate(moves):
        color = move[0]
        column = coordinates[move[2]]
        row = coordinates[move[3]]
        if color == 'B':
            x[row,column] = 1
        elif color == 'W':
            x[row,column] = -1            

        if remove_dead:
            remove_dead_stones(x, (row,column))
            # print(f'step{idx}',move )
            # show(x)
        if idx == len(moves)-1:
            last_move_row = row
            last_move_column = column
            last_move[last_move_row,last_move_column] = 1


    if sperate_posotion:
        #黑、白色各自佔據的位置(=1)、盤面為0的位置(=1)
        B_postion = np.where(x==1,1,0)
        W_postition = np.where(x==-1,1,0)
        if empty_is_one:
            non_position = np.where(x==0,1,0)#被佔領的位置=0
        else:
            non_position = np.where(x==0,0,1)#被佔領的位置=1

        if last_move_:
            B_W_postition = np.stack([B_postion,W_postition,non_position,last_move],axis=0)
        else:
            B_W_postition = np.stack([B_postion,W_postition,non_position],axis=0)

        #黑白子數量差距比例  黑棋/白棋
        B_num = np.count_nonzero(B_postion)
        W_num = np.count_nonzero(W_postition)
        if W_num==0:
            B_W_ration = -1 #如果是剛開始的盤面 可能沒有白色棋
        else:
            B_W_ration = B_num/W_num

        x = crop_board(x,crop_size,last_move_row,last_move_column)
        B_W_postition = crop_board(B_W_postition,crop_size,last_move_row,last_move_column)
        
        return x, B_W_postition, B_W_ration

    x = crop_board(x,crop_size,last_move_row,last_move_column)
    return x
#當前盤面表示 黑:1、白:-1、空:0  可以看是否需要提子
def prepare_board_status_liberties(moves,
                         remove_dead=True,
                         sperate_posotion=False,
                         empty_is_one = False,
                         last_move_ = False,
                         crop_size=19):
    '''
    `moves`:格式 [B[pq],W[qd],B[cp],W[dd],..]\n
    `remove_dead`:是否要提子\n
    `sperate_posotion`:是否回傳黑、白獨立的所在盤面\n
    `empty_is_one`:回傳的empty位置給予的值,True:代表為1(被佔領的位置=0),False:代表為0(被佔領的位置=1)\n
    `last_move`:是否為最後一手的位置\n
    `crop_size`:要切盤面的大小
    `Return`\n 
    `x`:最終盤面的狀況(黑=1、白=-1、空=0)
    `B_W_postition`:額外的資訊(sperate_posotion、empty_is_one、last_move)組成
    `B_W_ration`:黑白棋數量比例
    '''
    x = np.zeros((19,19)).astype(np.int8)
    last_move = np.zeros((19,19)).astype(np.int8)
    last_move_row = 0
    last_move_column = 0
    for idx,move in enumerate(moves):
        color = move[0]
        column = coordinates[move[2]]
        row = coordinates[move[3]]
        if color == 'B':
            x[row,column] = 1
        elif color == 'W':
            x[row,column] = -1            

        if remove_dead:
            remove_dead_stones(x, (row,column))
            # print(f'step{idx}',move )
            # show(x)
        if idx == len(moves)-1:
            last_move_row = row
            last_move_column = column
            last_move[last_move_row,last_move_column] = 1


    if sperate_posotion:
        #黑、白色各自佔據的位置(=1)、盤面為0的位置(=1)
        B_postion = np.where(x==1,1,0)
        W_postition = np.where(x==-1,1,0)
        liberties = calculate_liberties_for_each_position(x)#氣
        if empty_is_one:
            non_position = np.where(x==0,1,0)#被佔領的位置=0
        else:
            non_position = np.where(x==0,0,1)#被佔領的位置=1

        if last_move_:
            B_W_postition = np.stack([B_postion,W_postition,non_position,liberties,last_move],axis=0)
        else:
            B_W_postition = np.stack([B_postion,W_postition,non_position,liberties],axis=0)
        
        #黑白子數量差距比例  黑棋/白棋
        B_num = np.count_nonzero(B_postion)
        W_num = np.count_nonzero(W_postition)
        if W_num==0:
            B_W_ration = -1 #如果是剛開始的盤面 可能沒有白色棋
        else:
            B_W_ration = B_num/W_num

        x = crop_board(x,crop_size,last_move_row,last_move_column)
        B_W_postition = crop_board(B_W_postition,crop_size,last_move_row,last_move_column)
        
        return x, B_W_postition, B_W_ration

    x = crop_board(x,crop_size,last_move_row,last_move_column)
    return x

#當前盤面表示 黑:1、白:-1、空:0  可以看是否需要提子, crop sliding window
def prepare_board_status_aug(moves,
                         remove_dead=True,
                         sperate_posotion=False,
                         empty_is_one = False,
                         last_move_ = False,
                         crop_size=19,
                         crop_aug = 5):
    '''
    `moves`:格式 [B[pq],W[qd],B[cp],W[dd],..]\n
    `remove_dead`:是否要提子\n
    `sperate_posotion`:是否回傳黑、白獨立的所在盤面\n
    `empty_is_one`:回傳的empty位置給予的值,True:代表為1(被佔領的位置=0),False:代表為0(被佔領的位置=1)\n
    `last_move`:是否為最後一手的位置\n
    `crop_size`:要切盤面的大小
    `Return`\n 
    `x`:最終盤面的狀況(黑=1、白=-1、空=0)
    `B_W_postition`:額外的資訊(sperate_posotion、empty_is_one、last_move)組成
    `B_W_ration`:黑白棋數量比例
    '''
    x = np.zeros((19,19)).astype(np.int8)
    last_move = np.zeros((19,19)).astype(np.int8)
    last_move_row = 0
    last_move_column = 0
    for idx,move in enumerate(moves):
        color = move[0]
        column = coordinates[move[2]]
        row = coordinates[move[3]]
        if color == 'B':
            x[row,column] = 1
        elif color == 'W':
            x[row,column] = -1            

        if remove_dead:
            remove_dead_stones(x, (row,column))
            # print(f'step{idx}',move )
            # show(x)
        if idx == len(moves)-1:
            last_move_row = row
            last_move_column = column
            last_move[last_move_row,last_move_column] = 1


    if sperate_posotion:
        #黑、白色各自佔據的位置(=1)、盤面為0的位置(=1)
        B_postion = np.where(x==1,1,0)
        W_postition = np.where(x==-1,1,0)
        if empty_is_one:
            non_position = np.where(x==0,1,0)#被佔領的位置=0
        else:
            non_position = np.where(x==0,0,1)#被佔領的位置=1

        if last_move_:
            B_W_postition = np.stack([B_postion,W_postition,non_position,last_move],axis=0)
        else:
            B_W_postition = np.stack([B_postion,W_postition,non_position],axis=0)

        #黑白子數量差距比例  黑棋/白棋
        B_num = np.count_nonzero(B_postion)
        W_num = np.count_nonzero(W_postition)
        if W_num==0:
            B_W_ration = -1 #如果是剛開始的盤面 可能沒有白色棋
        else:
            B_W_ration = B_num/W_num

        x = crop_board_augmentation(x,crop_size,last_move_row,last_move_column,crop_aug)
        B_W_postition = crop_board_augmentation(B_W_postition,crop_size,last_move_row,last_move_column,crop_aug)
        
        return x, B_W_postition, B_W_ration

    x = crop_board_augmentation(x,crop_size,last_move_row,last_move_column)
    return x

#當前盤面表示 黑:1、白:-1、空:0  可以看是否需要提子, crop sliding window 只要8個方向
def prepare_board_status_aug_8_direction(moves,
                         remove_dead=True,
                         sperate_posotion=False,
                         empty_is_one = False,
                         last_move_ = False,
                         crop_size=19,
                         crop_aug = 9):
    '''
    `moves`:格式 [B[pq],W[qd],B[cp],W[dd],..]\n
    `remove_dead`:是否要提子\n
    `sperate_posotion`:是否回傳黑、白獨立的所在盤面\n
    `empty_is_one`:回傳的empty位置給予的值,True:代表為1(被佔領的位置=0),False:代表為0(被佔領的位置=1)\n
    `last_move`:是否為最後一手的位置\n
    `crop_size`:要切盤面的大小
    `Return`\n 
    `x`:最終盤面的狀況(黑=1、白=-1、空=0)
    `B_W_postition`:額外的資訊(sperate_posotion、empty_is_one、last_move)組成
    `B_W_ration`:黑白棋數量比例
    '''
    x = np.zeros((19,19)).astype(np.int8)
    last_move = np.zeros((19,19)).astype(np.int8)
    last_move_row = 0
    last_move_column = 0
    for idx,move in enumerate(moves):
        color = move[0]
        column = coordinates[move[2]]
        row = coordinates[move[3]]
        if color == 'B':
            x[row,column] = 1
        elif color == 'W':
            x[row,column] = -1            

        if remove_dead:
            remove_dead_stones(x, (row,column))
            # print(f'step{idx}',move )
            # show(x)
        if idx == len(moves)-1:
            last_move_row = row
            last_move_column = column
            last_move[last_move_row,last_move_column] = 1


    if sperate_posotion:
        #黑、白色各自佔據的位置(=1)、盤面為0的位置(=1)
        B_postion = np.where(x==1,1,0)
        W_postition = np.where(x==-1,1,0)
        if empty_is_one:
            non_position = np.where(x==0,1,0)#被佔領的位置=0
        else:
            non_position = np.where(x==0,0,1)#被佔領的位置=1

        if last_move_:
            B_W_postition = np.stack([B_postion,W_postition,non_position,last_move],axis=0)
        else:
            B_W_postition = np.stack([B_postion,W_postition,non_position],axis=0)

        #黑白子數量差距比例  黑棋/白棋
        B_num = np.count_nonzero(B_postion)
        W_num = np.count_nonzero(W_postition)
        if W_num==0:
            B_W_ration = -1 #如果是剛開始的盤面 可能沒有白色棋
        else:
            B_W_ration = B_num/W_num

        x = crop_board_around_augmentation(x,crop_size,last_move_row,last_move_column,crop_aug)
        B_W_postition = crop_board_around_augmentation(B_W_postition,crop_size,last_move_row,last_move_column,crop_aug)
        
        return x, B_W_postition, B_W_ration

    x = crop_board_around_augmentation(x,crop_size,last_move_row,last_move_column)
    return x
#當前盤面表示 黑:1、白:-1、空:0  可以看是否需要提子, 黑白相反
def prepare_board_status_BW_inverse(moves,
                         remove_dead=True,
                         sperate_posotion=False,
                         empty_is_one = False,
                         last_move_ = False,
                         crop_size=19,):
    '''
    `moves`:格式 [B[pq],W[qd],B[cp],W[dd],..]\n
    `remove_dead`:是否要提子\n
    `sperate_posotion`:是否回傳黑、白獨立的所在盤面\n
    `empty_is_one`:回傳的empty位置給予的值,True:代表為1(被佔領的位置=0),False:代表為0(被佔領的位置=1)\n
    `last_move`:是否為最後一手的位置\n
    `crop_size`:要切盤面的大小
    `Return`\n 
    `x`:最終盤面的狀況(黑=1、白=-1、空=0)
    `B_W_postition`:額外的資訊(sperate_posotion、empty_is_one、last_move)組成
    `B_W_ration`:黑白棋數量比例
    '''
    x = np.zeros((19,19)).astype(np.int8)
    last_move = np.zeros((19,19)).astype(np.int8)
    last_move_row = 0
    last_move_column = 0
    for idx,move in enumerate(moves):
        color = move[0]
        column = coordinates[move[2]]
        row = coordinates[move[3]]
        if color == 'B':
            x[row,column] = 1
        elif color == 'W':
            x[row,column] = -1            

        if remove_dead:
            remove_dead_stones(x, (row,column))
            # print(f'step{idx}',move )
            # show(x)
        if idx == len(moves)-1:
            last_move_row = row
            last_move_column = column
            last_move[last_move_row,last_move_column] = 1

    if sperate_posotion:
        #黑、白色各自佔據的位置(=1)、盤面為0的位置(=1)
        B_postion = np.where(x==1,1,0)
        W_postition = np.where(x==-1,1,0)
        if empty_is_one:
            non_position = np.where(x==0,1,0)#被佔領的位置=0
        else:
            non_position = np.where(x==0,0,1)#被佔領的位置=1

        if last_move_:
            B_W_postition = np.stack([B_postion,W_postition,non_position,last_move],axis=0)
            B_W_inverse_postition = np.stack([W_postition,B_postion,non_position,last_move],axis=0)
        else:
            B_W_postition = np.stack([B_postion,W_postition,non_position],axis=0)
            B_W_inverse_postition = np.stack([W_postition,B_postion,non_position],axis=0)
        B_W_postition_stack = np.stack([B_W_postition,B_W_inverse_postition],axis=0)

        #黑白子數量差距比例  黑棋/白棋
        B_num = np.count_nonzero(B_postion)
        W_num = np.count_nonzero(W_postition)
        if W_num==0:
            B_W_ration = -1 #如果是剛開始的盤面 可能沒有白色棋
        else:
            B_W_ration = B_num/W_num

        x = crop_board(x,crop_size,last_move_row,last_move_column)
        B_W_postition_stack = crop_board_BW_inverse(B_W_postition_stack,crop_size,last_move_row,last_move_column)
        
        return x, B_W_postition_stack, B_W_ration

    x = crop_board(x,crop_size,last_move_row,last_move_column)
    return x
#當前盤面表示 黑:1、白:-1、空:0  返回提子和不提子結果, 黑白相反
def prepare_board_status_BW_inverse_both(moves,
                         sperate_posotion=False,
                         empty_is_one = False,
                         last_move_ = False,
                         crop_size=19,):
    '''
    `moves`:格式 [B[pq],W[qd],B[cp],W[dd],..]\n
    `remove_dead`:是否要提子\n
    `sperate_posotion`:是否回傳黑、白獨立的所在盤面\n
    `empty_is_one`:回傳的empty位置給予的值,True:代表為1(被佔領的位置=0),False:代表為0(被佔領的位置=1)\n
    `last_move`:是否為最後一手的位置\n
    `crop_size`:要切盤面的大小
    `Return`\n 
    `x`:最終盤面的狀況(黑=1、白=-1、空=0)
    `B_W_postition`:額外的資訊(sperate_posotion、empty_is_one、last_move)組成
    `B_W_ration`:黑白棋數量比例
    '''
    x = np.zeros((19,19)).astype(np.int8)
    no_x = np.zeros((19,19)).astype(np.int8)
    last_move = np.zeros((19,19)).astype(np.int8)
    last_move_row = 0
    last_move_column = 0
    for idx,move in enumerate(moves):
        color = move[0]
        column = coordinates[move[2]]
        row = coordinates[move[3]]
        if color == 'B':
            x[row,column] = 1
            no_x[row,column] = 1
        elif color == 'W':
            x[row,column] = -1 
            no_x[row,column] = -1           

        
        x = remove_dead_stones(x, (row,column))

            # print(f'step{idx}',move )
            # show(x)
        if idx == len(moves)-1:
            last_move_row = row
            last_move_column = column
            last_move[last_move_row,last_move_column] = 1

    if sperate_posotion:
        #黑、白色各自佔據的位置(=1)、盤面為0的位置(=1)
        B_postion = np.where(x==1,1,0)
        W_postition = np.where(x==-1,1,0)
        no_B_postion = np.where(no_x==1,1,0)
        no_W_postion = np.where(no_x==-1,1,0)

        if empty_is_one:
            non_position = np.where(x==0,1,0)#被佔領的位置=0
            no_non_position = np.where(no_x==0,1,0)
        else:
            non_position = np.where(x==0,0,1)#被佔領的位置=1
            no_non_position = np.where(no_x==0,0,1)

        if last_move_:
            B_W_postition = np.stack([B_postion,W_postition,non_position,last_move],axis=0)
            B_W_inverse_postition = np.stack([W_postition,B_postion,non_position,last_move],axis=0)
            no_B_W_postition = np.stack([no_B_postion,no_W_postion,no_non_position,last_move],axis=0)
            no_B_W_inverse_postition = np.stack([no_W_postion,no_B_postion,no_non_position,last_move],axis=0)
        else:
            B_W_postition = np.stack([B_postion,W_postition,non_position],axis=0)
            B_W_inverse_postition = np.stack([W_postition,B_postion,non_position],axis=0)
            no_B_W_postition = np.stack([no_B_postion,no_W_postion,no_non_position],axis=0)
            no_B_W_inverse_postition = np.stack([no_W_postion,no_B_postion,no_non_position],axis=0)
            
        B_W_postition_stack = np.stack([B_W_postition,B_W_inverse_postition,no_B_W_postition,no_B_W_inverse_postition],axis=0)

        #黑白子數量差距比例  黑棋/白棋
        B_num = np.count_nonzero(B_postion)
        W_num = np.count_nonzero(W_postition)
        if W_num==0:
            B_W_ration = -1 #如果是剛開始的盤面 可能沒有白色棋
        else:
            B_W_ration = B_num/W_num

        x = crop_board(x,crop_size,last_move_row,last_move_column)
        B_W_postition_stack = crop_board_BW_inverse(B_W_postition_stack,crop_size,last_move_row,last_move_column)
        
        return x, B_W_postition_stack, B_W_ration

    x = crop_board(x,crop_size,last_move_row,last_move_column)
    return x


#當前盤面表示 黑:1、白:-1、空:0  前8步
def prepare_board_status_previous8(moves,
                         remove_dead=True,
                         sperate_posotion=False,
                         empty_is_one = False,
                         last_move_ = False,
                         crop_size=19):
    '''
    `moves`:格式 [B[pq],W[qd],B[cp],W[dd],..]\n
    `remove_dead`:是否要提子\n
    `sperate_posotion`:是否回傳黑、白獨立的所在盤面\n
    `empty_is_one`:回傳的empty位置給予的值,True:代表為1(被佔領的位置=0),False:代表為0(被佔領的位置=1)\n
    `last_move`:是否為最後一手的位置\n
    `crop_size`:要切盤面的大小
    `Return`\n 
    `x`:最終盤面的狀況(黑=1、白=-1、空=0)
    `B_W_postition`:額外的資訊(sperate_posotion、empty_is_one、last_move)組成
    '''
    x = np.zeros((19,19)).astype(np.int8)
    last_move = np.zeros((19,19)).astype(np.int8)
    previous_B = np.zeros((8,19,19)).astype(np.int8)
    previous_W = np.zeros((8,19,19)).astype(np.int8)
    total_move = len(moves)
    last_move_row = 0
    last_move_column = 0
    for idx,move in enumerate(moves):
        color = move[0]
        column = coordinates[move[2]]
        row = coordinates[move[3]]
        if color == 'B':
            x[row,column] = 1
        elif color == 'W':
            x[row,column] = -1
        
        previous_step_ = total_move - idx

        if previous_step_ <= 8:
            previous_B[previous_step_-1,:,:] =  np.where(x==1,1,0)
            previous_W[previous_step_-1,:,:] =  np.where(x==-1,1,0)

        if remove_dead:
            remove_dead_stones(x, (row,column))
            # print(f'step{idx}',move )
            # show(x)
        if idx == len(moves)-1:
            last_move_row = row
            last_move_column = column
            last_move[last_move_row,last_move_column] = 1


    if sperate_posotion:
        #黑、白色各自佔據的位置(=1)、盤面為0的位置(=1)
        B_postion = np.where(x==1,1,0)
        W_postition = np.where(x==-1,1,0)
        if empty_is_one:
            non_position = np.where(x==0,1,0)#被佔領的位置=0
        else:
            non_position = np.where(x==0,0,1)#被佔領的位置=1

        if last_move_:
            B_W_postition = np.stack([B_postion,W_postition,non_position,last_move],axis=0)
        else:
            B_W_postition = np.stack([B_postion,W_postition,non_position],axis=0)

        #黑白子數量差距比例  黑棋/白棋
        B_num = np.count_nonzero(B_postion)
        W_num = np.count_nonzero(W_postition)
        if W_num==0:
            B_W_ration = -1 #如果是剛開始的盤面 可能沒有白色棋
        else:
            B_W_ration = B_num/W_num

        # print(np.shape(previous_B))
        B_W_postition =  np.concatenate([B_W_postition,previous_B,previous_W],axis=0)
        # print(np.shape(B_W_postition))
        x = crop_board(x,crop_size,last_move_row,last_move_column)
        B_W_postition = crop_board(B_W_postition,crop_size,last_move_row,last_move_column)
        
        return x, B_W_postition, B_W_ration

    x = crop_board(x,crop_size,last_move_row,last_move_column)
    return x
#當前盤面表示 黑:1、白:-1、空:0  前8步
def prepare_board_status_BW_inverse_previous8(moves,
                         remove_dead=True,
                         sperate_posotion=False,
                         empty_is_one = False,
                         last_move_ = False,
                         crop_size=19):
    '''
    `moves`:格式 [B[pq],W[qd],B[cp],W[dd],..]\n
    `remove_dead`:是否要提子\n
    `sperate_posotion`:是否回傳黑、白獨立的所在盤面\n
    `empty_is_one`:回傳的empty位置給予的值,True:代表為1(被佔領的位置=0),False:代表為0(被佔領的位置=1)\n
    `last_move`:是否為最後一手的位置\n
    `crop_size`:要切盤面的大小
    `Return`\n 
    `x`:最終盤面的狀況(黑=1、白=-1、空=0)
    `B_W_postition`:額外的資訊(sperate_posotion、empty_is_one、last_move)組成
    '''
    x = np.zeros((19,19)).astype(np.int8)
    last_move = np.zeros((19,19)).astype(np.int8)
    previous_B = np.zeros((8,19,19)).astype(np.int8)
    previous_W = np.zeros((8,19,19)).astype(np.int8)
    total_move = len(moves)
    last_move_row = 0
    last_move_column = 0
    for idx,move in enumerate(moves):
        color = move[0]
        column = coordinates[move[2]]
        row = coordinates[move[3]]
        if color == 'B':
            x[row,column] = 1
        elif color == 'W':
            x[row,column] = -1
        
        previous_step_ = total_move - idx

        if previous_step_ <= 8:
            previous_B[previous_step_-1,:,:] =  np.where(x==1,1,0)
            previous_W[previous_step_-1,:,:] =  np.where(x==-1,1,0)

        if remove_dead:
            remove_dead_stones(x, (row,column))
            # print(f'step{idx}',move )
            # show(x)
        if idx == len(moves)-1:
            last_move_row = row
            last_move_column = column
            last_move[last_move_row,last_move_column] = 1


    if sperate_posotion:
        #黑、白色各自佔據的位置(=1)、盤面為0的位置(=1)
        B_postion = np.where(x==1,1,0)
        W_postition = np.where(x==-1,1,0)
        if empty_is_one:
            non_position = np.where(x==0,1,0)#被佔領的位置=0
        else:
            non_position = np.where(x==0,0,1)#被佔領的位置=1

        if last_move_:
            B_W_postition = np.stack([B_postion,W_postition,non_position,last_move],axis=0)
            B_W_inverse_postition = np.stack([W_postition,B_postion,non_position,last_move],axis=0)
        else:
            B_W_postition = np.stack([B_postion,W_postition,non_position],axis=0)
            B_W_inverse_postition = np.stack([W_postition,B_postion,non_position],axis=0)
        
        B_W_postition =  np.concatenate([B_W_postition,previous_B,previous_W],axis=0)
        B_W_inverse_postition = np.concatenate([B_W_inverse_postition,previous_B,previous_W],axis=0)
        B_W_postition_stack = np.stack([B_W_postition,B_W_inverse_postition],axis=0)
        
        #黑白子數量差距比例  黑棋/白棋
        B_num = np.count_nonzero(B_postion)
        W_num = np.count_nonzero(W_postition)
        if W_num==0:
            B_W_ration = -1 #如果是剛開始的盤面 可能沒有白色棋
        else:
            B_W_ration = B_num/W_num

        
       
        x = crop_board(x,crop_size,last_move_row,last_move_column)
        B_W_postition_stack = crop_board_BW_inverse(B_W_postition_stack,crop_size,last_move_row,last_move_column)

        return x, B_W_postition_stack, B_W_ration

    x = crop_board(x,crop_size,last_move_row,last_move_column)
    return x

#當前盤面表示 黑:1、白:-1、空:0、seqence  可以看是否需要提子
def prepare_board_status_with_sequence(moves,
                         remove_dead=True,
                         sperate_posotion=False,
                         empty_is_one = False,
                         last_move_ = False,
                         crop_size=19,
                         record_seq = np.inf):
    '''
    `moves`:格式 [B[pq],W[qd],B[cp],W[dd],..]\n
    `remove_dead`:是否要提子\n
    `sperate_posotion`:是否回傳黑、白獨立的所在盤面\n
    `empty_is_one`:回傳的empty位置給予的值,True:代表為1(被佔領的位置=0),False:代表為0(被佔領的位置=1)\n
    `last_move`:是否為最後一手的位置\n
    `crop_size`:要切盤面的大小
    `record_seq`:要記錄倒數多少手順序, np.inf代表全部都要記錄
    `Return`\n 
    `x`:最終盤面的狀況(黑=1、白=-1、空=0)
    `B_W_postition`:額外的資訊(sperate_posotion、empty_is_one、last_move)組成
    `seq`:黑白棋個別落子順序
    '''
    x = np.zeros((19,19)).astype(np.int8)
    seq =  np.zeros((16,19,19)).astype(np.int8)
    last_move = np.zeros((19,19)).astype(np.int8)
    last_move_row = 0
    last_move_column = 0
    counter_main = 0
    counter_enemy = 0
    total_len_B = len([x for x in moves if x[0]=='B'])
    total_len_W = len([x for x in moves if x[0]=='W'])

    for idx,move in enumerate(moves):
        color = move[0]
        column = coordinates[move[2]]
        row = coordinates[move[3]]           

        if remove_dead:
            remove_dead_stones(x, (row,column))
            # print(f'step{idx}',move )
            # show(x)
        
        if color == 'B':
            counter_main+= 1
            if total_len_B - counter_main <= record_seq:
                #紀錄下棋順序 idx=0:channl0、1:chennel1、2:channel0+1
                #因為只紀錄 黑or白的第幾手  所以會減半
                binary_idx = bin(counter_main)[2:]       
                binary_idx_inverse = binary_idx[::-1]#反轉字串 迴圈比較好做
                for channel,carry in enumerate(binary_idx_inverse):
                    if carry == '1':
                        seq[channel, row,column] = 1
        else:
            counter_enemy+= 1
            if total_len_W - counter_enemy <= record_seq:
                binary_idx = bin(counter_main)[2:]       
                binary_idx_inverse = binary_idx[::-1]
                for channel,carry in enumerate(binary_idx_inverse):
                    if carry == '1':
                        seq[channel+8, row,column] = 1
        
        if color == 'B':
            x[row,column] = 1
        elif color == 'W':
            x[row,column] = -1 

        if idx == len(moves)-1:
            last_move_row = row
            last_move_column = column
            last_move[last_move_row,last_move_column] = 1


    if sperate_posotion:
        #黑、白色各自佔據的位置(=1)、盤面為0的位置(=1)
        B_postion = np.where(x==1,1,0)
        W_postition = np.where(x==-1,1,0)
        if empty_is_one:
            non_position = np.where(x==0,1,0)#被佔領的位置=0
        else:
            non_position = np.where(x==0,0,1)#被佔領的位置=1

        if last_move_:
            B_W_postition = np.stack([B_postion,W_postition,non_position,last_move],axis=0)
        else:
            B_W_postition = np.stack([B_postion,W_postition,non_position],axis=0)

        x = crop_board(x,crop_size,last_move_row,last_move_column)
        B_W_postition = crop_board(B_W_postition,crop_size,last_move_row,last_move_column)
        seq = crop_board(seq,crop_size,last_move_row,last_move_column)
        
        return x, B_W_postition, seq

    x = crop_board(x,crop_size,last_move_row,last_move_column)
    return x

def show(board:np.array):
    if len(board.shape) == 3:
        board = np.squeeze(board,axis=0)

    board = board.tolist()  
    for idx,row in enumerate(board):
        print('第{:2}行'.format(idx+1), end='')
        for col in row:
            print(' {:2}'.format(col), end='')
        print()
        

if __name__ == '__main__':
    seq = 'B[pq],W[qd],B[cp],W[dd],B[qo],W[eq],B[iq],W[cq],B[bq],W[br],B[dq],W[cr],B[dp],W[dr],B[bp],W[gq],B[cf],W[ce],B[df],W[ck],B[cm],W[ch],B[di],W[dh],B[ci],W[bi],B[bj],W[bg],B[ai],W[bh],B[cj],W[bf],B[dk],W[kc],B[ff],W[fd],B[oc],W[pc],B[od],W[qg],B[ic],W[ke],B[he],W[og],B[ld],W[kd],B[mf],W[kg],B[mh],W[ki],B[jf],W[kf],B[qe],W[re],B[pe],W[rc],B[mj],W[qj],B[pi],W[qi],B[ql],W[pk],B[lh],W[kh],B[fb],W[gc],B[gb],W[hc],B[hb],W[id],B[ed],W[ec],B[fc],W[ee],B[hd],W[gd],B[jd],W[ib],B[jc],W[jb],B[je],W[ef],B[lb],W[kb],B[ig],W[kq],B[mq],W[lq],B[mp],W[ko],B[io],W[km],B[ln],W[kn],B[ml],W[fn],B[im],W[kk],B[jl],W[kl],B[ii],W[lo],B[mo],W[lm],B[nm],W[mn],B[nn],W[ij],B[hj],W[ik],B[jj],W[jk],B[ji],W[gl],B[kj],W[el],B[fo],W[go],B[gn],W[fp],B[fm],W[hn],B[en],W[ho],B[hm],W[gm],B[hl],W[hk],B[gk],W[fl],B[in],W[fn],B[db],W[eb],B[gn],W[fk],B[gj],W[fn],B[ea],W[fa],B[gn],W[fj],B[fi],W[fn],B[ga],W[cb],B[gn],W[gi],B[hi],W[fn],B[dc],W[ed],B[gn],W[ei],B[gh],W[fn],B[cc],W[bb],B[gn],W[fh],B[ej],W[gi],B[hp],W[gp],B[fi],W[fn],B[eh],W[em],B[bc],W[ia],B[bd],W[ha],B[ab],W[ba],B[ad],W[da],B[ca],W[dn],B[da],W[fa],B[cn],W[rk],B[mr],W[lr],B[jr],W[hr],B[lp],W[kp],B[mm],W[hq],B[ip],W[ir],B[jq],W[ms],B[nr],W[ns],B[ks],W[ls],B[or],W[is]'
    seq = [x for x in seq.split(',')]

    
    print('提子前:')
    board_before,B_W_postition,B_W_ration = prepare_board_status(seq,
                                               remove_dead=False,
                                               sperate_posotion=True,
                                               empty_is_one=False,
                                               last_move_ = True)
    show(board_before)
    print('黑/白比例:',B_W_ration)
    print('B_W_postition shape:',np.shape(B_W_postition))

    print('提子後:')
    board,B_W_postition,B_W_ration = prepare_board_status(seq,
                                               remove_dead = True,
                                               sperate_posotion=True,
                                               empty_is_one=False,
                                               last_move_ = True,
                                               )
    show(board)
    print('黑/白比例:',B_W_ration)
    print('B_W_postition shape:',np.shape(B_W_postition))

    board,B_W_postition,_ = prepare_board_status_aug(seq,
                                               remove_dead = True,
                                               sperate_posotion=True,
                                               empty_is_one=False,
                                               last_move_ = True,
                                               crop_size = 5,
                                               )
    
    board,B_W_postition,_ = prepare_board_status_BW_inverse_previous8(seq,
                                               remove_dead = True,
                                               sperate_posotion=True,
                                               empty_is_one=False,
                                               last_move_ = True,
                                               crop_size = 5,
                                               )

    # board,B_W_postition_inverse,_ = prepare_board_status_BW_inverse(seq,
    #                                            remove_dead = False,
    #                                            sperate_posotion=True,
    #                                            empty_is_one=False,
    #                                            last_move_ = True,
    #                                            crop_size = 3,)
    # print('board:',board.shape)
    # print('B_W_postition_inverse:',B_W_postition_inverse.shape)
    # print('黑:\n',B_W_postition_inverse[0,0,:,:])
    # print('白:\n',B_W_postition_inverse[0,1,:,:])
    # print('invert黑:\n',B_W_postition_inverse[1,0,:,:])
    # print('invert白:\n',B_W_postition_inverse[1,1,:,:])

    # board,B_W_postition,seq = prepare_board_status_with_sequence(seq,
    #                                            remove_dead = True,
    #                                            sperate_posotion=True,
    #                                            empty_is_one=False,
    #                                            last_move_ = True,
    #                                            crop_size=19,
    #                                            record_seq = 8
    #                                            )
    # print(seq[:8,:,:])


    
    