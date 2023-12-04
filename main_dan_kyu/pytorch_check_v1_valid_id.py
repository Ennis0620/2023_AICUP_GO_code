import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
import os
from utils.preprocess import prepare_board_status, augmentation_dan_kyu, prepare_label_19x19

class dataloader(Dataset):
    def __init__(self, csv_path, data_type ,limit_size = 50, split_ratio = 0.2):
        super(dataloader).__init__()
        #切資料
        csv = open(f'{csv_path[0]}').read().splitlines()
        csv2 = open(f'{csv_path[1]}').read().splitlines()
        
        self.limit_size  = limit_size #因為每個棋盤會下多少手數量不同,所以限制大小
        self.data_type = data_type
        self.split_ratio = split_ratio

        dan_games, dan_colors, games_id = self.shuffle_data(csv=csv)
        kyu_games, kyu_colors, games_id2 = self.shuffle_data(csv=csv2)
        self.total_games = dan_games + kyu_games
        self.total_colors = dan_colors + kyu_colors
        self.games_id = games_id + games_id2
        # for idx,i in enumerate(self.games_id):
        #     with open(f'{CURRENT_PATH}/valid_id.txt' , 'a+' ) as fp:
        #         if idx != len(self.games_id)-1:
        #             fp.write(i + '\n')
        #         else:
        #             fp.write(i)
        print(self.games_id)

        n_games = 0
        n_moves = 0
        for game in self.total_games:
            n_games += 1
            moves_list = game.split(',')
            for move in moves_list:
                n_moves += 1
        print(f"Total {data_type} Games: {n_games}, Total {data_type} Moves: {n_moves}")
        self.n_games = n_games
    
    def shuffle_data(self,csv):
        games = [i.split(',',2)[-1] for i in csv]
        colors = [i.split(',')[1] for i in csv]
        games_id = [i.split(',')[0] for i in csv]
        
        #shuffle
        games = np.array(games)
        colors = np.array(colors)
        games_id = np.array(games_id)
        np.random.seed(6670)
        idx = np.array([i for i in range(len(games))])
        np.random.shuffle(idx)
        games = games[idx]
        colors = colors[idx]
        games_id = games_id[idx]
        split_index = int(self.split_ratio*len(games))

        if self.data_type == 'train':
            train_games = games[split_index:].tolist()
            train_colors = colors[split_index:].tolist()
            train_games_id = games_id[split_index:].tolist()
            return train_games, train_colors, train_games_id
        elif self.data_type == 'valid':
            valid_games = games[:split_index].tolist()
            valid_colors = colors[:split_index].tolist()
            valid_games_id = games_id[:split_index].tolist()
            return valid_games, valid_colors, valid_games_id

    def __len__(self):
        return self.n_games

    def __getitem__(self, index):
        x = []
        y = []

        moves_list = self.total_games[index].split(',')
        color = self.total_colors[index]
        for count, move in enumerate(moves_list):
            color_move = move.split('[')[0]
            if color == color_move:
                _,process_board,_ = prepare_board_status(moves_list[:count],
                                    remove_dead=False,
                                    sperate_posotion=True,
                                    empty_is_one=True,
                                    last_move_ = True)
                x.append(process_board)
                y.append(prepare_label_19x19(moves_list[count]))
        x = np.array(x)
        y = np.array(y)

        idx = np.array([i for i in range(len(x))])
        np.random.shuffle(idx)
        x = x[idx]
        y = y[idx] 
        x = x[:self.limit_size]
        y = y[:self.limit_size]

        aug_all_x = []
        aug_all_y = []
        for per_x, per_y in zip(x, y):
            aug_x, aug_y = augmentation_dan_kyu(per_x,per_y)
            aug_all_x.append(aug_x)
            aug_all_y.append(aug_y)
        aug_all_x = np.array(aug_all_x)
        aug_all_y = np.array(aug_all_y)
        aug_all_x = aug_all_x.reshape(-1,4,19,19)
        aug_all_y = aug_all_y.reshape(-1,1)
        
        return aug_all_x, aug_all_y

if __name__ == '__main__':
    '''
    確認哪些在v1時被分到valid,將這些valid記錄下來 記錄到valid.txt
    '''
    CURRENT_PATH = os.path.dirname(__file__)
    BEFORE_PATH = os.path.abspath(os.path.join(CURRENT_PATH, os.path.pardir))
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 2

    Dan_csv_path = f'{BEFORE_PATH}/29_Training Dataset/Training Dataset/dan_train.csv'
    Kyu_csv_path = f'{BEFORE_PATH}/29_Training Dataset/Training Dataset/kyu_train.csv'
    # all_train_dataloader = dataloader(csv_path=[Dan_csv_path,Kyu_csv_path], data_type='train', split_ratio=0.07)
    all_valid_dataloader = dataloader(csv_path=[Dan_csv_path,Kyu_csv_path], data_type='valid', split_ratio=0.07)
    # all_train_dataloader = DataLoader(all_train_dataloader, batch_size=BATCH_SIZE, shuffle=True)
    all_valid_dataloader = DataLoader(all_valid_dataloader, batch_size=BATCH_SIZE, shuffle=True)
 

