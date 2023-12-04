import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from models_block.inception_v4_only_reduceA_with_seq import inception_resnet_v2_GO_V3
from torchsummaryX import summary
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
from utils.preprocess import prepare_board_status_with_sequence, augmentation_dan_kyu_2path, prepare_label_19x19
import logging

chars = 'abcdefghijklmnopqrst'
coordinates = {k:v for v,k in enumerate(chars)}

def prepare_label(move):
    column = coordinates[move[2]]
    row = coordinates[move[3]]
    return column*19+row

class dataloader(Dataset):
    def __init__(self, csv_path, data_type ,limit_size = 50, split_ratio = 0.2):
        super(dataloader).__init__()
        #切資料
        csv = open(f'{csv_path[0]}').read().splitlines()
        csv2 = open(f'{csv_path[1]}').read().splitlines()
        self.avoid_split_into_valid = set()
        with open(f'./valid_id.txt' , 'r' ) as fp:
            for ids in fp.readlines():
                self.avoid_split_into_valid.add(ids.strip('\n'))
        
        self.limit_size  = limit_size #因為每個棋盤會下多少手數量不同,所以限制大小
        self.data_type = data_type
        self.split_ratio = split_ratio

        dan_games, dan_colors, games_id = self.shuffle_avoid_set_data(csv=csv)
        kyu_games, kyu_colors, games_id2 = self.shuffle_avoid_set_data(csv=csv2)
        self.total_games = dan_games + kyu_games
        self.total_colors = dan_colors + kyu_colors
        self.games_id = games_id + games_id2

        print('self.total_games len:',np.shape(self.total_games))
        

        n_games = 0
        n_moves = 0
        for game in self.total_games:
            n_games += 1
            moves_list = game.split(',')
            for move in moves_list:
                n_moves += 1
        print(f"Total {data_type} Games: {n_games}, Total {data_type} Moves: {n_moves}")
        self.n_games = n_games
    
    def shuffle_avoid_set_data(self,csv):
        games = [i.split(',',2)[-1] for i in csv]
        colors = [i.split(',')[1] for i in csv]
        games_id = [i.split(',')[0] for i in csv]

        can_split_index = []
        avoid_split_index = []
        #先挑出 可以分到valid的 跟 不能分到valid的
        for idx, ids in enumerate(games_id):
            if ids in self.avoid_split_into_valid:
                avoid_split_index.append(idx)
            else:
                can_split_index.append(idx)
        
        can_split_index = np.array(can_split_index)
        avoid_split_index = np.array(avoid_split_index)

        games = np.array(games)
        colors = np.array(colors)
        games_id = np.array(games_id)

        can_games = games[can_split_index]
        can_colors = colors[can_split_index]
        can_games_id = games_id[can_split_index]

        avoid_games = games[avoid_split_index]
        avoid_colors = colors[avoid_split_index]
        avoid_games_id = games_id[avoid_split_index]

        print('原始長度:',len(games))
        print('只取可以分在valid中的長度:',len(can_games))
        print('不能出現在train中的長度:',len(avoid_games))
        
        #shuffle
        np.random.seed(6670)
        can_idx = np.array([i for i in range(len(can_games))])
        np.random.shuffle(can_idx)
        can_games = can_games[can_idx]
        can_colors = can_colors[can_idx]
        can_games_id = can_games_id[can_idx]
        split_index = len(avoid_games)
        if self.data_type == 'train':   
            train_games = can_games[split_index:].tolist() + avoid_games.tolist()
            train_colors = can_colors[split_index:].tolist() + avoid_colors.tolist()
            train_games_id = can_games_id[split_index:].tolist() + avoid_games_id.tolist()
            return train_games, train_colors, train_games_id
        elif self.data_type == 'valid':
            valid_games = can_games[:split_index].tolist()
            valid_colors = can_colors[:split_index].tolist()
            valid_games_id = can_games_id[:split_index].tolist()
            return valid_games, valid_colors, valid_games_id

    def __len__(self):
        return self.n_games

    def __getitem__(self, index):
        x = []
        seq = []
        y = []

        moves_list = self.total_games[index].split(',')
        color = self.total_colors[index]
        for count, move in enumerate(moves_list):
            color_move = move.split('[')[0]
            if color == color_move:
                _,process_board,seq_board = prepare_board_status_with_sequence(moves_list[:count],
                                    remove_dead=False,
                                    sperate_posotion=True,
                                    empty_is_one=True,
                                    last_move_ = True)
                x.append(process_board)
                seq.append(seq_board)
                y.append(prepare_label_19x19(moves_list[count]))
        x = np.array(x)
        seq = np.array(seq)
        y = np.array(y)

        idx = np.array([i for i in range(len(x))])
        np.random.shuffle(idx)
        x = x[idx]
        seq = seq[idx]
        y = y[idx] 
        
        x = x[:self.limit_size]
        seq = seq[:self.limit_size]
        y = y[:self.limit_size]
        
        aug_all_x = []
        aug_all_seq = []
        aug_all_y = []
        for per_x,per_seq, per_y in zip(x,seq, y):
            aug_x,aug_seq, aug_y = augmentation_dan_kyu_2path(per_x,per_seq,per_y)
            aug_all_x.append(aug_x)
            aug_all_seq.append(aug_seq)
            aug_all_y.append(aug_y)
        
        aug_all_x = np.array(aug_all_x)
        aug_all_seq = np.array(aug_all_seq)
        aug_all_y = np.array(aug_all_y)
        aug_all_x = aug_all_x.reshape(-1,4,19,19)
        aug_all_seq = aug_all_seq.reshape(-1,16,19,19)
        aug_all_y = aug_all_y.reshape(-1,1)


        return (aug_all_x,aug_all_seq), aug_all_y

def train(train_dataLoader, val_dataLoader, model, loss_func, opt):
    train_loss, train_acc = [], []
    val_loss, val_acc = [], []
    best_acc, best_loss = 0, 888888
    earlystop = 0

    with Progress(TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                TimeElapsedColumn()) as progress:
        
        epoch_tqdm = progress.add_task(description="Epoch progress", total=EPOCHS)
        train_batch_tqdm = progress.add_task(description="Train progress", total=len(train_dataLoader))
        val_batch_tqdm = progress.add_task(description="Val progress", total=len(val_dataLoader))
        for num_epochs in range(EPOCHS):
            train_correct, val_correct = 0, 0
            train_avg_acc, train_avg_loss = 0, 0
            val_avg_acc, val_avg_loss = 0, 0
            train_instance_num, val_instance_num = 0, 0
            for (board,seq), label in train_dataLoader:
                model.train()
                board = board.to(DEVICE)
                seq = seq.to(DEVICE)
                label = label.to(DEVICE)

                board = board.view(-1, CHANNELS_1, 19, 19).to(torch.float32)
                seq = seq.view(-1, 1, CHANNELS_2, 19, 19).to(torch.float32)
                label = label.view(-1, 1).squeeze(1).to(torch.long)
                
                # print(board.size())
                # print(label.size())
                
                pred = model(board,seq)
                
                loss = loss_func(pred, label)
                opt.zero_grad()
                loss.backward()
                opt.step()
                
                train_avg_loss += loss.item()
                pred = torch.softmax(pred, dim=1)
                train_p = torch.argmax(pred, dim=1)
                train_correct += torch.sum(train_p == label).item()
                train_instance_num += len(train_p)
                progress.advance(train_batch_tqdm, advance=1)
            ### Validation
            model.eval()
            with torch.no_grad():
                for (val_board,val_seq), val_label in val_dataLoader:
                    val_board = val_board.to(DEVICE)
                    val_seq = val_seq.to(DEVICE)
                    val_label = val_label.to(DEVICE)

                    val_board = val_board.view(-1, CHANNELS_1, 19, 19).to(torch.float32)
                    val_seq = val_seq.view(-1, 1, CHANNELS_2, 19, 19).to(torch.float32)
                    val_label = val_label.view(-1, 1).squeeze(1).to(torch.long)

                    val_pred = model(val_board,val_seq)

                    tmp_loss = loss_func(val_pred, val_label)
                    val_avg_loss += tmp_loss.item()
                    val_pred = torch.softmax(val_pred, dim=1)
                    val_p = torch.argmax(val_pred, dim=1)
                    val_correct += torch.sum(val_p == val_label).item()
                    val_instance_num += len(val_p)
                    progress.advance(val_batch_tqdm, advance=1)
            SCHEDULER.step(val_avg_loss)#動態調整LR
            train_avg_loss = train_avg_loss / len(train_dataLoader)
            train_loss.append(train_avg_loss)
            train_avg_acc = train_correct / train_instance_num
            train_acc.append(train_avg_acc)
            val_avg_loss = val_avg_loss / len(val_dataLoader)
            val_loss.append(val_avg_loss)
            val_avg_acc = val_correct / val_instance_num
            val_acc.append(val_avg_acc)

            LOGGER.info('Epoch {}: train acc : {} | train loss: {} | val acc : {} | val loss : {}'.format(num_epochs, train_avg_acc ,train_avg_loss, val_avg_acc, val_avg_loss)) #紀錄訓練資訊

            progress.reset(train_batch_tqdm)
            progress.reset(val_batch_tqdm)
            progress.advance(epoch_tqdm, advance=1)

            print('Epoch {}: train acc : {} | train loss: {} | val acc : {} | val loss : {}'.format(num_epochs, train_avg_acc ,train_avg_loss, val_avg_acc, val_avg_loss))
            print("Current model saving...")
            cur_path = MODEL_PATH+f'{MODEL_NAME}_last.pth'
            torch.save(model.state_dict(), cur_path)
            if best_loss > val_avg_loss:
                print("Best model saving...")
                best_loss = val_avg_loss
                best_path = MODEL_PATH+f'{MODEL_NAME}_best.pth'
                torch.save(model.state_dict(), best_path)
                earlystop = 0
            else:
                earlystop += 1
                print(f"EarlyStop times: {earlystop}")
                if earlystop >= 5:
                    print("Earlystop triggered!")
                    logging.shutdown()
                    return train_loss ,train_acc ,val_loss ,val_acc  
                    break

def plot_statistics(train_loss,
                    train_performance,
                    valid_loss,
                    valid_performance,
                    performance_name,
                    SAVE_MODELS_PATH):
    '''
    統計train、valid的loss、acc
    '''
    plt.figure()

    t_loss = plt.plot(train_loss)
    v_loss = plt.plot(valid_loss)
    
    plt.legend([t_loss,v_loss],
               labels=['train_loss',
                        'valid_loss'])

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(f'{SAVE_MODELS_PATH}/{MODEL_NAME}_loss',bbox_inches='tight')
    plt.figure()

    t_per = plt.plot(train_performance)
    v_per = plt.plot(valid_performance)

    plt.legend([t_per,v_per],
               labels=[f'train_{performance_name}',
                       f'valid_{performance_name}'])

    plt.xlabel("epoch")
    plt.ylabel(f"{performance_name}")
    plt.savefig(f'{SAVE_MODELS_PATH}/{MODEL_NAME}_performance',bbox_inches='tight')
    plt.figure()

if __name__ == '__main__':
    '''
    混合一起練
    '''
    CURRENT_PATH = os.path.dirname(__file__)
    BEFORE_PATH = os.path.abspath(os.path.join(CURRENT_PATH, os.path.pardir))
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    MODEL_PATH = f'{BEFORE_PATH}/models/'
    MODEL_NAME = f'Merge_model_CNN17_4_16channels_valloss_v2'
    EPOCHS = 10000
    LEARNING_RATE = 0.001
    LOSS = nn.CrossEntropyLoss() 
    NUM_CLASSES = 361
    BATCH_SIZE = 4
    CHANNELS_1 = 4
    CHANNELS_2 = 16
    # 配置 Logger
    logging.basicConfig(level=logging.INFO)  # 配置info
    LOGGER = logging.getLogger(f'{MODEL_NAME}')  #建立一個叫做(f'{MODEL_NAME}')的記錄器
    LOG_FILE = f'{MODEL_PATH}/{MODEL_NAME}.log' #記錄檔案名稱
    file_handler = logging.FileHandler(LOG_FILE)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    # FileHandler 添加到 Logger
    LOGGER.addHandler(file_handler)
    Dan_csv_path = f'{BEFORE_PATH}/29_Training Dataset/Training Dataset/dan_train.csv'
    Kyu_csv_path = f'{BEFORE_PATH}/29_Training Dataset/Training Dataset/kyu_train.csv'
    all_train_dataloader = dataloader(csv_path=[Dan_csv_path,Kyu_csv_path], data_type='train', split_ratio=0.07)
    all_valid_dataloader = dataloader(csv_path=[Dan_csv_path,Kyu_csv_path], data_type='valid', split_ratio=0.07)
    all_train_dataloader = DataLoader(all_train_dataloader, batch_size=BATCH_SIZE, shuffle=True,num_workers=5)
    all_valid_dataloader = DataLoader(all_valid_dataloader, batch_size=BATCH_SIZE, shuffle=True,num_workers=2)

    model = inception_resnet_v2_GO_V3(in_channels=CHANNELS_1, num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    summary(model, torch.rand(1,CHANNELS_1,19,19).to(DEVICE),torch.rand(1,1,CHANNELS_2,19,19).to(DEVICE))


    if os.path.exists(f'{MODEL_PATH}/{MODEL_NAME}_best.pth'):
        model.load_state_dict(torch.load(f'{MODEL_PATH}/{MODEL_NAME}_best.pth'))
        print('reload weight training...')

    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    SCHEDULER = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=2, factor=0.5, verbose=True)
    train_loss ,train_acc ,val_loss ,val_acc = train(all_train_dataloader, all_valid_dataloader, model, loss_func=LOSS, opt=opt)
    plot_statistics(train_loss,train_acc,val_loss,val_acc,'acc',MODEL_PATH)  

