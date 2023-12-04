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
from utils.preprocess import prepare_board_status_with_sequence
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
        
        self.limit_size  = limit_size #因為每個棋盤會下多少手數量不同,所以限制大小
        self.data_type = data_type
        self.split_ratio = split_ratio

        dan_games, dan_colors = self.shuffle_data(csv=csv)
        kyu_games, kyu_colors = self.shuffle_data(csv=csv2)
        self.total_games = dan_games + kyu_games
        self.total_colors = dan_colors + kyu_colors

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
        #shuffle
        games = np.array(games)
        colors = np.array(colors)
        np.random.seed(6670)
        idx = np.array([i for i in range(len(games))])
        np.random.shuffle(idx)
        games = games[idx]
        colors = colors[idx]
        split_index = int(self.split_ratio*len(games))

        if self.data_type == 'train':
            train_games = games[split_index:].tolist()
            train_colors = colors[split_index:].tolist()
            return train_games, train_colors
        elif self.data_type == 'valid':
            valid_games = games[:split_index].tolist()
            valid_colors = colors[:split_index].tolist()
            return valid_games, valid_colors

    def __len__(self):
        return self.n_games

    def __getitem__(self, index):
        x,seq = [],[]
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
                y.append(prepare_label(moves_list[count]))
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

        return (x, seq), y

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
    MODEL_NAME = f'Merge_model_CNN17_4_16channels_valloss'
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
    all_train_dataloader = DataLoader(all_train_dataloader, batch_size=BATCH_SIZE, shuffle=True)
    all_valid_dataloader = DataLoader(all_valid_dataloader, batch_size=BATCH_SIZE, shuffle=True)

    model = inception_resnet_v2_GO_V3(in_channels=CHANNELS_1, num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    # summary(model, torch.rand(1,CHANNELS_1,19,19).to(DEVICE),torch.rand(1,1,CHANNELS_2,19,19).to(DEVICE))

    if os.path.exists(f'{MODEL_PATH}/{MODEL_NAME}_last.pth'):
        model.load_state_dict(torch.load(f'{MODEL_PATH}/{MODEL_NAME}_last.pth'))
        print('reload weight training...')

    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    SCHEDULER = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=2, factor=0.5, verbose=True)
    train_loss ,train_acc ,val_loss ,val_acc = train(all_train_dataloader, all_valid_dataloader, model, loss_func=LOSS, opt=opt)
    plot_statistics(train_loss,train_acc,val_loss,val_acc,'acc',MODEL_PATH)  

