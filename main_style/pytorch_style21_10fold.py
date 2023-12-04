import torch, os
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from torchsummaryX import summary
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from utils.preprocess import prepare_board_status_BW_inverse, prepare_board_status, augmentation_crop_aug
from models_block.inception_v4_only_reduceA import inception_resnet_v2_GO_V2
import logging

chars = 'abcdefghijklmnopqrst'
coordinates = {k:v for v,k in enumerate(chars)}

class Go_Dataset(Dataset):
    def __init__(self, csv_path, data_type, split_ratio = 0.2, crop_size=19):
        super(Go_Dataset).__init__()
        ###切資料
        csv = open(f'{csv_path}').read().splitlines()
        self.games = [i.split(',',2)[-1] for i in csv]
        self.game_styles = [int(i.split(',',2)[-2]) for i in csv]
        self.data_type = data_type
        self.crop_size = crop_size
        
        #shuffle
        self.games = np.array(self.games)
        self.game_styles = np.array(self.game_styles)

        np.random.seed(6670)
        all_styles_train_games = []
        all_styles_train_games_label = []
        all_styles_valid_games = []
        all_styles_valid_games_label = []
        
        #分成10fold
        keep_each_classes_fold = []
        keep_each_classes_label_fold = []
        #依照3個label分別取固定比例當valid
        for i in range(1,4):
            tmp_indx = np.where(self.game_styles==i)[0]
            tmp_games = self.games[tmp_indx]
            tmp_len = len(tmp_indx)
            idx = np.array([i for i in range(tmp_len)])
            np.random.shuffle(idx)
            tmp_games = tmp_games[idx]
            split_index = int(split_ratio*tmp_len)
            tmp_fold = [tmp_games[i:i + split_index] for i in range(0, len(tmp_games), split_index)]
            if len(tmp_fold) > 10 :
                tmp_fold[-2] = np.append(tmp_fold[-2], tmp_fold[-1])
                tmp_fold = tmp_fold[:-1]
            keep_each_classes_fold.append(tmp_fold)
            #label
            tmp_label = []
            for per_fold in tmp_fold:  
                tmp_label.append([i for x in range(len(per_fold))])
            keep_each_classes_label_fold.append(tmp_label)

        
        merge_each_classes_fold = []
        merge_each_classes_label_fold = []
        for per_fold in range(10):
            tmp_arr = np.array([])
            tmp_label = np.array([])
            for cls in range(3):
                tmp_arr = np.append(tmp_arr,keep_each_classes_fold[cls][per_fold])
                tmp_label = np.append(tmp_label,keep_each_classes_label_fold[cls][per_fold])
            merge_each_classes_fold.append(tmp_arr)
            merge_each_classes_label_fold.append(tmp_label)

        for per_fold in range(10):
            if per_fold == FOLD:
                tmp_valid = merge_each_classes_fold[per_fold].tolist()
                all_styles_valid_games += tmp_valid
                tmp_valid_label = merge_each_classes_label_fold[per_fold].tolist()
                all_styles_valid_games_label += tmp_valid_label
            else:
                tmp_train = merge_each_classes_fold[per_fold].tolist()
                all_styles_train_games += tmp_train
                tmp_train_label = merge_each_classes_label_fold[per_fold].tolist()
                all_styles_train_games_label += tmp_train_label


        #重新打亂
        all_styles_train_games = np.array(all_styles_train_games)
        all_styles_train_games_label = np.array(all_styles_train_games_label)
        train_idx = np.array([i for i in range(len(all_styles_train_games))])
        np.random.shuffle(train_idx)
        all_styles_train_games = all_styles_train_games[train_idx]
        all_styles_train_games_label = all_styles_train_games_label[train_idx]

        all_styles_valid_games = np.array(all_styles_valid_games)
        all_styles_valid_games_label = np.array(all_styles_valid_games_label)
        valid_idx = np.array([i for i in range(len(all_styles_valid_games))])
        np.random.shuffle(valid_idx)
        all_styles_valid_games = all_styles_valid_games[valid_idx]
        all_styles_valid_games_label = all_styles_valid_games_label[valid_idx]

        if data_type == 'train':
            self.train_games = all_styles_train_games.tolist()
            self.train_game_styles = all_styles_train_games_label.tolist()
            n_games = 0
            for game in self.train_games:
                n_games += 1
            print(f"Total Train Games: {len(self.train_games)}")

        elif data_type == 'valid':
            self.valid_games = all_styles_valid_games.tolist()
            self.valid_game_styles = all_styles_valid_games_label.tolist()
            n_games = 0
            for game in self.valid_games:
                n_games += 1
            print(f"Total Valid Games: {len(self.valid_games)}")
            
        self.n_games = n_games
        
    def __len__(self):
        return self.n_games

    def __getitem__(self, index):
        x, x2, y  = [], [], []

        if self.data_type == 'train':
            moves_list = self.train_games[index].split(',')
            styles = self.train_game_styles[index]
            target_color = moves_list[-1][0]#最後一手是哪個顏色下的
            _,x,_ = prepare_board_status(moves_list,
                                        remove_dead=True,
                                        sperate_posotion=True,
                                        empty_is_one=False,
                                        last_move_ = True,
                                        crop_size = self.crop_size)
            y.append(styles-1)
            y = np.array(y)

            return x ,y
        
        elif self.data_type == 'valid':
        
            moves_list = self.valid_games[index].split(',')
            styles = self.valid_game_styles[index]
            target_color = moves_list[-1][0]#最後一手是哪個顏色下的
            _,x,_ = prepare_board_status(moves_list,
                                        remove_dead=True,
                                        sperate_posotion=True,
                                        empty_is_one=False,
                                        last_move_ = True,
                                        crop_size = self.crop_size)
            y.append(styles-1)
            y = np.array(y)

            return x,y

class Go_Dataset_aug(Dataset):
    def __init__(self, csv_path, data_type, split_ratio = 0.2, crop_size = 19):
        super(Go_Dataset_aug).__init__()
        ###切資料
        csv = open(f'{csv_path}').read().splitlines()
        self.games = [i.split(',',2)[-1] for i in csv]
        self.game_styles = [int(i.split(',',2)[-2]) for i in csv]
        self.data_type = data_type
        self.crop_size = crop_size
        
        #shuffle
        self.games = np.array(self.games)
        self.game_styles = np.array(self.game_styles)

        np.random.seed(6670)
        all_styles_train_games = []
        all_styles_train_games_label = []
        all_styles_valid_games = []
        all_styles_valid_games_label = []
        
        #分成5fold
        keep_each_classes_fold = []
        keep_each_classes_label_fold = []
        #依照3個label分別取固定比例當valid
        for i in range(1,4):
            tmp_indx = np.where(self.game_styles==i)[0]
            tmp_games = self.games[tmp_indx]
            tmp_len = len(tmp_indx)
            idx = np.array([i for i in range(tmp_len)])
            np.random.shuffle(idx)
            tmp_games = tmp_games[idx]
            split_index = int(split_ratio*tmp_len)
            tmp_fold = [tmp_games[i:i + split_index] for i in range(0, len(tmp_games), split_index)]
            if len(tmp_fold) > 10 :
                tmp_fold[-2] = np.append(tmp_fold[-2], tmp_fold[-1])
                tmp_fold = tmp_fold[:-1]
            keep_each_classes_fold.append(tmp_fold)
            #label
            tmp_label = []
            for per_fold in tmp_fold:  
                tmp_label.append([i for x in range(len(per_fold))])
            keep_each_classes_label_fold.append(tmp_label)

        
        merge_each_classes_fold = []
        merge_each_classes_label_fold = []
        for per_fold in range(10):
            tmp_arr = np.array([])
            tmp_label = np.array([])
            for cls in range(3):
                tmp_arr = np.append(tmp_arr,keep_each_classes_fold[cls][per_fold])
                tmp_label = np.append(tmp_label,keep_each_classes_label_fold[cls][per_fold])
            merge_each_classes_fold.append(tmp_arr)
            merge_each_classes_label_fold.append(tmp_label)

        for per_fold in range(10):
            if per_fold == FOLD:
                tmp_valid = merge_each_classes_fold[per_fold].tolist()
                all_styles_valid_games += tmp_valid
                tmp_valid_label = merge_each_classes_label_fold[per_fold].tolist()
                all_styles_valid_games_label += tmp_valid_label
            else:
                tmp_train = merge_each_classes_fold[per_fold].tolist()
                all_styles_train_games += tmp_train
                tmp_train_label = merge_each_classes_label_fold[per_fold].tolist()
                all_styles_train_games_label += tmp_train_label

        #重新打亂
        all_styles_train_games = np.array(all_styles_train_games)
        all_styles_train_games_label = np.array(all_styles_train_games_label)
        train_idx = np.array([i for i in range(len(all_styles_train_games))])
        np.random.shuffle(train_idx)
        all_styles_train_games = all_styles_train_games[train_idx]
        all_styles_train_games_label = all_styles_train_games_label[train_idx]

        all_styles_valid_games = np.array(all_styles_valid_games)
        all_styles_valid_games_label = np.array(all_styles_valid_games_label)
        valid_idx = np.array([i for i in range(len(all_styles_valid_games))])
        np.random.shuffle(valid_idx)
        all_styles_valid_games = all_styles_valid_games[valid_idx]
        all_styles_valid_games_label = all_styles_valid_games_label[valid_idx]

        if data_type == 'train':
            self.train_games = all_styles_train_games.tolist()
            self.train_game_styles = all_styles_train_games_label.tolist()
            n_games = 0
            for game in self.train_games:
                n_games += 1
            print(f"Total Train Games: {len(self.train_games)}")

        elif data_type == 'valid':
            self.valid_games = all_styles_valid_games.tolist()
            self.valid_game_styles = all_styles_valid_games_label.tolist()
            n_games = 0
            for game in self.valid_games:
                n_games += 1
            print(f"Total Valid Games: {len(self.valid_games)}")
            
        self.n_games = n_games
        
    def __len__(self):
        return self.n_games

    def __getitem__(self, index):
        x, x2, y  = [], [], []

        if self.data_type == 'train':
            moves_list = self.train_games[index].split(',')
            styles = self.train_game_styles[index]
            target_color = moves_list[-1][0]#最後一手是哪個顏色下的
            _,x,_ = prepare_board_status_BW_inverse(moves_list,
                                        remove_dead=True,
                                        sperate_posotion=True,
                                        empty_is_one=False,
                                        last_move_ = True,
                                        crop_size = self.crop_size)
            x = augmentation_crop_aug(x)
            y = [styles-1 for x in range(8*2)]
            x = np.array(x)
            y = np.array(y)

            return x, y
        
        elif self.data_type == 'valid':
        
            moves_list = self.valid_games[index].split(',')
            styles = self.valid_game_styles[index]
            target_color = moves_list[-1][0]#最後一手是哪個顏色下的
            _,x,_ = prepare_board_status_BW_inverse(moves_list,
                                        remove_dead=True,
                                        sperate_posotion=True,
                                        empty_is_one=False,
                                        last_move_ = True,
                                        crop_size = self.crop_size)
            x = augmentation_crop_aug(x)
            y = [styles-1 for x in range(8*2)]
            x = np.array(x)
            y = np.array(y)

            return x, y

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
            for board, label in train_dataLoader:
                model.train()

                board = board.to(DEVICE)
                label = label.to(DEVICE)

                board = board.view(-1, CHANNELS_1, CROP_SIZE, CROP_SIZE).to(torch.float32)
                label = label.view(-1, 1).squeeze(1).to(torch.long)

                pred = model(board)
                
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
                for val_board, val_label in val_dataLoader:
                    
                    val_board = val_board.to(DEVICE)
                    val_label = val_label.to(DEVICE)
     
                    val_board = val_board.view(-1, CHANNELS_1, CROP_SIZE, CROP_SIZE).to(torch.float32)                    
                    val_label = val_label.view(-1, 1).squeeze(1).to(torch.long)

                    val_pred = model(val_board)

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
                # path = model_path+f'Dan_model_SimpleMLP_acc_{train_avg_acc}_loss_{train_avg_loss}_epochs_{num_epochs}.pth'
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
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    CURRENT_PATH = os.path.dirname(__file__)
    BEFORE_PATH = os.path.abspath(os.path.join(CURRENT_PATH, os.path.pardir))
    CROP_SIZE = 19
    BATCH_SIZE = 64
    EPOCHS = 10000
    NUM_CLASSES = 3
    LEARNING_RATE = 0.001
    CHANNELS_1 = 4
    LOSS = nn.CrossEntropyLoss() 

    # 配置 Logger
    for per_fold in range(10):
        FOLD = per_fold
        MODEL_PATH = f'{BEFORE_PATH}/models_style/'
        MODEL_NAME = f'Style_model21_10fold_channel{CHANNELS_1}_{CROP_SIZE}_fold{FOLD}_valloss'

        logging.basicConfig(level=logging.INFO)  # 配置info
        LOGGER = logging.getLogger(f'{MODEL_NAME}')  #建立一個叫做(f'{MODEL_NAME}')的記錄器
        LOG_FILE = f'{MODEL_PATH}/{MODEL_NAME}.log' #記錄檔案名稱
        file_handler = logging.FileHandler(LOG_FILE)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        # FileHandler 添加到 Logger
        LOGGER.addHandler(file_handler)
        
        Style_csv_path = f'{BEFORE_PATH}/29_Training Dataset/Training Dataset/play_style_train.csv'
        Style_train_dataloader = Go_Dataset_aug(csv_path=Style_csv_path, data_type='train', crop_size=CROP_SIZE, split_ratio=0.1)
        Style_valid_dataloader = Go_Dataset(csv_path=Style_csv_path, data_type='valid', crop_size=CROP_SIZE, split_ratio=0.1)
        Style_train_dataloader = DataLoader(Style_train_dataloader, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        Style_valid_dataloader = DataLoader(Style_valid_dataloader, batch_size=BATCH_SIZE, shuffle=True)

        model = inception_resnet_v2_GO_V2(in_channels=CHANNELS_1, num_classes=NUM_CLASSES).to(DEVICE)
        summary(model, torch.randn(1, CHANNELS_1, CROP_SIZE, CROP_SIZE).to(DEVICE))

        if os.path.exists(f'{MODEL_PATH}/{MODEL_NAME}_last.pth'):
            model.load_state_dict(torch.load(f'{MODEL_PATH}/{MODEL_NAME}_last.pth'))
            print('reload weight training...')
        
        opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        SCHEDULER = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=3, factor=0.5, verbose=True)
        train_loss ,train_acc ,val_loss ,val_acc = train(Style_train_dataloader, Style_valid_dataloader, model, loss_func=LOSS, opt=opt)
        plot_statistics(train_loss,train_acc,val_loss,val_acc,'acc',MODEL_PATH)
        del model

