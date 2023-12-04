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
from utils.preprocess import prepare_board_status, rand_augmentation
from models_block.inception_v4_only_reduceA import inception_resnet_v2_GO_V2

chars = 'abcdefghijklmnopqrst'
coordinates = {k:v for v,k in enumerate(chars)}

class test_dataloader(Dataset):
    def __init__(self, csv_path, crop_size):
        super(test_dataloader).__init__()
        #切資料
        csv = open(f'{csv_path}').read().splitlines()
        self.games = [i.split(',',2)[-1] for i in csv]
        self.games_id = [i.split(',')[0] for i in csv]
        self.crop_size = crop_size
        n_games = 0
        for game in self.games:
            n_games += 1
        print(f"Total Test Games: {len(self.games)}")

        self.n_games = n_games
        
    def __len__(self):
        return self.n_games

    def __getitem__(self, index):
        moves_list = self.games[index].split(',')
        moves_list = [g for g in moves_list if g != '']
        games_id = self.games_id[index]
        _,x,_ = prepare_board_status(moves_list,
                                        remove_dead=True,
                                        sperate_posotion=True,
                                        empty_is_one=False,
                                        last_move_ = True,
                                        crop_size = self.crop_size)
        x = rand_augmentation(x)
        x = np.array(x)
        return x, games_id
        
def inference(model, dataloader):
    pred_arr=[]
    pred_id=[]
    with Progress(TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                TimeElapsedColumn()) as progress:
        test_batch_tqdm = progress.add_task(description="Test progress", total=len(dataloader))
        model.eval()
        with torch.no_grad():
            for test_board, test_id in dataloader:
                test_board = test_board.to(DEVICE)
                test_board = test_board.view(-1, CHANNELS_1, CROP_SIZE, CROP_SIZE).to(torch.float32)

                test_pred = model(test_board)
                test_pred = torch.softmax(test_pred, dim=1)
                test_p = torch.argmax(test_pred, dim=1)

                pred_arr = np.append(pred_arr, test_p.detach().cpu().numpy())
                pred_id = np.append(pred_id, test_id)

                progress.advance(test_batch_tqdm, advance=1)
            progress.reset(test_batch_tqdm)
        return pred_arr, pred_id

if __name__ == '__main__':
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    CURRENT_PATH = os.path.dirname(__file__)
    BEFORE_PATH = os.path.abspath(os.path.join(CURRENT_PATH, os.path.pardir))
    MODEL_PATH = f'{BEFORE_PATH}/models_style/'
    public_test_path = f'{BEFORE_PATH}/submissions/'
    style_public_test_path = f'{BEFORE_PATH}/29_Public Testing Dataset_Public Submission Template_v2/29_Public Testing Dataset_v2/play_style_test_public.csv'
    style_private_test_path = f'{BEFORE_PATH}/29_Public Testing Dataset_Private Submission Template_v2/29_Private Testing Dataset_v2/play_style_test_private.csv'
    
    CROP_SIZE_LIST = [19]#要預測的model crop size
    CHANNELS_1 = 4
    NUM_CLASSES = 3
    BATCH_SIZE = 256
    PER_MODEL_PREDICT_NUM = 5   #要TTA多少次

    for test_csv in [style_public_test_path, style_private_test_path]:
        keep_info = {1:0,2:0,3:0}
        all_pred_arr = []
        all_pred_id = []
        #每個model要predict多少次
        for CROP_SIZE in CROP_SIZE_LIST:
            for per_fold in range(10):
                FOLD = per_fold
                Style_test_dataloader = test_dataloader(csv_path=test_csv,crop_size=CROP_SIZE)
                Style_test_dataloader = DataLoader(Style_test_dataloader, batch_size=BATCH_SIZE, num_workers=5)
                MODEL_NAME = f'Style_model21_10fold_channel{CHANNELS_1}_{CROP_SIZE}_fold{FOLD}_valloss_best'
                print(f'Loading {MODEL_NAME}')
                model = inception_resnet_v2_GO_V2(in_channels=CHANNELS_1, num_classes=NUM_CLASSES)
                model = model.to(DEVICE)
                model.load_state_dict(torch.load(f'{MODEL_PATH}/{MODEL_NAME}.pth'))
                for t in range(PER_MODEL_PREDICT_NUM):
                    pred_arr, pred_id = inference(model=model,dataloader=Style_test_dataloader)
                    all_pred_arr.append(pred_arr)
                    all_pred_id = pred_id
                del model

        for idx, per_pred_id in enumerate(pred_id):
            all_ans_arr = []
            for i in range(len(all_pred_arr)):
                ans = all_pred_arr[i][idx]
                all_ans_arr.append(ans)
            all_ans_arr = np.array(all_ans_arr).astype(np.int8)
            occur_count =  np.bincount(all_ans_arr)#取得每個label的出現數量從0開始 ex:[0,1,1,2] => [1,2,1]
            # 找到出現次數最多的數值
            max_count = np.max(occur_count)
            most_frequent_values = np.where(occur_count == max_count)[0]
            # 有可能有1種以上次數出現相同
            
            if len(most_frequent_values) >= 2:
                most_occur = most_frequent_values[0]
                for v in most_frequent_values:
                    p = np.random.rand(1)
                    if p>=0.5:
                        most_occur = v
            else:
                most_occur = most_frequent_values[0]
            
            df = pd.DataFrame({'name' : per_pred_id, '0' : int(most_occur+1)}, index=[0])
            df.to_csv(f'{public_test_path}/Style_model21_10fold_valloss_ensemble_TTA_{PER_MODEL_PREDICT_NUM}_{[x for x in CROP_SIZE_LIST]}.csv', mode='a', index=False, header=False)
            keep_info[most_occur+1] += 1
        
        print('棋風分布:',keep_info)
    


