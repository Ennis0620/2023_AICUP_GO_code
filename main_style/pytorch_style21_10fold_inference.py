import torch, os, csv
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.progress import track
from torch.utils.data import Dataset
from utils.preprocess import prepare_board_status
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
        x = []
        moves_list = self.games[index].split(',')
        moves_list = [g for g in moves_list if g != '']
        games_id = self.games_id[index]
        _,x,_ = prepare_board_status(moves_list,
                                        remove_dead=True,
                                        sperate_posotion=True,
                                        empty_is_one=False,
                                        last_move_ = True,
                                        crop_size = self.crop_size)
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

### Configs
if __name__ == "__main__":
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    CURRENT_PATH = os.path.dirname(__file__)
    BEFORE_PATH = os.path.abspath(os.path.join(CURRENT_PATH, os.path.pardir))
    MODEL_PATH = f'{BEFORE_PATH}/models_style/'
    public_test_path = f'{BEFORE_PATH}/submissions/'
    style_public_test_path = f'{BEFORE_PATH}/29_Public Testing Dataset_Public Submission Template_v2/29_Public Testing Dataset_v2/play_style_test_public.csv'
    style_private_test_path = f'{BEFORE_PATH}/29_Public Testing Dataset_Private Submission Template_v2/29_Private Testing Dataset_v2/play_style_test_private.csv'
    
    CROP_SIZE = 19
    CHANNELS_1 = 4
    BATCH_SIZE = 256
    NUM_CLASSES = 3
    FOLD = 0

    file_name = f'Style_model21_10fold_channel{CHANNELS_1}_{CROP_SIZE}_fold{FOLD}_valloss_submission.csv'
    MODEL_NAME = f'Style_model21_10fold_channel{CHANNELS_1}_{CROP_SIZE}_fold{FOLD}_valloss_best'
    
    for test_csv in [style_public_test_path, style_private_test_path]:
        keep_info = {1:0,2:0,3:0}
        model = inception_resnet_v2_GO_V2(in_channels=CHANNELS_1, num_classes=NUM_CLASSES)
        model = model.to(DEVICE)
        
        Style_test_dataloader = test_dataloader(csv_path=test_csv,crop_size=CROP_SIZE)
        Style_test_dataloader = DataLoader(Style_test_dataloader, batch_size=BATCH_SIZE, num_workers=5)
        model.load_state_dict(torch.load(f'{MODEL_PATH}/{MODEL_NAME}.pth'))
        pred_arr, pred_id = inference(model=model,dataloader=Style_test_dataloader)

        for pred, id in zip(pred_arr, pred_id):
            keep_info[pred+1] += 1
            df = pd.DataFrame({'name' : id, '0' : int(pred+1)}, index=[0])
            df.to_csv(f'{public_test_path}/{file_name}', mode='a', index=False, header=False)
        
        print('棋風分布',keep_info)