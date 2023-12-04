import torch, os, csv
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.progress import track
from torch.utils.data import Dataset
from vit_pytorch import ViT
from models_block.inception_v4_only_reduceA_with_seq import inception_resnet_v2_GO_V3
from utils.preprocess import prepare_board_status_with_sequence

chars = 'abcdefghijklmnopqrst'
coordinates = {k:v for v,k in enumerate(chars)}
chartonumbers = {k:v for k,v in enumerate(chars)}

def number_to_char(number):
    number_1, number_2 = divmod(number, 19)
    return chartonumbers[number_1] + chartonumbers[number_2]

def top_5_preds_with_chars(predictions):
    return_topk = []
    test_topk = torch.topk(predictions, 5).indices
    test_topk = test_topk[0].cpu().numpy()
    for k in test_topk:
        return_topk.append(number_to_char(k))
    # print(test_topk)
    # print(return_topk)
    return return_topk

class test_dataloader(Dataset):
    def __init__(self, csv_path):
        super(test_dataloader).__init__()
        #切資料
        csv = open(f'{csv_path[0]}').read().splitlines()
        csv2 = open(f'{csv_path[1]}').read().splitlines()

        self.games = [i.split(',',2)[-1] for i in csv]
        self.games_id = [i.split(',')[0] for i in csv]
        self.colors = [i.split(',')[1] for i in csv]
        self.games2 = [i.split(',',2)[-1] for i in csv2]
        self.games_id2 = [i.split(',')[0] for i in csv2]
        self.colors2 = [i.split(',')[1] for i in csv2]

        self.games = self.games + self.games2
        self.games_id = self.games_id + self.games_id2
        self.colors = self.colors + self.colors2

        n_games = 0

        for game in self.games:
            n_games += 1
        print(f"Total Test Games: {len(self.games)}")

        self.n_games = n_games
        
    def __len__(self):
        return self.n_games

    def __getitem__(self, index):
        x,x2 = [],[]
        moves_list = self.games[index].split(',')
        target_color = self.colors[index]
        moves_list = [g for g in moves_list if g != '']
        games_id = self.games_id[index]
        _,process_board,seq_board = prepare_board_status_with_sequence(moves_list,
                                        remove_dead=False,
                                        sperate_posotion=True,
                                        empty_is_one=True,
                                        last_move_ = True)
        
        x = np.array(process_board)
        return (x,seq_board), games_id
        
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
            for (test_board,test_seq), test_id in dataloader:
                test_board = test_board.to(DEVICE)     
                test_seq = test_seq.to(DEVICE)           
                test_board = test_board.view(-1, CHANNELS_1, 19, 19).to(torch.float32)
                test_seq = test_seq.view(-1, 1, CHANNELS_2, 19, 19).to(torch.float32)
                test_pred = model(test_board,test_seq)
                test_pred = torch.softmax(test_pred, dim=1)
                # test_p = torch.argmax(test_pred, dim=1)

                test_p = top_5_preds_with_chars(test_pred)
                
                pred_arr.append(test_p)
                pred_id = np.append(pred_id, test_id)
                progress.advance(test_batch_tqdm, advance=1)
            progress.reset(test_batch_tqdm)
        return pred_arr, pred_id
    
if __name__ == '__main__':
    ### Configs
    CURRENT_PATH = os.path.dirname(__file__)
    BEFORE_PATH = os.path.abspath(os.path.join(CURRENT_PATH, os.path.pardir))
    MODEL_PATH = f'{BEFORE_PATH}/models/'

    kyu_public_test_path = f'{BEFORE_PATH}/29_Public Testing Dataset_Public Submission Template_v2/29_Public Testing Dataset_v2/kyu_test_public.csv'
    dan_public_test_path = f'{BEFORE_PATH}/29_Public Testing Dataset_Public Submission Template_v2/29_Public Testing Dataset_v2/dan_test_public.csv'
    kyu_private_test_path = f'{BEFORE_PATH}/29_Public Testing Dataset_Private Submission Template_v2/29_Private Testing Dataset_v2/kyu_test_private.csv'
    dan_private_test_path = f'{BEFORE_PATH}/29_Public Testing Dataset_Private Submission Template_v2/29_Private Testing Dataset_v2/dan_test_private.csv'
    
    csv_path = [[kyu_public_test_path,kyu_private_test_path],[dan_public_test_path,dan_private_test_path]]
    
    
    
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('GPU state:',DEVICE)
    BATCH_SIZE = 1
    NUM_CLASSES = 361
    CHANNELS_1 = 4
    CHANNELS_2 = 16

    model = inception_resnet_v2_GO_V3(in_channels=CHANNELS_1, num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    MODEL_NAME = f'Merge_model_CNN17_4_16channels_valloss_best_best_best_best_best_best'
    model.load_state_dict(torch.load(f'{MODEL_PATH}/{MODEL_NAME}.pth'))

    for per_csv_path in csv_path:
        all_test_dataloader = test_dataloader(csv_path=per_csv_path)
        all_test_dataloader = DataLoader(all_test_dataloader, batch_size=BATCH_SIZE, num_workers=3)
        pred_arr, pred_id = inference(model=model,dataloader=all_test_dataloader)
        public_test_path = f'{BEFORE_PATH}/submissions/'
        file_name = 'CNN17_Merge_private_best_best_best_best_best_best.csv'
        for pred, id in zip(pred_arr, pred_id):
            df = pd.DataFrame({'name' : id, '0' : pred[0], '1' : pred[1], '2' : pred[2], '3' : pred[3], '4' : pred[4]}, index=[0])
            df.to_csv(f'{public_test_path}/{file_name}', mode='a', index=False, header=False)