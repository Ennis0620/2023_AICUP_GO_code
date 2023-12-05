## AI CUP 2023 秋季賽 圍棋棋力模仿與棋風辨識競賽

### 安裝套件
```
pip install -r requirements.txt
```


### 資料夾說明

- 29_Public Testing Dataset_Private Submission Template_v2 - private測試資料
- 29_Public Testing Dataset_Public Submission Template_v2 - public測試資料
- 29_Training Dataset - 訓練資料
- models - 棋力訓練時的model weight會儲存在這裡
- models_style - 棋風訓練時的model weight會儲存在這裡
- submissions - inference後的csv檔會存在這裡
>棋力
- main_dan_kyu - 棋力code
  - models_block - 模型架構
  - utils
    - board_up - 提子相關
    - preprocess - 前處理
    - models - 淘汰的模型架構
  ***********
  - valid_id.txt - v1 data被分去valid的id，在執行v2時會需要用到( pytorch_check_v1_valid_id.py產生)
  ***********
  - pytorch_Merge17_v1_train.py -訓練有落子順序的v1
  - pytorch_Merge17_v1_train(augmentation).py -訓練有落子順序的v1 augmentation
  - pytorch_Merge17_v2_train_another_data.py -訓練有落子順序的v2
  - pytorch_Merge17_v2_train_another_data(augmentation).py -訓練有落子順序的v2 augmentation
  ***********
  - pytorch_Merge20_no_Seq_v1_train.py -訓練沒落子順序的v1
  - pytorch_Merge20_no_Seq_v1_train(augmentation).py -訓練沒落子順序的v1 augmentation
  - pytorch_Merge20_no_Seq_v2_train_another_data.py -訓練沒落子順序的v2
  - pytorch_Merge20_no_Seq_v2_train_another_data(augmentation).py -訓練沒落子順序的v2 augmentation
  ***********
  - pytorch_Merge17_inference.py -單獨一個有落子順序的model做inference
  - pytorch_Merge20_no_Seq_inference -單獨一個沒落子順序的model做inference
  - pytorch_ensemble_TTA.py -合併v1、v2有落子順序的 + v1、v2無落子順序的weight做ensemble + TTA

>棋風
- main_style - 棋風code
  - models_block - 模型架構
  - utils
    - board_up - 提子相關
    - preprocess - 前處理
    - models - 淘汰的模型架構
  ***********
  - pytorch_style21_10fold.py -(提子+黑白互換 會練10個model)
  - pytorch_style28_5fold -(提子+不提子+黑白互換 會練5個model)
  ***********
  - pytorch_style21_10fold_inference.py -單獨的model做預測
  - pytorch_style21_10fold_inference_ensemble.py - 讀CROP_SIZE_LIST內的model weight投票
  - pytorch_style21_10fold_inference_ensemble_TTA.py - 讀CROP_SIZE_LIST內的model weight投票、TTA
  - pytorch_style21_10fold_inference_ensemble_TTA_use_prob.py - 讀CROP_SIZE_LIST內的model weight投票(用機率累加)、TTA 
  ***********
  - pytorch_style28_5fold_inference.py -單獨的model做預測
  - pytorch_style28_5fold_inference_ensemble.py - 讀CROP_SIZE_LIST內的model weight投票
  - pytorch_style28_5fold_inference_ensemble_TTA.py - 讀CROP_SIZE_LIST內的model weight投票、TTA
  - pytorch_style28_5fold_inference_ensemble_TTA_use_prob.py - 讀CROP_SIZE_LIST內的model weight投票(用機率累加)、TTA
  ***********
  - pytorch_style21_28_all_ensemble_TTA.py - 合併2種用不同augmentation訓練的weight做預測、投票、TTA
  - pytorch_style21_28_all_ensemble_TTA_use_prob.py - 合併2種用不同augmentation訓練的weight做預測、投票(用機率累加)、TTA


### 執行訓練步驟

>棋力

* Step1. pytorch_Merge17_v1_train.py
* Step2. pytorch_Merge17_v1_train(augmentation).py 訓練好第1個有seq的weight
* Step3. pytorch_Merge17_v2_train_another_data.py 
* Step4. pytorch_Merge17_v2_train_another_data(augmentation).py 訓練好第2個有seq的weight

* Step5. pytorch_Merge20_no_Seq_v1_train.py
* Step6. pytorch_Merge20_no_Seq_v1_train(augmentation).py 訓練好第1個無seq的weight
* Step7. pytorch_Merge20_no_Seq_v2_train_another_data.py 
* Step8. pytorch_Merge20_no_Seq_v2_train_another_data(augmentation).py 訓練好第2個無seq的weight

棋力最後會產生4個best weight

>棋風

* Step1. pytorch_style21_10fold.py 訓練好10個設定的crop size weight
* Step2. pytorch_style28_5fold.py 訓練好5個設定的crop size weight


## 訓練變數說明
>棋力 trainig
```
BATCH_SIZE = 4 (視電腦GPU大小可以調小)
LEARNING_RATE = 0.001(可調整)
CHANNELS_1 = 4 (無法更動)
CHANNELS_2 = 16(無法更動)
NUM_CLASSES = 361 (無法更動)
```

>棋風 trainig
```
CROP_SIZE = 19(crop棋盤的輸入大小，最小為5，最大為19)
BATCH_SIZE = 64(視電腦GPU大小可以調小)
LEARNING_RATE = 0.001(可調整)
CHANNELS_1 = 4 (無法更動)
NUM_CLASSES = 3(無法更動)
```

## 預測變數說明
>棋力 inference
```
file_name (要輸出的csv檔名)
TTA_TIMES = 5 (要做幾次TTA)
MODEL_NAME (每個要預測的model weight完整名稱)
BATCH_SIZE = 1 (無法更動)
CHANNELS_1 = 4 (無法更動)
CHANNELS_2 = 16(無法更動)
NUM_CLASSES = 361 (無法更動)
```

>棋風 inference
```
CROP_SIZE_LIST1 = [8,10,12,14,16,19](是style21的crop size)
CROP_SIZE_LIST2 = [13](是style28的crop size)
file_name (要輸出的csv檔名)
PER_MODEL_PREDICT_NUM = 5 (要做幾次TTA)
MODEL_NAME (每個要預測的model weight完整名稱)
BATCH_SIZE = 256 (視電腦GPU大小可以調小)
CHANNELS_1 = 4 (無法更動)
NUM_CLASSES = 3 (無法更動)
```

## Weight連結
>棋力

|Model      |weight|
|-----------|-------|
|有落子順序v1| [sequence weight v1](https://drive.google.com/file/d/1sr7GstUrPY8k1_TWuPfaS04rdY3ry-B1/view?usp=sharing)     |
|有落子順序v1(不同weight)| [sequence weight v1(不同weight)](https://drive.google.com/file/d/1WkPKuxShDeQgcqpQydr63yrIIePzAOBx/view?usp=sharing)     |
|有落子順序v2|[sequence weight v2](https://drive.google.com/file/d/1KWafMBAS_wKb31mS1w-H94Fa9X8r97W-/view?usp=drive_link)      |
|無落子順序v1|[no sequence weight v1](https://drive.google.com/file/d/1N5NlFc9mps2co7pseQbnqGdxlC8F7Hxi/view?usp=drive_link)      |
|無落子順序v1(不同weight)|[no sequence weight v1(不同weight)](https://drive.google.com/file/d/1RVVcEDYyQStdtVFx7iWgx2eL1iw0zQs-/view?usp=sharing)      |
|無落子順序v2|[no sequence weight v2](https://drive.google.com/file/d/1Sie8fcfYLqVuLc4o5LQT2fy9zorBgina/view?usp=drive_link)      |

建議可以直接全部載下來放到model資料夾，inference時就會自動讀取做Ensemble+TTA

>棋風

目前weight的crop size如下
|Model      |crop size|
|-----------|--------|
|style21|8,10,12,14,16,19|
|style28|13|

|Model      |weight|
|-----------|--------|
|不同crop size|[weight](https://drive.google.com/drive/folders/1YfbbbcYyXpGju2HPiqRv5_j_WsoiASZ2?usp=drive_link)      |

Ex:
  - Style_model21_10fold_channel4_8_fold0_valloss_best.pth

代表這個weight是 input channels = 4 、crop size = 8、valid是用fold0

建議可以直接全部載下來放到model_style資料夾，inference時就會自動讀取做Ensemble+TTA

## 聯絡資訊

> - [E-mail](ennis06205668@gmail.com)
> - [Linkedin](https://www.linkedin.com/in/ting-en-hsu-010728225/)





