# ecg-classification

- python: 3.10.12
- refernce: https://github.com/branislavhesko/ecg-classification?tab=readme-ov-file

## Dataset: 

取用 [ECG Heartbeat Cetegory](https://www.kaggle.com/datasets/shayanfazeli/heartbeat) 中的 The PTB Diagnostic ECG Database，以下摘錄原論文對於此資料庫的描述。選取第二導程的資料，在每一筆資料的最後一個數字表示為該資料的標籤， 0 標示為正常，1 標示為有心肌梗塞

> The PTB Diagnostics dataset consists of ECG records from 290 subjects: 148 diagnosed as MI , 52 healthy control, and the rest are diagnosed with 7 different disease. Each record contains ECG signals from 12 leads sampled at the frequency of 1000Hz. In this study we have only used ECG lead II, and worked with MI and healthy control categories in our analyses.

The PTB Diagnostic ECG Database 辨別有沒有心肌梗塞，訓練集與驗證集為8:2，將 ptbdb_normal 的前 3237筆放進訓練集，ptbdb_abnormal 前 8405 放進訓練集。





