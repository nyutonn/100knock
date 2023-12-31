import pandas as pd
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from sklearn.metrics import accuracy_score
from torch.nn import functional as F
from typing import Any
import optuna
from collections import Counter, defaultdict
from nltk.tokenize import word_tokenize
from pprint import pprint
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import wandb, pickle, logging

def make_vocab(train_data):
    vocab = defaultdict(int)
    for id, (title, category) in train_data.iterrows():
        # words = title.split()
        words = word_tokenize(title)
        for word in words:
            vocab[word] += 1
    vocab = Counter(vocab)
    return vocab

class CreateDataset2(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    # len(Dataset) で返す値を指定
    def __len__(self):
        return len(self.y)

    # Dataset[index] で返す値を指定
    def __getitem__(self, index):
        titles = self.X[index]
        # print(titles)
        # スライス記法のとき
        if type(index) == slice:
            labels = self.y[index]
            input_features = []
            labels_tensor = []
            for title, label in zip(titles, labels):
                input_feature = torch.tensor(title)
                input_features.append(input_feature)
                labels_tensor.append(torch.tensor(label, dtype=torch.int64))
        else:
            text = self.X[index]
            input_features = torch.tensor(text, dtype=torch.int64)
            labels_tensor = torch.tensor(self.y.iloc[index], dtype=torch.int64)
        return {
            'inputs': input_features,
            'labels': labels_tensor
        }
    
# 単語列から出現頻度インデックスを返す関数
def words2index(words, vocab):
    # 単語のみのリストに分割する
    vocab_order, cnt_list = zip(*vocab.most_common())
    index_output = []
    for word in words:
        # 語彙にないときは 0
        if word not in vocab:
            index = 0
        # 回数が1のときも0
        elif cnt_list[vocab_order.index(word)] == 1:
            index = 0
        # 語彙にあるとき，0 インデックスなので +1 する
        else:  
            index = vocab_order.index(word) + 1

        index_output.append(index)
    return index_output


def tokenize_dataset(dataset):
    tokenized_dataset = []
    for data in dataset:
        tokenized_dataset.append(word_tokenize(data))
    return tokenized_dataset


class CNN(nn.Module):
    def __init__(self, vocab_size, padding_idx, out_channels,  emb_size=300, kernel_heights=3, stride=1, n_labels=4, device="cpu", emb_weight=None, active_func='relu', dropout=0.3) -> None:
        """
        stride: 動かす単位（小さいほど細かい）
        kenel_height: 窓の大きさ
        out_channels: 
        conv2d: convolution 層（次元を維持しつつ畳み込み）
        max_pool1d: pooling層（最大値を取り，ダウンサンプリングする）
        """
        super(CNN, self).__init__()
        # 入力ベクトルの大きさが異なるので，emb層で形をそろえる
        if emb_weight is None:
            self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        else:
            self.emb = nn.Embedding.from_pretrained(emb_weight, padding_idx=padding_idx)
        self.conv = nn.Conv2d(1, out_channels, (kernel_heights, emb_size), stride, (padding_idx, 0))
        self.drop = nn.Dropout(dropout)
        self.func = nn.Linear(out_channels, n_labels)
        self.active_func = active_func # 活性化関数をパラメータにする
        
    def forward(self, x):
        emb = self.emb(x).unsqueeze(1)
        conv = self.conv(emb)  # 畳み込み層

        # 活性化関数の最適化を行う
        if self.active_func == 'relu':
            act = F.relu(conv.squeeze(3))
        elif self.active_func == 'tanh':
            act = torch.tanh(conv.squeeze(3))
        elif self.active_func == 'mish':
            act = F.mish(conv.squeeze(3))
        else:
            act = F.relu(conv.squeeze(3))

        max_pool = F.max_pool1d(act, act.size()[2])  # pooling 層
        out = self.func(self.drop(max_pool.squeeze(2)))  # 全結合層？
        return out
    
# early stoppingを差し込む
class EarlyStopping:
    """earlystoppingクラス"""

    def __init__(self, patience=5, verbose=False, path='checkpoint_model.pth'):
        """引数：最小値の非更新数カウンタ、表示設定、モデル格納path"""

        self.patience = patience    #設定ストップカウンタ
        self.verbose = verbose      #表示の有無
        self.counter = 0            #現在のカウンタ値
        self.best_score = None      #ベストスコア
        self.early_stop = False     #ストップフラグ
        self.val_loss_min = np.Inf   #前回のベストスコア記憶用
        self.path = path             #ベストモデル格納path

    def __call__(self, val_loss, model):
        """
        特殊(call)メソッド
        実際に学習ループ内で最小lossを更新したか否かを計算させる部分
        """
        score = -val_loss

        if self.best_score is None:  #1Epoch目の処理
            self.best_score = score   #1Epoch目はそのままベストスコアとして記録する
            self.checkpoint(val_loss, model)  #記録後にモデルを保存してスコア表示する
        elif score < self.best_score:  # ベストスコアを更新できなかった場合
            self.counter += 1   #ストップカウンタを+1
            if self.verbose:  #表示を有効にした場合は経過を表示
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')  #現在のカウンタを表示する 
                logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:  #設定カウントを上回ったらストップフラグをTrueに変更
                self.early_stop = True
        else:  #ベストスコアを更新した場合
            self.best_score = score  #ベストスコアを上書き
            self.checkpoint(val_loss, model)  #モデルを保存してスコア表示
            self.counter = 0  #ストップカウンタリセット

    def checkpoint(self, val_loss, model):
        '''ベストスコア更新時に実行されるチェックポイント関数'''
        if self.verbose:  #表示を有効にした場合は、前回のベストスコアからどれだけ更新したか？を表示
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            logging.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)  #ベストモデルを指定したpathに保存
        self.val_loss_min = val_loss  #その時のlossを記録する

# # 正解率を計算
def measure_acc(model, X, y, device):
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        pred_y = []
        for xi in X:
            xi = xi.to(device)
            pred_yi = model(xi[None]).argmax()
            pred_yi = pred_yi.to('cpu')
            pred_y.append(pred_yi)
    return accuracy_score(pred_y, y)

# 学習率を引数に追加
def train(model, X_train, train_loader, X_valid, valid_loader, output_path, total_epochs, device, lr=0.01, op='sgd', trial_number=0):
    # wandb
    wandb.init(project='chapter-88-ver2')
    wandb.run.name = f'trial-{trial_number}'

    earlystopping = EarlyStopping(patience=3, verbose=True)
    
    # 最適化手法を変更
    if op == 'sgd':  
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif op == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif op == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr)
        
    loss_func = nn.CrossEntropyLoss()

    model = model.to(device)
    # 指定した epoch 数だけ学習
    for epoch in range(total_epochs):
        train_total_loss = 0.
        train_acc_cnt = 0

        # パラメータ更新
        model.train()
        for data in tqdm(train_loader):
            x = data['inputs']
            x = x.to(device)
            y = data['labels']
            y = y.to(device)
            y_pred = model(x)

            # バッチの中で損失計算
            train_loss = 0.
            for yi, yi_pred in zip(y, y_pred):
                loss_i = loss_func(yi_pred, yi)
                train_loss += loss_i
            
            optimizer.zero_grad()  # 勾配の初期化
            train_loss.backward()  # 勾配計算
            optimizer.step()  # パラメータ修正
            train_total_loss += train_loss.item()

            # バッチの中で正解率の計算
            for yi, yi_pred in zip(y, y_pred):
                if yi.item() == yi_pred.argmax():
                    train_acc_cnt += 1
        
        #★毎エポックearlystoppingの判定をさせる★
        train_ave_loss = train_total_loss / len(X_train)
        
        earlystopping(train_ave_loss, model) #callメソッド呼び出し
        if earlystopping.early_stop: #ストップフラグがTrueの場合、breakでforループを抜ける
            print(f"epoch{epoch}: train_loss = {train_ave_loss}")
            print("Early Stopping!")
            logging.info(f"epoch{epoch}: train_loss = {train_ave_loss}")
            logging.info("Early Stopping!")
            break
                
        # train のロスと正解率の計算
        # model.eval()
        # train_acc = measure_acc(model, X_train[:]['inputs'], X_train[:]['labels'], device)


        # valid のロスと正解率の計算
        model.eval()
        valid_acc_cnt = 0
        valid_total_loss = 0.
        with torch.no_grad():
            for data in tqdm(valid_loader):
                x = data['inputs']
                x = x.to(device)
                y = data['labels']
                y = y.to(device)
                y_pred = model(x)

                # バッチの中で損失計算
                valid_loss = 0.
                for yi, yi_pred in zip(y, y_pred):
                    # print(yi)
                    # print(yi_pred)
                    loss_i = loss_func(yi_pred, yi)
                    valid_loss += loss_i

                optimizer.zero_grad()  # 勾配の初期化
                # valid_loss.backward()  # 勾配計算
                # optimizer.step()  # パラメータ修正
                valid_total_loss += valid_loss

                # バッチの中で正解率の計算
                for yi, yi_pred in zip(y, y_pred):
                    if yi.item() == yi_pred.argmax():
                        valid_acc_cnt += 1

            # valid のロスと正解率の計算
            # valid_acc = measure_acc(model, X_valid[:]['inputs'], X_valid[:]['labels'], device)

        # 表示
        train_ave_loss = train_total_loss / len(X_train)
        train_acc = train_acc_cnt / len(X_train)
        valid_ave_loss = valid_total_loss / len(X_valid)
        valid_acc = valid_acc_cnt / len(X_valid)
        print(f"epoch{epoch}: train_loss = {train_ave_loss}, train_acc = {train_acc}, valid_loss = {valid_ave_loss}, valid_acc = {valid_acc}")
        logging.info(f"epoch{epoch}: train_loss = {train_ave_loss}, train_acc = {train_acc}, valid_loss = {valid_ave_loss}, valid_acc = {valid_acc}")
        wandb.log({'train_loss': train_loss, 'train_acc': train_acc, 'valid_loss': valid_loss, 'valid_acc': valid_acc})

    # パラメータを保存
    torch.save(model.state_dict(), output_path)

    wandb.finish()
    
    # valid loss を返り値とする
    return valid_acc

def make_dataset():
    # データ取得
    train_data = pd.read_table('./../work/train.txt', names=['TITLE', 'CATEGORY'])
    valid_data = pd.read_table('./../work/valid.txt', names=['TITLE', 'CATEGORY'])
    test_data = pd.read_table('./../work/test.txt', names=['TITLE', 'CATEGORY'])

    # nltk で tokenized する
    # train_tokenized = tokenize_dataset(train_data['TITLE'])
    # valid_tokenized = tokenize_dataset(train_data['TITLE'])
    # test_tokenized = tokenize_dataset(train_data['TITLE'])

    # tokenized_text をロード
    # f = open('./../work/train_tokenized.txt', 'rb')
    # train_tokenized = pickle.load(f)
    # f = open('./../work/valid_tokenized.txt', 'rb')
    # valid_tokenized = pickle.load(f)
    # f = open('./../work/test_tokenized.txt', 'rb')
    # test_tokenized = pickle.load(f)

    # vocabを作る
    f = open('./../work/vocab.txt', 'rb')
    vocab = pickle.load(f)
    # vocab = make_vocab(train_data)

    # id に変換する
    # train_tokenized_id = [words2index(words, vocab) for words in train_tokenized]
    # valid_tokenized_id = [words2index(words, vocab) for words in valid_tokenized]
    # test_tokenized_id = [words2index(words, vocab) for words in test_tokenized]
    
    f = open('./../work/train-id.txt', 'rb')
    train_tokenized_id = pickle.load(f)
    f = open('./../work/valid-id.txt', 'rb')
    valid_tokenized_id = pickle.load(f)
    f = open('./../work/test-id.txt', 'rb')
    test_tokenized_id = pickle.load(f)

    # y を作る
    category_dict = {'b': 0, 't': 1, 'e': 2, 'm': 3}
    y_train = train_data['CATEGORY'].map(category_dict)
    y_valid = valid_data['CATEGORY'].map(category_dict)
    y_test = test_data['CATEGORY'].map(category_dict)

    # rnn に実際に渡す形にする
    X_train_tokenized = CreateDataset2(train_tokenized_id, y_train)
    X_valid_tokenized = CreateDataset2(valid_tokenized_id, y_valid)
    X_test_tokenized = CreateDataset2(test_tokenized_id, y_test)

    return X_train_tokenized, X_valid_tokenized, X_test_tokenized, vocab


def main():
    # データセットの作成
    X_train, X_valid, X_test, vocab = make_dataset()
    padding_idx = len(vocab)  # 空き単語を埋めるときは最大値を入れる

    # 関数を引数にする関係で，引数の渡し方が分からなかったので，vocab をグローバルにした
    #ミニバッチを取り出して長さを揃える関数
    def collate_fn(batch):
        sorted_batch = sorted(batch, key=lambda x: x['inputs'].shape[0], reverse=True)
        sequences = [x['inputs'] for x in sorted_batch]
        sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=padding_idx)
        labels = torch.LongTensor([x['labels'] for x in sorted_batch])
        return {'inputs': sequences_padded, 'labels': labels}

    # make_dataset()が時間がかかるのでmain()関数で一度だけにする
    # optuna でパラメータの自動最適化
    def objective(trial):
        # 固定のもの
        vocab_size = len(vocab) + 1  # padding の分 +1 する
        n_labels = 4  # ラベル数
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = 32
        train_loader = DataLoader(X_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        valid_loader = DataLoader(X_valid, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(X_test, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        output_path = "./../trained_param.npz"
        total_epochs = 100
        

        # ハイパラを変更させる
        out_channels = trial.suggest_categorical('out_channels', [16, 32, 64, 128])  # これだけよくわかっていない
        emb_size = trial.suggest_categorical('emb_size', [50, 100, 200, 300, 400, 500])  # 特徴ベクトルの次元数
        kernel_height = trial.suggest_int('kernel_height', 1, 5, step=1)  # 窓の大きさ
        stride = trial.suggest_int('stride', 1, 3, step=1)  # 窓を動かす単位
        active_func = trial.suggest_categorical('active_func', ['relu', 'tanh', 'mish'])  # 活性化関数
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)  # 学習率
        dropout = trial.suggest_float('dropout', 0.2, 0.5)  # ドロップアウト
        op = trial.suggest_categorical('optimizer', ['rmsprop', 'adam', 'sgd'])  # 最適化手法 

        print(f"device: {device}")
        logging.info(f"device: {device}")
        model = CNN(vocab_size, padding_idx, out_channels, emb_size, kernel_height, stride, n_labels, device, active_func=active_func, dropout=dropout)
        valid_acc = train(model, X_train, train_loader, X_valid, valid_loader, output_path, total_epochs, device, lr, op, trial.number)

        # 訓練の最後で得られた valid_loss でパラメータチューニングを行う
        return valid_acc

    # 流石に時間がかかりすぎるのでサーバで実行 (100時間くらいかかる？)
    logging.basicConfig(filename='q88.log', level=logging.INFO)
    logging.info('Start running...')
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    print(f"最大のvalid ACC：{study.best_value}")
    print('最高精度のパラメータ')
    pprint(study.best_params)
    logging.info(f"最大の valid ACC：{study.best_value}")
    logging.info(f'最高精度のパラメータ：{study.best_params}')


if __name__ == '__main__':
    main()
