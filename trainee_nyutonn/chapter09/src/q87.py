import pandas as pd
from nltk.tokenize import word_tokenize
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
from pprint import pprint
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import logging
import wandb
import pickle

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

class CNN(nn.Module):
    def __init__(self, vocab_size, padding_idx, out_channels,  emb_size=300, kernel_heights=3, stride=1, n_labels=4, device="cpu", emb_weight=None) -> None:
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
        self.drop = nn.Dropout(0.3)
        self.func = nn.Linear(out_channels, n_labels)
        
    def forward(self, x):
        emb = self.emb(x).unsqueeze(1)
        conv = self.conv(emb)  # 畳み込み層
        act = F.relu(conv.squeeze(3))  # 活性化関数
        max_pool = F.max_pool1d(act, act.size()[2])  # pooling 層
        out = self.func(self.drop(max_pool.squeeze(2)))  # 全結合層？
        return out
    
def make_vocab(train_data):
    vocab = defaultdict(int)
    for id, (title, category) in train_data.iterrows():
        # words = title.split()
        words = word_tokenize(title)
        for word in words:
            vocab[word] += 1
    vocab = Counter(vocab)
    return vocab

def tokenize_dataset(dataset):
    tokenized_dataset = []
    for data in dataset:
        tokenized_dataset.append(word_tokenize(data))
    return tokenized_dataset

def words2index(words, vocab):
    # 単語のみのリストに分割する
    print(5)
    vocab_order, cnt_list = zip(*vocab.most_common())
    index_output = []
    for word in words:
        print(6)
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

def make_dataset():
    # データ取得
    train_data = pd.read_table('./../work/train.txt', names=['TITLE', 'CATEGORY'])
    valid_data = pd.read_table('./../work/valid.txt', names=['TITLE', 'CATEGORY'])
    test_data = pd.read_table('./../work/test.txt', names=['TITLE', 'CATEGORY'])
    # tokenized_text をロード
    # f = open('./../work/train_tokenized.txt', 'rb')
    # train_tokenized = pickle.load(f)
    # f = open('./../work/valid_tokenized.txt', 'rb')
    # valid_tokenized = pickle.load(f)
    # f = open('./../work/test_tokenized.txt', 'rb')
    # test_tokenized = pickle.load(f)
    # nltk で tokenized する
    # train_tokenized = tokenize_dataset(train_data['TITLE'])
    # valid_tokenized = tokenize_dataset(valid_data['TITLE'])
    # test_tokenized = tokenize_dataset(test_data['TITLE'])

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


def train(model, X_train, train_loader, X_valid, valid_loader, output_path, total_epochs, device, lr=0.01):
    wandb.init(project="chapter09_87")
    wandb.run.name = 'modify-valid-data'
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()

    model = model.to(device)
    # 指定した epoch 数だけ学習
    for epoch in range(total_epochs):
        train_total_loss = 0.
        train_acc_cnt = 0
        train_cnt = 0

        # パラメータ更新
        model.train()
        for i, data in enumerate(tqdm(train_loader)):
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

            # バッチの中で正解率の計算 # ここを修正
            for yi, yi_pred in zip(y, y_pred):
                if yi.item() == yi_pred.argmax():
                    train_acc_cnt += 1
                train_cnt += 1

            # wandb に10ステップごとにログを書く
            if i % 10 == 0:
                train_running_loss = train_total_loss / train_cnt
                train_running_acc = train_acc_cnt / train_cnt
                
        # train のロスと正解率の計算
        # model.eval()
        # train_acc2 = measure_acc(model, X_train[:]['inputs'], X_train[:]['labels'], device)


        # valid のロスと正解率の計算
        model.eval()
        valid_acc_cnt = 0
        valid_total_loss = 0.
        with torch.no_grad():
            for i, data in enumerate(tqdm(valid_loader)):
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

                # バッチの中で正解率の計算  # ここを修正
                for yi, yi_pred in zip(y, y_pred):
                    if yi.item() == yi_pred.argmax():
                        valid_acc_cnt += 1


            # valid のロスと正解率の計算
            # valid_acc2 = measure_acc(model, X_valid[:]['inputs'], X_valid[:]['labels'], device)

        # 表示
        train_ave_loss = train_total_loss / len(X_train)
        train_acc = train_acc_cnt / len(X_train)
        valid_ave_loss = valid_total_loss / len(X_valid)
        valid_acc = valid_acc_cnt / len(X_valid)
        print(f"epoch{epoch}: train_loss = {train_ave_loss}, train_acc = {train_acc}, valid_loss = {valid_ave_loss}, valid_acc = {valid_acc}")
        wandb.log({'train_loss': train_ave_loss, 'train_acc': train_acc, 'valid_loss': valid_ave_loss, 'valid_acc': valid_acc})
        logging.info(f"epoch{epoch}: train_loss = {train_ave_loss}, train_acc = {train_acc}, valid_loss = {valid_ave_loss}, valid_acc = {valid_acc}")

    # パラメータを保存
    torch.save(model.state_dict(), output_path)
    wandb.finish()

def main():
    # データセットを作成
    X_train, X_valid, X_test, vocab = make_dataset()
    padding_idx = len(vocab)  # 空き単語を埋めるときは最大値を入れる
    
    vocab_size = len(vocab) + 1  # padding の分 +1 する
    padding_idx = len(vocab)  # 空き単語を埋めるときは最大値を入れる
    out_channels = 50 # ハイパラ？
    emb_size = 300  # ハイパラ
    kernel_height = 3
    stride = 1

    #ミニバッチを取り出して長さを揃える関数
    def collate_fn(batch):
        sorted_batch = sorted(batch, key=lambda x: x['inputs'].shape[0], reverse=True)
        sequences = [x['inputs'] for x in sorted_batch]
        sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=padding_idx)
        labels = torch.LongTensor([x['labels'] for x in sorted_batch])
        return {'inputs': sequences_padded, 'labels': labels}

    # 固定のもの
    batch_size = 32
    train_loader = DataLoader(X_train, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    valid_loader = DataLoader(X_valid, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(X_test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    output_path = "./../trained_param.npz"
    total_epochs = 10

    n_labels = 4  # ラベル数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")


    model = CNN(vocab_size, padding_idx, out_channels, emb_size, kernel_height, stride, n_labels, device)
    output_path = "./trained_param.npz"
    total_epochs = 10
    train(model, X_train, train_loader, X_valid, valid_loader, output_path, total_epochs, device)


if __name__ == '__main__':
    logging.basicConfig(filename='q87.log', level=logging.INFO)
    main()
