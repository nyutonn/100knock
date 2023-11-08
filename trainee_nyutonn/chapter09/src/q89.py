import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader # データローダ使ってみる
from tqdm import tqdm
import time
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
import tensorflow as tf
import transformers
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
from collections import Counter, defaultdict
import logging


# vocab だけここでつくりたい
path = "/Users/nyuton/Documents/100knock-2023/trainee_nyutonn/chapter08/data/newsCorpora.csv"
df = pd.read_table(path, header=None, sep='\\t', engine='python')
df.columns = ['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP']
# PUBLISHERが特定の行のみを取り出す
publishers = ['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail']
daily_mails = df[df['PUBLISHER'].isin(publishers)]
train_data, non_train, train_target, non_train_target = train_test_split(daily_mails[['TITLE', 'CATEGORY']], daily_mails['CATEGORY'], train_size=0.8, random_state=10, stratify=daily_mails['CATEGORY'])
vocab = defaultdict(int)
for id, (title, category) in train_data.iterrows():
    # words = title.split()
    words = word_tokenize(title)
    for word in words:
        vocab[word] += 1
vocab = Counter(vocab)
PADDING_IDX = vocab_size = len(vocab)

class BERTmodel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bert_sc = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)

    def forward(self, encoding):
        outputs = self.bert_sc(**encoding)
        return outputs
    
class CreateDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return {
            'inputs': self.X[index],
            'labels': self.y[index]
        }

def bert_train(model, X_train, train_loader, X_valid, valid_loader, output_path, total_epochs, device, lr=0.01):
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()

    # BERTモデルのエンコード用
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


    model = model.to(device)
    # 指定した epoch 数だけ学習
    for epoch in range(total_epochs):
        train_total_loss = 0.
        train_acc_cnt = 0

        # パラメータ更新
        model.train()
        for batch in tqdm(train_loader):
            x_texts = batch['inputs']
            x_encordings = tokenizer(
                list(x_texts), 
                max_length=128, 
                padding='max_length', 
                truncation=True, 
                return_tensors='pt', 
                return_attention_mask=True, 
                return_token_type_ids=True
            )
            x_encordings = x_encordings.to(device)
            y = batch['labels']
            y = y.to(device)
            y_pred = model(x_encordings).logits

            # バッチの中で損失計算
            train_loss = loss_func(y_pred, y)

            # train_loss = 0.
            # for yi, yi_pred in zip(y, y_pred):
            #     loss_i = loss_func(yi_pred, yi)
            #     train_loss += loss_i
            
            optimizer.zero_grad() # 勾配の初期化
            train_loss.backward()  # 勾配計算
            optimizer.step()  # パラメータ修正
            train_total_loss += train_loss.item()

            # バッチの中で正解率の計算 # ここを修正
            for yi, yi_pred in zip(y, y_pred):
                if yi.item() == yi_pred.argmax():
                    train_acc_cnt += 1
                
        # train のロスと正解率の計算
        model.eval()
        # train_acc = measure_acc(model, X_train[:]['inputs'], X_train[:]['labels'], device)


        # valid のロスと正解率の計算
        model.eval()
        valid_acc_cnt = 0
        valid_total_loss = 0.
        with torch.no_grad():
            for batch in tqdm(valid_loader):
                x_texts = batch['inputs']
                x_encordings = tokenizer(
                    list(x_texts), 
                    max_length=128, 
                    padding='max_length', 
                    truncation=True, 
                    return_tensors='pt', 
                    return_attention_mask=True, 
                    return_token_type_ids=True
                )
                x_encordings = x_encordings.to(device)
                y = batch['labels']
                y = y.to(device)
                y_pred = model(x_encordings).logits

                # バッチの中で損失計算
                valid_loss = loss_func(y_pred, y)
                # valid_loss = 0.
                # for yi, yi_pred in zip(y, y_pred):
                #     # print(yi)
                #     # print(yi_pred)
                #     loss_i = loss_func(yi_pred, yi)
                #     valid_loss += loss_i

                optimizer.zero_grad()  # 勾配の初期化
                # valid_loss.backward()  # 勾配計算
                # optimizer.step()  # パラメータ修正
                valid_total_loss += valid_loss

                # バッチの中で正解率の計算  # ここを修正
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

    # パラメータを保存
    torch.save(model.state_dict(), output_path)

# 単語列から出現頻度インデックスを返す関数
def sentence2index(sentence):
    # 文を単語列に分割
    words = word_tokenize(sentence)
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

def main():
    # BERTモデルに入れるためのデータセットの作成
    path = "/Users/nyuton/Documents/100knock-2023/trainee_nyutonn/chapter08/data/newsCorpora.csv"
    df = pd.read_table(path, header=None, sep='\\t', engine='python')
    df.columns = ['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP']

    # PUBLISHERが特定の行のみを取り出す
    publishers = ['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail']
    daily_mails = df[df['PUBLISHER'].isin(publishers)]

    # 訓練データ、検証データ、テストデータに分ける
    train_data, non_train, train_target, non_train_target = train_test_split(daily_mails[['TITLE', 'CATEGORY']], daily_mails['CATEGORY'], train_size=0.8, random_state=10, stratify=daily_mails['CATEGORY'])
    valid_data, test_data, valid_target, test_target = train_test_split(non_train, non_train_target, train_size=0.5, random_state=10,  stratify=non_train_target)
    train_data, non_train, train_target, non_train_target = train_test_split(daily_mails[['TITLE', 'CATEGORY']], daily_mails['CATEGORY'], train_size=0.8, random_state=10, stratify=daily_mails['CATEGORY'])
    valid_data, test_data, valid_target, test_target = train_test_split(non_train, non_train_target, train_size=0.5, random_state=10,  stratify=non_train_target)
    
    
    category_dict = {'b': 0, 't': 1, 'e': 2, 'm': 3}
    batch_size = 32

    y_train = torch.tensor(train_data['CATEGORY'].map(category_dict).values, dtype=torch.int64)
    y_valid = torch.tensor(valid_data['CATEGORY'].map(category_dict).values, dtype=torch.int64)
    y_test = torch.tensor(test_data['CATEGORY'].map(category_dict).values, dtype=torch.int64)

    X_train = CreateDataset(train_data['TITLE'], y_train, sentence2index)
    X_valid = CreateDataset(valid_data['TITLE'], y_valid, sentence2index)
    X_test = CreateDataset(test_data['TITLE'], y_test, sentence2index)

    train_set = CreateDataset(train_data['TITLE'].to_list(), y_train)
    valid_set = CreateDataset(valid_data['TITLE'].to_list(), y_valid)
    test_set = CreateDataset(test_data['TITLE'].to_list(), y_test)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    model = BERTmodel()
    total_epochs = 10
    lr = 0.01
    device = 'cpu'

    bert_train(model, X_train, train_loader, X_valid, valid_loader, 'bert_param.npz', total_epochs, device, lr)


if __name__ == '__main__':
    logging.basicConfig(filename='q89_running.log', level=logging.INFO)
    main()