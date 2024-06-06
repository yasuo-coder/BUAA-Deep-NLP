# -*- coding: utf-8 -*-
import os
import jieba
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
from collections import Counter

# 读取指定目录下的所有txt文件
def read_novels(directory_path):
    novels = []
    f_names = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            with open(os.path.join(directory_path, filename), 'r', encoding='ANSI') as file:
                text = file.read()
                novels.append(text)
                f_names.append(filename)
    return novels, f_names


# 预处理内容
def content_deal(content):  # 语料预处理，进行分词并去除一些广告和无意义内容
    stopwords = ['‘', ']', '：', '’', '“', '！', '”', "\n", ",", "，", "。", "？", "、", "；", "（", "）", "《", "》", "…", "「", "」", "“", "：", "。", "》", "，",
                 "—", "～", "\\", ", ", "'", "n", "一", "u3000", "【", "】", "……", "-", " ", "　", "w", ".", "c", "r", "1", "7", "3", "o", "m", "t", "x"]
    words = jieba.lcut(content)  # 使用jieba进行分词
    words = [word for word in words if word not in stopwords]  # 去除停用词
    return words


def generate_text(model, start_text, vocab, max_len=100):
    model.eval()
    tokens = [vocab.stoi[word] for word in start_text.split()] + [vocab.stoi['<eos>']]
    start_tensor = torch.LongTensor(tokens).unsqueeze(1).to(device)
    generated_text = model.generate(start_tensor, max_len=max_len)
    return generated_text


directory_path = 'E:\\PyCharmProject\\BUAA\\NLP\\homework_4\\中文语料库'

# 读取小说内容
novels, f_names = read_novels(directory_path)

# 对每部小说进行预处理并分词
tokenized_sentences = []
for novel in novels:
    processed_content = content_deal(novel)
    tokenized_sentences.append(processed_content)


class NovelDataset(Dataset):
    def __init__(self, sentences, vocab, max_length=256):
        self.sentences = [torch.tensor([vocab[word] for word in sentence[:max_length]], dtype=torch.long) for sentence
                          in sentences]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]

    @staticmethod
    def collate_fn(batch):
        max_seq_length = 256  # 设置合理的最大序列长度
        batch = [item[:max_seq_length] for item in batch]
        batch = pad_sequence(batch, padding_value=0, batch_first=True)
        return batch


# 构建词汇表
def build_vocab(sentences):
    word_counter = Counter(word for sentence in sentences for word in sentence)
    vocab = {word: i + 2 for i, word in enumerate(word_counter)}  # 从2开始索引
    vocab['<pad>'] = 0  # 填充标记
    vocab['<sos>'] = 1  # 句子开始标记
    vocab['<eos>'] = len(vocab) + 1  # 句子结束标记
    vocab['<unk>'] = len(vocab) + 1  # 未知词标记
    print("词汇表大小：", len(vocab))  # 打印词汇表大小
    return vocab



vocab = build_vocab(tokenized_sentences)
dataset = NovelDataset(tokenized_sentences, vocab)
loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=NovelDataset.collate_fn)


# 定义编码器
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell


# 定义解码器
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim  # 初始化 output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(1)  # LSTM需要序列长度作为第一维
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden, cell


# 定义Seq2Seq模型
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim  # 使用 decoder 的 output_dim 属性

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(src)

        input = trg[:, 0]
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t] = output
            top1 = output.argmax(1)
            input = trg[:, t] if random.random() < teacher_forcing_ratio else top1

        return outputs



# 参数设定
INPUT_DIM = len(vocab)
OUTPUT_DIM = len(vocab)  # 确保这里正确设置了词汇表大小
ENC_EMB_DIM= 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
# 如果上述都没有问题，可能需要调整模型初始化的部分：
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Seq2Seq(enc, dec, device).to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])


# 训练循环，包括梯度累积
def train(model, iterator, optimizer, criterion, device, accumulation_steps=4):
    model.train()
    epoch_loss = 0
    optimizer.zero_grad()

    for i, batch in enumerate(iterator):
        src = batch.to(device)
        trg = batch.to(device)

        output = model(src, trg)
        loss = criterion(output.view(-1, output.shape[-1]), trg.view(-1))
        loss = loss / accumulation_steps  # 平均分配到各个累积步骤
        loss.backward()

        if (i + 1) % accumulation_steps == 0:  # 每accumulation_steps步执行一次参数更新
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss += loss.item() * accumulation_steps  # 将分散的损失重新累积

        return epoch_loss / len(iterator)


for epoch in range(1):
    train_loss = train(model, loader, optimizer, criterion, device, accumulation_steps=4)
    print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}')


index_to_string = {index: word for word, index in vocab.items()}


# 在 generate_text 函数中使用这个映射
def generate_text(model, start_text, vocab, max_len=100):
    model.eval()
    device = next(model.parameters()).device  # 获取模型正在使用的设备
    tokens = [vocab['<sos>']] + [vocab.get(word, vocab['<unk>']) for word in start_text.split()]
    start_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)  # 输入转为合适的形状 (1, 序列长度)

    with torch.no_grad():
        hidden, cell = model.encoder(start_tensor)  # 从编码器获取初始隐藏状态和细胞状态
        inputs = torch.tensor([vocab['<sos>']], dtype=torch.long).unsqueeze(0).to(device)  # 初始化输入为 <sos>，形状为 (1, 1)

        outputs = []
        for _ in range(max_len):
            embedded = model.decoder.embedding(inputs)  # 应用嵌入层
            output, (hidden, cell) = model.decoder.rnn(embedded, (hidden, cell))
            top1 = output.argmax(-1).item()  # 获取最可能的下一个单词索引
            inputs = torch.tensor([[top1]], device=device)
            if top1 == vocab['<eos>']:
                break  # 如果预测到 <eos> 则结束
            outputs.append(top1)

        generated_text = ' '.join([index_to_string[idx] for idx in outputs])  # 将索引转换回单词
    return generated_text


# 生成文本
start_text = "青光闪动，一柄青钢剑倏地刺出，指向在年汉子左肩，使剑少年不等招用老，"
vocab = {'<pad>': 0, '<bos>': 1, '<eos>': 2, '<unk>': 3, '<sos>': 4}
max_len = 50
print(generate_text(model, start_text, vocab, max_len))