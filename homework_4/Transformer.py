import os
import jieba
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, AutoTokenizer

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

def content_deal(content):
    stopwords = ['‘', ']', '：', '’', '“', '！', '”', "\n", ",", "，", "。", "？", "、", "；", "（", "）", "《", "》", "…", "「", "」", "“", "：", "。", "》", "，", "—", "～", "\\", ", ", "'", "n", "一", "u3000", "【", "】", "……", "-", " ", "　", "w", ".", "c", "r", "1", "7", "3", "o", "m", "t", "x"]
    words = jieba.lcut(content)
    words = [word for word in words if word not in stopwords]
    return words

def load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return model, tokenizer

class NovelDataset(Dataset):
    def __init__(self, tokenized_texts, max_length=512):
        self.tokenized_texts = tokenized_texts
        self.max_length = max_length
        self.pad_token_id = 50256

    def __len__(self):
        return len(self.tokenized_texts)

    def __getitem__(self, idx):
        token_ids = self.tokenized_texts[idx]
        attention_mask = [1] * len(token_ids)

        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
        elif len(token_ids) < self.max_length:
            padding_length = self.max_length - len(token_ids)
            token_ids += [self.pad_token_id] * padding_length
            attention_mask += [0] * padding_length

        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }

def train_model(data_loader, model, epochs=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(data_loader)}')

def generate_text(model, tokenizer, prompt, max_length=100, temperature=1.0, top_k=50, top_p=0.95, do_sample=True):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            do_sample=do_sample
        )[0]
    generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    return generated_text

if __name__ == "__main__":
    directory_path = 'E:\\PyCharmProject\\BUAA\\NLP\\homework_4\\中文语料库'
    model_path = "E:\\PyCharmProject\\BUAA\\NLP\\homework_4"

    novels, f_names = read_novels(directory_path)
    model, tokenizer = load_model_and_tokenizer(model_path)

    tokenized_sentences = [tokenizer.encode(" ".join(content_deal(novel)), max_length=512, truncation=True, padding="max_length") for novel in novels]
    dataset = NovelDataset(tokenized_sentences)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

    train_model(data_loader, model, epochs=3)

    prompt_text = "江湖上人称老侠客，一身剑术无人能敌。一日，老侠客行至古镇，突然"
    generated_text = generate_text(model, tokenizer, prompt_text, max_length=100)
    print("生成结果：", generated_text)
