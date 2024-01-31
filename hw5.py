from cgitb import reset
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer , BertModel
import torchvision
import argparse


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.txt_model = BertModel.from_pretrained('./bert-base-uncased')
        self.img_model = torchvision.models.resnet18(pretrained=True)
        self.linear1 = nn.Linear(768, 128)
        self.linear2 = nn.Linear(1000, 128)
        self.fc = nn.Linear(256, 3)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask, image):
        img_out = self.img_model(image)
        img_out = self.linear2(img_out)
        img_out = self.relu(img_out)
        txt_out = self.txt_model(input_ids=input_ids, attention_mask=attention_mask)
        txt_out = txt_out.last_hidden_state[:,0,:]
        txt_out.view(txt_out.shape[0], -1)
        txt_out = self.linear1(txt_out)
        txt_out = self.relu(txt_out)
        out = torch.cat((txt_out, img_out), dim=-1)
        out = self.fc(out)
        return out


class txtonlyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.txt_model = BertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Linear(768, 256)
        self.fc = nn.Linear(256, 3)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask, image):
        txt_out = self.txt_model(input_ids=input_ids, attention_mask=attention_mask)
        txt_out = txt_out.last_hidden_state[:,0,:]
        txt_out.view(txt_out.shape[0], -1)
        txt_out = self.linear(txt_out)
        txt_out = self.relu(txt_out)
        out = self.fc(txt_out)
        return out


class imgonlyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_model = torchvision.models.resnet18(pretrained=True)
        self.linear = nn.Linear(1000, 256)
        self.fc = nn.Linear(256, 3)
        self.relu = nn.ReLU()


    def forward(self, input_ids, attention_mask, image):
        img_out = self.img_model(image)
        img_out = self.linear(img_out)
        img_out = self.relu(img_out)
        out = self.fc(img_out)
        return out


def txt_(txt, token):
    result = token.batch_encode_plus(batch_text_or_text_pairs=txt, truncation=True, padding='max_length', max_length=32, return_tensors='pt')
    input_ids = result['input_ids']
    attention_mask = result['attention_mask']
    return input_ids, attention_mask


class MultimodalDataset():
    def __init__(self, images, descriptions, tags, token):
        self.images = images
        self.descriptions = descriptions
        self.tags = tags
        self.input_ids, self.attention_masks = txt_(self.descriptions, token)

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        img = self.images[idx]
        des = self.descriptions[idx]
        tag = self.tags[idx]
        input_id = self.input_ids[idx]
        attention_mask = self.attention_masks[idx]
        return img, des, tag, input_id, attention_mask


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def train_process(model, epoch_num, optimizer, train_dataloader, valid_dataloader, train_count, valid_count):
    Loss_C = nn.CrossEntropyLoss()
    train_acc = []
    valid_acc = []
    for epoch in range(epoch_num):
        loss = 0.0
        train_cor_count = 0
        valid_cor_count = 0
        for b_idx, (img, des, target, idx, mask) in enumerate(train_dataloader):
            img, mask, idx, target = img.to(device), mask.to(device), idx.to(device), target.to(device)
            output = model(idx, mask, img)
            optimizer.zero_grad()
            loss = Loss_C(output, target)
            loss.backward()
            optimizer.step()
            pred = output.argmax(dim=1)
            train_cor_count += int(pred.eq(target).sum())
        train_acc.append(train_cor_count / train_count)
        for img, des, target, idx, mask in valid_dataloader:
            img, mask, idx, target = img.to(device), mask.to(device), idx.to(device), target.to(device)
            output = model(idx, mask, img)
            pred = output.argmax(dim=1)
            valid_cor_count += int(pred.eq(target).sum())
        valid_acc.append(valid_cor_count / valid_count)
        print('Train Epoch: {}, Train_Loss: {:.4f}, Train Accuracy: {:.4f}, Valid Accuracy: {:.4f}'.format(epoch + 1, loss.item(), train_cor_count / train_count, valid_cor_count / valid_count))


def main():
    parser = argparse.ArgumentParser(description='params') 
    parser.add_argument('--image_only', action='store_true')
    parser.add_argument('--text_only', action='store_true')
    args = parser.parse_args()
    if args.image_only:
        model = imgonlyModel().to(device)
    if args.text_only:
        model = txtonlyModel().to(device)
    else:
        model = Model().to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    epoch_num = 10
    images = []
    descriptions = []
    tags = []
    tag_ = {"neutral": 0, "negative": 1, "positive": 2}
    train_df = pd.read_csv("./train.txt")
    token = BertTokenizer.from_pretrained('./bert-base-uncased')

    for i in range(train_df.shape[0]):
        guid = train_df.iloc[i]['guid']
        tag = train_df.iloc[i]['tag']
        img = Image.open('./data/' + str(guid) + '.jpg')
        img = img.resize((224, 224), Image.LANCZOS)
        img = np.asarray(img, dtype='float32')
        with open('./data/' + str(guid) + '.txt', encoding='gb18030') as f:
            des = f.read()
        images.append(img.transpose(2, 0, 1))
        descriptions.append(des)
        tags.append(tag_[tag])
 
    for i in range(len(descriptions)):
        a = descriptions[i]
        word_list = a.replace("#", "").split(" ")
        words_result = []
        for word in word_list:
            if len(word) < 1:
                continue
            elif word[0]=='@':
                continue
            else:
                words_result.append(word)
        descriptions[i] = " ".join(words_result)
    img_txt_pairs = [(images[i], descriptions[i]) for i in range(len(descriptions))]
    
    X_train, X_valid, tag_train, tag_valid = train_test_split(img_txt_pairs, tags, test_size=0.2, random_state=1458, shuffle=True)
    image_train, txt_train = [X_train[i][0] for i in range(len(X_train))], [X_train[i][1] for i in range(len(X_train))]
    image_valid, txt_valid = [X_valid[i][0] for i in range(len(X_valid))], [X_valid[i][1] for i in range(len(X_valid))]

    train_dataset = MultimodalDataset(image_train, txt_train, tag_train, token)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_dataset = MultimodalDataset(image_valid, txt_valid, tag_valid, token)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=True)

    train_process(model, epoch_num, optimizer, train_dataloader, valid_dataloader, len(X_train), len(X_valid))
 
    tag_list = ["neutral", "negative", "positive"]
    test_df = pd.read_csv("./test_without_label.txt")
    guid_list = test_df['guid'].tolist()
    tag_test_list = []
    for idx in guid_list:
        img = Image.open('./data/' + str(idx) + '.jpg')
        img = img.resize((224,224), Image.LANCZOS)
        image = np.asarray(img, dtype = 'float32')
        image = image.transpose(2,0,1)
        with open('./data/' + str(idx) + '.txt', encoding='gb18030') as fp:
            description = fp.read()
        input_id, mask = txt_([description],token)
        image = image.reshape(1,image.shape[0],image.shape[1],image.shape[2])
        y_pred = model(input_id.to(device), mask.to(device), torch.Tensor(image).to(device))
        tag_test_list.append(tag_list[y_pred[0].argmax(dim=-1).item()])
    
    result_df = pd.DataFrame({'guid':guid_list, 'tag':tag_test_list})
    result_df.to_csv('./test_with_label.txt',sep=',',index=False)


if __name__ == '__main__':
    main()