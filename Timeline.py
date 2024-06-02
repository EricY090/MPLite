# The original TimeLine is from: https://github.com/tiantiantu/Timeline
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import os
from metrics import f1, top_k_prec_recall
import _pickle as pickle

torch.manual_seed(6669)
mimic3_path = os.path.join('data', 'mimic3')
standard_path = os.path.join(mimic3_path, 'standard')
pre_trained = os.path.join(mimic3_path, 'Pre-trained')

def load_file(path):
    train_visitset=np.load(open(os.path.join(path, 'train_visitfile.npy'), 'rb'), allow_pickle = True)
    train_labelset=np.load(open(os.path.join(path, 'train_labelfile.npy'), 'rb'), allow_pickle = True)
    train_gapset=np.load(open(os.path.join(path, 'train_gapfile.npy'), 'rb'), allow_pickle = True)
    valid_visitset=np.load(open(os.path.join(path, 'valid_visitfile.npy'), 'rb'), allow_pickle = True)
    valid_labelset=np.load(open(os.path.join(path, 'valid_labelfile.npy'), 'rb'), allow_pickle = True)
    valid_gapset=np.load(open(os.path.join(path, 'valid_gapfile.npy'), 'rb'), allow_pickle = True)
    test_visitset=np.load(open(os.path.join(path, 'test_visitfile.npy'), 'rb'), allow_pickle = True)
    test_labelset=np.load(open(os.path.join(path, 'test_labelfile.npy'), 'rb'), allow_pickle = True)
    test_gapset=np.load(open(os.path.join(path, 'test_gapfile.npy'), 'rb'), allow_pickle = True)

    return train_visitset, valid_visitset, test_visitset, train_gapset, valid_gapset, test_gapset, train_labelset, valid_labelset, test_labelset


def code_map(train_visitset, valid_visitset, test_visitset, train_labelset, valid_labelset, test_labelset):
    visitset = np.concatenate((train_visitset, valid_visitset, test_visitset), axis = 0)
    labelset = np.concatenate((train_labelset, valid_labelset, test_labelset), axis = 0)
    
    data=[]
    for i in range(0,len(visitset)):
        data.append((visitset[i], labelset[i]))  
    
    code_to_ix = {}
    for visits, labels in data:
        for visit in visits:
            for code in visit:
                if code not in code_to_ix:
                    code_to_ix[code] = len(code_to_ix)+1
        for label in labels:
            if label not in code_to_ix:
                code_to_ix[label]=len(code_to_ix)+1
    print ("The number of codes:", len(code_to_ix))
    return code_to_ix


def preprocessing(code_to_ix, batchsize, visitset, gapset, labelset, lab_x):
    data=[]
    for i in range(0,len(visitset)):
        data.append((visitset[i], labelset[i], gapset[i], lab_x[i]))
    
    batch_data=[]
    
    for start_ix in range(0, len(data)-batchsize+1, batchsize):
        thisblock=data[start_ix:start_ix+batchsize]
        mybsize= len(thisblock)
        mynumvisit=np.max([len(ii[0]) for ii in thisblock])
        mynumcode=np.max([len(jj) for ii in thisblock for jj in ii[0] ])
        main_matrix = np.zeros((mybsize, mynumvisit, mynumcode), dtype= np.int32)
        mask_matrix = np.zeros((mybsize, mynumvisit, mynumcode), dtype= np.float32)
        gap_matrix = np.zeros((mybsize, mynumvisit), dtype= np.float32)
        # Label multi-hot
        label_matrix = np.zeros((mybsize, len(code_to_ix)), dtype= np.int32)
        # Lab multi-hot
        lab_matrix = np.zeros((mybsize, lab_x.shape[1]), dtype= np.float32)
        for i in range(main_matrix.shape[0]):
            for j in range(main_matrix.shape[1]):
                for k in range(main_matrix.shape[2]):
                    try:
                        main_matrix[i,j,k] = code_to_ix[thisblock[i][0][j][k]]
                        
                    except IndexError:
                        mask_matrix[i,j,k] = 1e+20
                        
        for i in range(gap_matrix.shape[0]):
            mylabel = [code_to_ix[x]-1 for x in thisblock[i][1]]
            label_matrix[i][mylabel] = 1
            lab_matrix[i] = thisblock[i][3]
            for j in range(gap_matrix.shape[1]):
                try:
                    gap_matrix[i,j]=thisblock[i][2][j]
                except IndexError:
                    pass
        batch_data.append(( (autograd.Variable(torch.from_numpy(main_matrix)), autograd.Variable(torch.from_numpy(mask_matrix)), autograd.Variable(torch.from_numpy(gap_matrix)), autograd.Variable(torch.from_numpy(lab_matrix))),autograd.Variable(torch.from_numpy(label_matrix)) ))
    
    print ("The number of batches:", len(batch_data))
    return batch_data

######################################################################
# Create the model:


class Timeline(nn.Module):

    def __init__(self, batch_size, embedding_dim, hidden_dim, attention_dim, vocab_size, dropoutrate, use_lab, weight, bias, lab_dim):
        super(Timeline, self).__init__()
        self.use_lab = use_lab
        self.hidden_dim = hidden_dim
        self.batchsi=batch_size
        self.word_embeddings = nn.Embedding(vocab_size+1, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.hidden2label = nn.Linear(hidden_dim*2, vocab_size)
        self.hidden = self.init_hidden()
        self.attention=nn.Linear(embedding_dim, attention_dim)
        self.vector1=nn.Parameter(torch.randn(attention_dim,1))
        self.decay=nn.Parameter(torch.FloatTensor([-0.1]*(vocab_size+1)))    
        self.initial=nn.Parameter(torch.FloatTensor([1.0]*(vocab_size+1)))   
        self.tanh=nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.relu=nn.ReLU()
        self.attention_dimensionality=attention_dim
        self.WQ1=nn.Linear(embedding_dim, attention_dim,bias=False)
        self.WK1=nn.Linear(embedding_dim, attention_dim,bias=False)
        self.embed_drop = nn.Dropout(p=dropoutrate)
        if use_lab:
            self.hidden2label = nn.Linear(hidden_dim*2 + 200, vocab_size)
            self.pretrained_layer = nn.Linear(lab_dim, 200)
            self.pretrained_layer.weight = torch.nn.parameter.Parameter(torch.transpose(weight, 0, 1), requires_grad=False)
            self.pretrained_layer.bias = torch.nn.parameter.Parameter(bias, requires_grad=False)
        

    def init_hidden(self):
        
        # return (autograd.Variable(torch.zeros(2, self.batchsi, self.hidden_dim).cuda()),
        #         autograd.Variable(torch.zeros(2, self.batchsi, self.hidden_dim).cuda()))
        return (autograd.Variable(torch.zeros(2, self.batchsi, self.hidden_dim)),
                autograd.Variable(torch.zeros(2, self.batchsi, self.hidden_dim)))

    def forward(self, sentence, Mode):
        numcode=sentence[0].size()[2]
        numvisit=sentence[0].size()[1]
        numbatch=sentence[0].size()[0]
        thisembeddings =self.word_embeddings(sentence[0].view(-1,numcode))
        thisembeddings = self.embed_drop(thisembeddings)
        myQ1=self.WQ1(thisembeddings)
        myK1=self.WK1(thisembeddings)
        dproduct1= torch.bmm(myQ1, torch.transpose(myK1,1,2)).view(numbatch,numvisit,numcode,numcode)
        dproduct1=dproduct1-sentence[1].view(numbatch,numvisit,1,numcode)-sentence[1].view(numbatch,numvisit,numcode,1)
        sproduct1=self.softmax(dproduct1.view(-1,numcode)/np.sqrt(self.attention_dimensionality)).view(-1,numcode,numcode) 
        fembedding11=torch.bmm(sproduct1,thisembeddings)
        fembedding11=(((sentence[1]-(1e+20))/(-1e+20)).view(-1,numcode,1)*fembedding11)
        mydecay = self.decay[sentence[0].view(-1).to(dtype=torch.int64)].view(numvisit*numbatch,numcode,1)
        myini = self.initial[sentence[0].view(-1).to(dtype=torch.int64)].view(numvisit*numbatch, numcode,1)
        temp1= torch.bmm( mydecay, sentence[2].view(-1,1,1))
        temp2 = self.sigmoid(temp1+myini)   
        vv=torch.bmm(temp2.view(-1,1,numcode),fembedding11)
        vv=vv.view(numbatch,numvisit,-1).transpose(0,1)
        lstm_out, self.hidden = self.lstm(vv, self.hidden)
        out = lstm_out[-1]
        if self.use_lab:
            lab = self.pretrained_layer(sentence[3])
            out = torch.cat((out, lab), axis = 1)
        label_space = self.hidden2label(out)
        label_scores = self.sigmoid(label_space)
        return label_scores

######################################################################
# Train the model:

def train_model(batch_data, val_data, code_to_ix, EMBEDDING_DIM, HIDDEN_DIM, ATTENTION_DIM, EPOCH, batchsize,dropoutrate, use_lab, weight, bias, lab_dim):


    model = Timeline(batchsize, EMBEDDING_DIM, HIDDEN_DIM, ATTENTION_DIM, len(code_to_ix), dropoutrate, use_lab, weight, bias, lab_dim)
    # model.cuda()
    loss_function = nn.BCELoss()
    
    optimizer = optim.Adam(model.parameters())
    
    ep=0
    while ep <EPOCH:  
        model.train()
        for mysentence in batch_data:
            model.zero_grad()
            model.hidden = model.init_hidden()
            # targets = mysentence[1].cuda()
            # label_scores = model((mysentence[0][0].cuda(),mysentence[0][1].cuda(),mysentence[0][2].cuda(), mysentence[0][3].cuda()), 1)
            targets = mysentence[1]
            label_scores = model((mysentence[0][0],mysentence[0][1],mysentence[0][2], mysentence[0][3]), 1)
            loss = loss_function(label_scores, targets.to(dtype = torch.float32))
            loss.backward()
            optimizer.step()
        torch.save(model.state_dict(), timeline_path + '/model_epoch_'+str(ep))         
        print ('finished', ep, 'epochs')
        print ('on validation set:')
        model.eval()
        model.hidden = model.init_hidden()
    
        y_true=[]
        y_pred=[]
        for inputs in val_data:
            model.hidden = model.init_hidden()  
            # tag_scores = model((inputs[0][0].cuda(),inputs[0][1].cuda(),inputs[0][2].cuda(), inputs[0][3].cuda()), 1).data
            tag_scores = model((inputs[0][0],inputs[0][1],inputs[0][2], inputs[0][3]), 1).data
            y_true.append(inputs[1].data)
            pred = torch.argsort(tag_scores, dim=-1, descending=True)
            y_pred.append(pred.numpy())
        y_pred = np.vstack(y_pred)
        y_true = np.vstack(y_true)
        f1_score = f1(y_true, y_pred)
        prec, recall = top_k_prec_recall(y_true, y_pred, ks=[10, 20, 30, 40])
        print('\t', 'f1_score:', f1_score, '\t', 'top_k_recall:', recall)
        ep=ep+1
    print ("training done")
    

def test_model(batch_data, model): 
        model.eval()      
        model.hidden = model.init_hidden()
        
        y_true=[]
        y_pred=[]
        for inputs in batch_data:
            model.hidden = model.init_hidden() 
            # label_scores = model((inputs[0][0].cuda(),inputs[0][1].cuda(),inputs[0][2].cuda(), inputs[0][3].cuda()), 1).data
            label_scores = model((inputs[0][0], inputs[0][1],inputs[0][2], inputs[0][3]), 1).data
            y_true.append(inputs[1].data)
            pred = torch.argsort(label_scores, dim=-1, descending=True)
            y_pred.append(pred.numpy())
        y_pred = np.vstack(y_pred)
        y_true = np.vstack(y_true)
        f1_score = f1(y_true, y_pred)
        prec, recall = top_k_prec_recall(y_true, y_pred, ks=[10, 20, 30, 40])
        print('\t', 'f1_score:', f1_score, '\t', 'top_k_recall:', recall)
       
    
    
def parse_arguments(parser):
    parser.add_argument('--EMBEDDING_DIM', type=int, default=100)
    parser.add_argument('--HIDDEN_DIM', type=int, default=128)
    parser.add_argument('--ATTENTION_DIM', type=int, default=100)
    parser.add_argument('--EPOCH', type=int, default=150)
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--dropoutrate', type=float, default=0.4)
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    use_lab = True
    data_path = os.path.join('data', 'mimic3')
    timeline_path = os.path.join(data_path, 'timeline')
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)
    print (args)

    codes_dataset = pickle.load(open(os.path.join(standard_path, 'codes_dataset.pkl'), 'rb'))
    train_codes_data, valid_codes_data, test_codes_data = codes_dataset['train_codes_data'], codes_dataset['valid_codes_data'], codes_dataset['test_codes_data']
    (_, _, train_lab_x, _) = train_codes_data
    (_, _, valid_lab_x, _) = valid_codes_data
    (_, _, test_lab_x, _) = test_codes_data
    lab_dim = train_lab_x.shape[1]
    weight = np.load(os.path.join(pre_trained, 'saved_weights.npy'))
    bias = np.load(os.path.join(pre_trained, 'saved_bias.npy'))
    weight = torch.from_numpy(weight)
    bias = torch.from_numpy(bias)

    train_visitset, valid_visitset, test_visitset, train_gapset, valid_gapset, test_gapset, train_labelset, valid_labelset, test_labelset = load_file(timeline_path)
    c2ix = code_map(train_visitset, valid_visitset, test_visitset, train_labelset, valid_labelset, test_labelset)
    
    training_data = preprocessing(c2ix, args.batchsize, train_visitset, train_gapset, train_labelset, train_lab_x)
    validation_data = preprocessing(c2ix, args.batchsize, valid_visitset, valid_gapset, valid_labelset, valid_lab_x)
    test_data = preprocessing(c2ix, args.batchsize, test_visitset, test_gapset, test_labelset, test_lab_x)

    train_model(training_data, validation_data, c2ix, args.EMBEDDING_DIM,args.HIDDEN_DIM, args.ATTENTION_DIM,args.EPOCH, args.batchsize,args.dropoutrate, use_lab, weight, bias, lab_dim)
    
    epoch=0
    print ("performance on the test set:")
    while epoch < args.EPOCH:
        model = Timeline(args.batchsize, args.EMBEDDING_DIM, args.HIDDEN_DIM, args.ATTENTION_DIM, len(c2ix), args.dropoutrate, use_lab, weight, bias, lab_dim)
        # model.cuda()
        model.load_state_dict(torch.load(timeline_path + '/model_epoch_'+str(epoch)))
        print ("model after",str(epoch),"epochs")
        test_model(test_data, model)
        epoch=epoch+1
    