from nltk.corpus import brown
import random

# Get sentences from the Brown Corpus
sentences = brown.sents()
import torch
random.seed(4231312)
# Count the number of sentences
num_sentences = len(sentences)
i=0
sentence = sentences[0]
mapw={}
mapc={}

mapw['<S>'] = torch.randn(len(sentence),requires_grad=True)
mapc['<S>'] = torch.randn(len(sentence),requires_grad=True)

mapw['<E>'] = torch.randn(len(sentence),requires_grad=True)
mapc['<E>'] = torch.randn(len(sentence),requires_grad=True)

for i in sentence:
    mapw[i] = torch.randn(len(sentence),requires_grad=True)
    mapc[i] = torch.randn(len(sentence),requires_grad=True)



training_examples=[]

vocab={}

i=0
while i in range(len(sentence)):
    target=[]
    if i-1<0:
        target.append('<S>')
    else:
        target.append(sentence[i-1])
    target.append(sentence[i])
    if i+1 == len(sentence):
        target.append('<E>')
    else:
        target.append(sentence[i+1])
        

    training_examples.append(target)
    i+=1



def loss(w,c,currint):

    
    pos_prob = torch.log(torch.sigmoid(torch.dot(mapw[w],mapc[c[currint]])))
    neg_words=[]

    while(len(neg_words)!=2):
        x = random.randint(0,len(sentence)-1)
        if(sentence[x] not in c and sentence[x]!=w and sentence[x] not in neg_words):
            neg_words.append(sentence[x])
    neg_prob=0
    for i in range(2):
        neg_prob+=torch.log(torch.sigmoid(torch.dot(-mapc[neg_words[i]], mapw[w])))

    total_loss = -(pos_prob+neg_prob)
    print(total_loss)
    for i in range(2):
        mapc[neg_words[i]].grad=None
    mapw[w].grad=None
    total_loss.backward()

    for i in range(2):
        mapc[neg_words[i]].data = mapc[neg_words[i]] - 0.1*mapc[neg_words[i]].grad
    mapw[w].data = mapw[w] - 0.1*mapw[w].grad

for i in range(75): 
    for i in training_examples:
        w = i[1]
        c = []
        c.append(i[0])
        c.append(i[2])

        loss(w,c,0)
        loss(w,c,1)

print(sentence)

for i in range(5):
    w = input("word")
    c = input("context1")
    prob=torch.sigmoid(torch.dot(mapw[w],mapc[c]))
    print(prob)

