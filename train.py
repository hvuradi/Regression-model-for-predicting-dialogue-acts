import csv
import nltk
from collections import OrderedDict
from sklearn import cluster
from sklearn import linear_model

with open(r'DialogueActs.csv', "r") as f:
    reader = csv.reader(f)
    l2 = []
    l1 = list(reader)
    l3 = []
    for r in l1:
        l2.append(r[0])
        l3.append(r[1])
    pos_list = []
    feature = OrderedDict()
    feature_vectors = []
    F = []
    for sen in range(0,len(l2)):
        text = nltk.word_tokenize(l2[sen])
        pos = nltk.pos_tag(text)
        for tag in pos:
            if tag[1] not in feature:
                feature[tag[1]] = 0
    
    feature_vectors = []
    one_hot_vectors = []
    total_vectors = []
    dialog_act_dict = {
        'N' : [1,0,0],
        'R' : [0,1,0],
        'L' : [0,0,1]
    }
    for s in range(1,len(l2)):
        text = nltk.word_tokenize(l2[s])
        pos = nltk.pos_tag(text)
        
        vector = [0]*(len(feature))
        for tag in pos:
            index = list(feature.keys()).index(tag[1])
            vector[index] = vector[index] + 1
            
        one_hot_vectors.append(dialog_act_dict[l3[s]])
        feature_vectors.append(vector)
        total_vectors.append((vector, dialog_act_dict[l3[s]]))

    with open('features.csv', 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        for val in total_vectors:
            writer.writerow(val)
    
    lm = linear_model.LinearRegression()
    model = lm.fit(feature_vectors,one_hot_vectors)
    print(model)
    predictions = lm.predict(feature_vectors)
    with open('predictions.csv', 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        for v in predictions:
            writer.writerow(v)
    print(predictions)
