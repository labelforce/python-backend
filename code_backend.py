'''
Created on 3 Oct 2015

@author: timdettmers
'''
import cPickle as pickle
import numpy as np
import time
import Queue
from firebase import firebase
f = firebase.FirebaseApplication('https://boiling-heat-2521.firebaseio.com', None)





'''
data = np.random.randint(0,9,(500,))
for i, lbl in enumerate(data):
    f.post('net'.format(i), { 'a' + str(i) : lbl })
    
pickle.dump(f.get('net', None),open('/home/tim/net.p','wb'))
'''
data = pickle.load(open('/home/tim/net.p','r'))

print len(data.keys())

net = {}
for key in data:
    net[data[key].keys()[0]] = key


codes = np.float32(pickle.load(open('/home/tim/codes.p','r')))[0:500]
label_init = pickle.load(open('/home/tim/codesy.p','r'))[0:20].flatten().tolist()


f.delete('labelme', None)
f.delete('swiped', None)

#for i, lbl in enumerate(label_init):
    #f.post_async('net'.format(i), { 'a' + str(i) : lbl})

true_labels = {}

for i, lbl in enumerate(label_init):
    true_labels[i] = lbl

'''
neighbors = {}

t0 = time.time()

print codes.shape

a = gpu.array(codes)
buffer = gpu.zeros_like(a)
vec_buffer = gpu.zeros((codes.shape[0],1))


for i, code in enumerate(codes):
    vec = gpu.array(np.float32(np.tile(code,(500,1))))
    gpu.subtract(a,vec,buffer)
    gpu.abs(buffer,buffer)
    gpu.sum(buffer,1, vec_buffer)
    dist = vec_buffer.data
    if i % 1000 == 0: 
        print int(time.time()-t0)
        t0 = time.time()
        print i
        
    neighbors[i] = np.argsort(dist,0)[0:500].flatten().tolist()
    
    
pickle.dump(neighbors,open('/home/tim/neighbors.p','wb'))
'''
neighbors = pickle.load(open('/home/tim/neighbors.p','r'))   

guess_labels = {}

swipe_cache = {}


def get_guess():
    overlap = {}
    pure = {}
    for idx in true_labels:
    
        lbl = true_labels[idx]
        #print lbl
        candidates = neighbors[idx][1:15]
        #print candidates
        for candidate in candidates:
            #print candidate
            if candidate in overlap:
                overlap[candidate].append(lbl)
            else:
                if candidate not in pure: pure[candidate] = lbl
                else: overlap[candidate] = [lbl,pure.pop(candidate, None)]
        
    return [overlap , pure]


labeled_count = {}
while True:
        
    overlap, pure = get_guess()
    changes = []
    
    for idx in pure:
        if idx not in guess_labels:
            guess_labels[idx] = pure[idx]
            changes.append([idx,pure[idx]])
        
        
    priority_guess = {}
    
    
    for key in overlap:
        
        buckets = np.bincount(overlap[key])
        if (np.max(buckets)/np.sum(buckets)) > 0.8:
            priority_guess[key] = np.argmax(buckets)
    
    
    q = Queue.Queue()
    
    for idx in priority_guess.keys():
        q.put([idx, priority_guess[idx]])
        
    for idx in guess_labels.keys():
        q.put([idx, guess_labels[idx]])
        
        
    print "queue size: {0}; true_label size: {1}, guess size: {2}, changes length: {3}".format(q.qsize(),len(true_labels.keys()), len(guess_labels.keys()), len(changes))
    while q.qsize() > 0:
        idx, lbl = q.get()
        if idx not in labeled_count: labeled_count[idx] = 0
        #print labeled_count[idx]
        if labeled_count[idx] >= 3: continue
        labeled_count[idx] += 1
        
        #if np.random.rand(1) > 0.95:
            #true_labels[idx] = lbl
        f.post_async('labelme',{ idx : lbl})
        
        results = f.get('swiped', None)
        if not results: continue
        for value in results.values():
            if value['picture_id'] not in swipe_cache: swipe_cache[value['id']] = (0,[])
            swipe_cache[value['picture_id']][0] += 1
            swipe_cache[value['picture_id']][1].append(value['label'])
            if swipe_cache >= 1: 
                true_labels[value['picture_id']] = value['label']
                if 'a' + str(value['picture_id']) in net:
                    f.delete('net',net['a' + str(value['picture_id'])])
                    f.post_async('net'.format(i), { 'a' + str(value['picture_id']) : value['label']})
                    hash_value = f.get('net',None).keys()[-1]
                    net['a' + str(value['picture_id'])] = hash_value
            
    for idx, lbl in changes:
        f.delete('net','a'+str(i))
        
        if 'a' + str(idx) in net:
            f.delete('net',net['a' + str(idx)])
            f.post_async('net'.format(i), { 'a'+str(i) : lbl})
            hash_value = f.get('net',None).keys()[-1]
            net['a' + str(idx)] = hash_value

    print 'saving idx...'    
    pickle.dump(f.get('net', None),open('/home/tim/net.p','wb'))
    print 'saved idx!'


    
    
      
      
    
    
