'''
Created on 2 Oct 2015

@author: timdettmers
'''

from firebase import firebase
firebase = firebase.FirebaseApplication('https://boiling-heat-2521.firebaseio.com', None)
result = firebase.get('/users', None)

result = firebase.post('/users', 'aasdfdsaf')

print result