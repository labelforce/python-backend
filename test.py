

from firebase import Firebase

f = Firebase('https://boiling-heat-2521.firebaseio.com/', auth_token="eoyakpIFmf4LTm6JcUPElixc8ieeQujvDF7bCGNh")


r = f.push({'user_id': 'wilma', 'text': 'Hello'})