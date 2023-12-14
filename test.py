# This file is used to verify your http server acts as expected
# Run it with `python3 test.py``

import requests

model_inputs = {'prompt': 'My favorite part about working with AI is'}

res = requests.post('http://localhost:8000/', json = model_inputs)

try:
    print(res.json())
except:
    print(res.code)
    print(res.text)