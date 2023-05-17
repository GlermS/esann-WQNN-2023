import json
def store(result, filename ='results.json'):
    with open(filename,'w') as file:
        file.write(json.dumps(result))
        
def load(filename ='results.json'):
    with open(filename,'r') as file:
        n = json.loads(file.read())
        results = {eval(k): n[k] for k in n.keys()}
    return results
        
def storeModel(model, filename):
    with open(filename,'w') as file:
        file.write(model.json())