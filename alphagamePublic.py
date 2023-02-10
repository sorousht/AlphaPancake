from flask import Flask, request,jsonify
import torch
from Pancake.Pytorch.NNet import NNetWrapper as NNet

from PancakeGame import PancakeGame as Game
import numpy as np
import json
from flask_cors import CORS
from MCTS import MCTS
from utils import *
import Arena
app = Flask(__name__)
CORS(app,origins="*")
g = Game(5)
# nnet players
custom_model = NNet(g)
g = Game(5)
custom_model = NNet(g)
custom_model.load_checkpoint('/Users/apple/Documents','best.pth.tar')
args1 = dotdict({'numMCTSSims': 20, 'cpuct':1,'minDepth':20})
mcts1 = MCTS(g, custom_model, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0)[0])
#custom_model.load_checkpoint('./','checkpoint_93.pth.tar')
#@app.route("/predict", methods=["GET",POST"])
def predict(input_data):
    # Load your PyTorch model
    # ...

    # Get input data from the request
    #g = Game(5)
    #custom_model = NNet(g)
    #custom_model.load_checkpoint('/Users/apple/Documents','best.pth.tar')
    print(input_data,"hallo")
    input_data = list(map(int, input_data))
    print(input_data)
    input_data = np.array(input_data)
    print(input_data)
    if g.getGameEnded(input_data)==1:
        print(0)
        return 0
    #t = g.getTensorFromBoards(np.array(input_data))
    #args1 = dotdict({'numMCTSSims': 20, 'cpuct':1,'minDepth':20})
    #mcts1 = MCTS(g, custom_model, args1)
    #n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0)[0])
    action = n1p(input_data)
    print(action)
    #print(t.shape)
    #predictions = custom_model.predict(t)
    #print(predictions)
    #max_index = np.argmax(predictions[0])
    #input_data = g.flip(input_data,max_index)
    
    # Use the model to make predictions
    # ...

    # Return the predictions as a response
    return action
@app.route("/do_predict", methods=["POST"])
#@cross_origin()
def do_predict():
    # Get the input array from the request body
    print("hallo")
    
    '''
    json_string = request.get_data().decode()
    data = json.loads(json_string)
    #input_array = data['input_array']
    print(data)
    # Call the predict function
    result = predict(data.get('input_array'))
    '''
    input_data = request.get_json()
    input_string = input_data.get("input_string")
    result = predict(input_string)
    # Return the result as JSON
    print(result)
    response = jsonify({"output_string": str(result)})
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Methods", "POST")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Credentials", "true")
    
    
    return response
if __name__ == "__main__":
    app.run(debug=True,port=5000)
