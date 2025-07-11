from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import os
from werkzeug.utils import secure_filename
import tempfile

app = Flask(__name__)
UPLOAD_FOLDER = tempfile.gettempdir()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class SimpleLinearModel(nn.Module):
    def __init__(self, state_dict):
        super().__init__()
        # Build linear layers from keys like 'layer1.weight', 'layer1.bias'
        layers = []
        i = 0
        while True:
            w_key = f'layer{i}.weight'
            b_key = f'layer{i}.bias'
            if w_key in state_dict and b_key in state_dict:
                w = state_dict[w_key]
                b = state_dict[b_key]
                layers.append(nn.Linear(w.shape[1], w.shape[0]))
                i += 1
            else:
                break
        self.linears = nn.ModuleList(layers)
        # Load weights manually
        for idx, layer in enumerate(self.linears):
            layer.weight.data = state_dict[f'layer{idx}.weight']
            layer.bias.data = state_dict[f'layer{idx}.bias']

    def forward(self, x):
        outputs = []
        for layer in self.linears:
            x = layer(x)
            outputs.append(x.detach().tolist())
        return outputs

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        state_dict = torch.load(filepath, map_location='cpu')
        # Create nodes for visualization
        nodes = []
        for i, (name, tensor) in enumerate(state_dict.items()):
            nodes.append({
                'data': {'id': str(i), 'label': f"{name}\n{list(tensor.shape)}"}
            })
        return jsonify({'nodes': nodes, 'edges': []})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/simulate', methods=['POST'])
def simulate():
    """
    Expects JSON:
    {
      "input": [float, float, ...],
      "state_dict_path": "optional"
    }
    """
    data = request.json
    user_input = data.get('input')
    if user_input is None:
        return jsonify({'error': 'No input provided'}), 400

    # For demo: build dummy linear model with 2 layers
    # In real app, load model from uploaded file or predefined

    # Build example model with weights matching input size
    input_tensor = torch.tensor(user_input).float().unsqueeze(0)  # batch 1

    # Dummy 2 layer model hardcoded for simulation (replace as needed)
    model = nn.Sequential(
        nn.Linear(input_tensor.shape[1], 4),
        nn.ReLU(),
        nn.Linear(4, 1)
    )
    # Simulate forward
    outputs = []
    x = input_tensor
    for layer in model:
        x = layer(x)
        outputs.append(x.detach().tolist())

    return jsonify({'outputs': outputs})

if __name__ == '__main__':
    app.run(debug=True)
