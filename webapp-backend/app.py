from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import numpy as np
import os
import logging

app = Flask(__name__)
CORS(app)  #cors 4 frontend connection

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enhanced RNN Model Architecture
class EnhancedCricketRNN(nn.Module):
    def __init__(self, input_size, hidden_size=512, num_layers=4, dropout=0.3):
        super(EnhancedCricketRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn1 = nn.BatchNorm1d(hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.bn2 = nn.BatchNorm1d(hidden_size // 4)
        self.fc3 = nn.Linear(hidden_size // 4, hidden_size // 8)
        self.bn3 = nn.BatchNorm1d(hidden_size // 8)
        self.fc4 = nn.Linear(hidden_size // 8, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        last_output = attn_out[:, -1, :]
        
        out = self.fc1(last_output)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc4(out)
        
        return out.squeeze()

# Global variables for model
model = None
feature_mean = None
feature_std = None
device = None

def load_model():
    """Load the trained model from the saved checkpoint"""
    global model, feature_mean, feature_std, device
    
    try:
        # Determine device
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        
        logger.info(f"Using device: {device}")
        
        # Model path - THIS IS THE KEY: Load the saved model from training
        model_path = '../data/processed/enhanced_cricket_model.pth'
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        logger.info(f"Loading model from: {model_path}")
        
        # Load checkpoint (saved by 05_enhanced_model_training.ipynb)
        checkpoint = torch.load(model_path, map_location=device)
        
        # Initialize model with correct input size (19 enhanced features)
        input_size = 19
        model = EnhancedCricketRNN(input_size=input_size)
        
        # Load the trained weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        # Load normalization parameters (saved during training)
        feature_mean = checkpoint.get('feature_mean', torch.zeros(input_size))
        feature_std = checkpoint.get('feature_std', torch.ones(input_size))
        
        logger.info("Model loaded successfully")
        logger.info(f"Model input size: {input_size}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

def calculate_advanced_features(match_data):
    """Calculate all 19 advanced features from match data (same as training)"""
    
    overs_data = []
    
    # Basic over-by-over data
    for i, over in enumerate(match_data['overs']):
        over_data = {
            'over': i,
            'runs_in_over': over['runs'],
            'wickets_in_over': over['wickets'],
            'cumulative_runs': sum([o['runs'] for o in match_data['overs'][:i+1]]),
            'cumulative_wickets': sum([o['wickets'] for o in match_data['overs'][:i+1]]),
            'overs_remaining': 20 - (i + 1),
            'run_rate': sum([o['runs'] for o in match_data['overs'][:i+1]]) / (i + 1),
            'momentum': 0.0,  # Will calculate below
            'batsman_sr': match_data.get('batsman_sr', 120.0),
            'bowler_economy': match_data.get('bowler_economy', 7.5)
        }
        
        # Phase encoding (same as training)
        if i < 6:
            over_data.update({'phase_Powerplay': 1, 'phase_Middle': 0, 'phase_Death': 0})
        elif i < 15:
            over_data.update({'phase_Powerplay': 0, 'phase_Middle': 1, 'phase_Death': 0})
        else:
            over_data.update({'phase_Powerplay': 0, 'phase_Middle': 0, 'phase_Death': 1})
        
        overs_data.append(over_data)
    
    # Calculate advanced momentum features (EXACT same as training)
    for i, over_data in enumerate(overs_data):
        # 3-over momentum
        if i >= 3:
            recent_rr = np.mean([overs_data[j]['runs_in_over'] for j in range(i-3, i)])
            overall_rr = over_data['run_rate']
            over_data['rr_momentum_3'] = recent_rr - overall_rr
        else:
            over_data['rr_momentum_3'] = 0.0
        
        # 5-over momentum
        if i >= 5:
            recent_rr = np.mean([overs_data[j]['runs_in_over'] for j in range(i-5, i)])
            overall_rr = over_data['run_rate']
            over_data['rr_momentum_5'] = recent_rr - overall_rr
        else:
            over_data['rr_momentum_5'] = 0.0
        
        # Acceleration
        if i >= 1:
            prev_rr = overs_data[i-1]['run_rate']
            curr_rr = over_data['run_rate']
            over_data['acceleration'] = curr_rr - prev_rr
        else:
            over_data['acceleration'] = 0.0
        
        # Pressure index
        wickets_factor = over_data['cumulative_wickets'] / 10
        overs_factor = (i + 1) / 20
        over_data['pressure_index'] = wickets_factor * 0.6 + overs_factor * 0.4
        
        # Wicket cluster
        if i >= 4:
            over_data['wicket_cluster'] = sum([overs_data[j]['wickets_in_over'] for j in range(i-4, i+1)])
        else:
            over_data['wicket_cluster'] = 0.0
        
        # Economy pressure
        if i >= 2:
            avg_economy = match_data.get('bowler_economy', 7.5)
            over_data['economy_pressure'] = max(0, (8 - avg_economy) / 8)
        else:
            over_data['economy_pressure'] = 0.0
    
    return overs_data

def prepare_model_input(overs_data):
    """Prepare input for the model (EXACT same feature order as training)"""
    
    # Feature order MUST match training exactly
    feature_order = [
        'over', 'runs_in_over', 'wickets_in_over', 'cumulative_runs', 
        'cumulative_wickets', 'overs_remaining', 'run_rate', 'momentum',
        'batsman_sr', 'bowler_economy', 'phase_Powerplay', 'phase_Middle', 
        'phase_Death', 'rr_momentum_3', 'rr_momentum_5', 'acceleration', 
        'pressure_index', 'wicket_cluster', 'economy_pressure'
    ]
    
    # Convert to feature matrix
    sequence = []
    for over_data in overs_data:
        over_features = []
        for feature in feature_order:
            over_features.append(over_data.get(feature, 0.0))
        sequence.append(over_features)
    
    # Pad to length 20 (same as training)
    max_length = 20
    if len(sequence) > max_length:
        sequence = sequence[-max_length:]
    else:
        padding = [[0.0] * len(feature_order)] * (max_length - len(sequence))
        sequence = padding + sequence
    
    # Convert to tensor and normalize (same as training)
    sequence = np.array(sequence)
    
    # Normalize using training statistics
    sequence = (sequence - feature_mean.numpy()) / feature_std.numpy()
    
    return torch.FloatTensor(sequence).unsqueeze(0)

# ===== NEW ROUTES TO FIX 404 ERRORS =====

@app.route('/')
def home():
    """Root endpoint - API information"""
    return jsonify({
        'name': 'CricPredict Pro API',
        'version': '1.0.0',
        'status': 'running',
        'model_loaded': model is not None,
        'device': str(device) if device else None,
        'endpoints': {
            '/': 'API information',
            '/health': 'Health check',
            '/predict': 'Cricket score prediction (POST)',
            '/api/info': 'Detailed API information'
        },
        'model_info': {
            'input_features': 19,
            'parameters': f"{sum(p.numel() for p in model.parameters()):,}" if model else "Model not loaded",
            'architecture': 'Enhanced RNN with LSTM + Attention'
        }
    })

@app.route('/favicon.ico')
def favicon():
    """Handle favicon requests"""
    return '', 204

@app.route('/api/info')
def api_info():
    """Detailed API information"""
    return jsonify({
        'api_name': 'CricPredict Pro',
        'description': 'Advanced cricket score prediction using Enhanced RNN',
        'version': '1.0.0',
        'model_status': 'loaded' if model is not None else 'not_loaded',
        'device': str(device) if device else 'unknown',
        'endpoints': {
            'GET /': 'API overview',
            'GET /health': 'Health check',
            'GET /api/info': 'This endpoint',
            'POST /predict': 'Score prediction'
        },
        'predict_endpoint': {
            'method': 'POST',
            'url': '/predict',
            'required_fields': ['overs'],
            'optional_fields': ['target', 'batsman_sr', 'bowler_economy'],
            'example_request': {
                'overs': [
                    {'runs': 8, 'wickets': 0},
                    {'runs': 12, 'wickets': 1},
                    {'runs': 6, 'wickets': 0}
                ],
                'target': 180,
                'batsman_sr': 125.0,
                'bowler_economy': 7.2
            }
        }
    })

# ===== EXISTING ROUTES =====

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy', 
        'model_loaded': model is not None,
        'device': str(device) if device else None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.json
        
        # Validate input
        if 'overs' not in data or len(data['overs']) == 0:
            return jsonify({'error': 'No over data provided'}), 400
        
        logger.info(f"Received prediction request for {len(data['overs'])} overs")
        
        # Calculate advanced features (same as training)
        overs_data = calculate_advanced_features(data)
        
        # Prepare model input (same format as training)
        model_input = prepare_model_input(overs_data).to(device)
        
        # Make prediction using the trained model
        with torch.no_grad():
            remaining_runs = model(model_input).item()
        
        # Calculate results
        current_score = sum([over['runs'] for over in data['overs']])
        predicted_final = current_score + max(0, remaining_runs)
        
        # Additional insights
        current_over = len(data['overs'])
        current_rr = current_score / current_over if current_over > 0 else 0
        overs_left = 20 - current_over
        
        # Win probability (simplified)
        win_prob = None
        if 'target' in data:
            target = data['target']
            runs_needed = target - current_score
            win_prob = min(95, max(5, 100 * (1 - runs_needed / max(1, predicted_final - current_score))))
        
        response = {
            'predicted_final_score': round(predicted_final, 1),
            'remaining_runs': round(max(0, remaining_runs), 1),
            'current_score': current_score,
            'current_run_rate': round(current_rr, 2),
            'overs_completed': current_over,
            'overs_remaining': overs_left,
            'win_probability': round(win_prob, 1) if win_prob is not None else None
        }
        
        logger.info(f"Prediction successful: {response}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Load model on startup
if __name__ == '__main__':
    print("Starting CricPredict Pro Backend...")
    print("Loading trained Enhanced RNN model...")
    
    if load_model():
        print("✅ Model loaded successfully!")
        print(f"🚀 Server starting on http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load model. Please check:")
        print("1. Model file exists at '../data/processed/enhanced_cricket_model.pth'")
        print("2. Model was trained and saved correctly")
        print("3. File path is correct")
        exit(1)