# Physics-Informed Neural Network (PINN) for Crop Yield Prediction
## Detailed Code Documentation

### 1. Model Architecture

```python
class CropPINN(nn.Module):
    def __init__(self):
        super(CropPINN, self).__init__()
        # Genetic encoder: Takes 5 genetic features
        self.genetic_encoder = nn.Sequential(
            nn.Linear(5, 64),   # Input layer: 5 genetic features → 64 neurons
            nn.ReLU(),          # Activation function: introduces non-linearity
            nn.Linear(64, 32)   # Output layer: 64 → 32 neurons
        )
        
        # Environmental encoder: Takes 5 environmental features
        self.environmental_encoder = nn.Sequential(
            nn.Linear(5, 64),   # Input layer: 5 environmental features → 64 neurons
            nn.ReLU(),          # Activation function
            nn.Linear(64, 32)   # Output layer: 64 → 32 neurons
        )
        
        # Decoder: Combines genetic and environmental features
        self.decoder = nn.Sequential(
            nn.Linear(64, 32),  # Input: 64 (32+32 from encoders) → 32 neurons
            nn.ReLU(),          # Activation function
            nn.Linear(32, 1)    # Output layer: 32 → 1 (final yield prediction)
        )
```
**Explanation**:
- The model has 3 main components:
  1. **Genetic Encoder**: Processes genetic features (height potential, growth rate, etc.)
  2. **Environmental Encoder**: Processes environmental features (temperature, CO2, etc.)
  3. **Decoder**: Combines both encodings to predict yield

### 2. Forward Pass

```python
def forward(self, genetic_features, environmental_features):
    # Encode genetic features
    genetic_encoding = self.genetic_encoder(genetic_features)
    
    # Encode environmental features
    env_encoding = self.environmental_encoder(environmental_features)
    
    # Combine encodings
    combined = torch.cat((genetic_encoding, env_encoding), dim=1)
    
    # Decode to get yield prediction
    yield_prediction = self.decoder(combined)
    
    return yield_prediction
```
**Explanation**:
- Takes genetic and environmental features as input
- Processes them through respective encoders
- Concatenates the encoded features
- Produces final yield prediction through decoder

### 3. Physics Loss Component

```python
def physics_loss(self, environmental_features):
    # Extract environmental variables
    T_max = environmental_features[:, 0]  # Maximum temperature
    T_min = environmental_features[:, 1]  # Minimum temperature
    rad = environmental_features[:, 2]    # Solar radiation
    CO2 = environmental_features[:, 3]    # CO2 levels
    RH = environmental_features[:, 4]     # Relative humidity
    
    # Initialize crop model
    crop_model = Crop()
    
    # Calculate growth using physics-based model
    fSolar, biomass = crop_model.growth(T_max, T_min, rad, CO2)
    
    return torch.tensor(biomass, dtype=torch.float32)
```
**Explanation**:
- Extracts environmental variables from input features
- Uses SIMPLE crop model to calculate physics-based growth
- Returns biomass prediction based on crop physics

### 4. Training Loop

```python
def train_epoch(model, train_loader, optimizer, alpha=0.5):
    model.train()  # Set model to training mode
    total_loss = 0
    total_mse = 0
    total_physics = 0
    n_batches = 0
    
    for g_batch, e_batch, y_batch in train_loader:
        # Forward pass
        y_pred = model(g_batch, e_batch)
        
        # Calculate data-driven loss (MSE)
        mse_loss = nn.MSELoss()(y_pred, y_batch)
        
        # Calculate physics-based loss
        physics_pred = model.physics_loss(e_batch)
        physics_loss = nn.MSELoss()(y_pred, physics_pred)
        
        # Combined loss with weighting factor alpha
        loss = alpha * mse_loss + (1 - alpha) * physics_loss
        
        # Backward pass and optimization
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()        # Calculate gradients
        optimizer.step()       # Update weights
        
        # Track losses
        total_loss += loss.item()
        total_mse += mse_loss.item()
        total_physics += physics_loss.item()
        n_batches += 1
    
    # Return average losses
    return total_loss/n_batches, total_mse/n_batches, total_physics/n_batches
```
**Explanation**:
- Iterates through batches of training data
- For each batch:
  1. Makes predictions using current model
  2. Calculates both MSE and physics-based losses
  3. Combines losses using weighting factor α
  4. Updates model weights using backpropagation
- Tracks and returns average losses

### 5. Evaluation

```python
def evaluate(model, test_loader, alpha=0.5):
    model.eval()  # Set model to evaluation mode
    predictions = []
    actuals = []
    
    with torch.no_grad():  # Disable gradient calculation
        for g_batch, e_batch, y_batch in test_loader:
            # Make predictions
            y_pred = model(g_batch, e_batch)
            
            # Calculate losses
            mse_loss = nn.MSELoss()(y_pred, y_batch)
            physics_pred = model.physics_loss(e_batch)
            physics_loss = nn.MSELoss()(y_pred, physics_pred)
            
            # Store predictions and actual values
            predictions.extend(y_pred.numpy().flatten())
            actuals.extend(y_batch.numpy().flatten())
    
    # Calculate metrics
    mse = np.mean((np.array(predictions) - np.array(actuals))**2)
    rmse = np.sqrt(mse)
    r2 = r2_score(actuals, predictions)
    
    return mse, rmse, r2
```
**Explanation**:
- Evaluates model performance on test data
- Calculates key metrics:
  - MSE (Mean Squared Error)
  - RMSE (Root Mean Squared Error)
  - R² (Coefficient of Determination)
- No gradient calculation during evaluation (faster and memory-efficient)

### 6. Key Metrics Interpretation

1. **MSE (Mean Squared Error)**
   - Measures average squared difference between predictions and actual values
   - Lower values indicate better predictions
   - Units are squared yield units

2. **RMSE (Root Mean Square Error)**
   - Square root of MSE
   - In same units as yield
   - More interpretable than MSE
   - Example: RMSE of 0.5 means average prediction is off by 0.5 units

3. **R² Score (Coefficient of Determination)**
   - Measures proportion of variance explained by model
   - Range: 0 to 1
   - Interpretation:
     - 0.9-1.0: Excellent fit
     - 0.7-0.9: Good fit
     - 0.5-0.7: Moderate fit
     - <0.5: Poor fit

### 7. Training Tips

1. **Hyperparameter Selection**
   - Learning rate: Start with 0.001
   - Batch size: 32 or 64 typically works well
   - α (loss weight): 0.5 gives equal weight to data and physics

2. **Monitoring Training**
   - Watch for decreasing losses
   - Check if physics loss and MSE loss decrease together
   - Monitor test metrics for overfitting

3. **Common Issues**
   - If losses don't decrease: Try lower learning rate
   - If physics loss stays high: Check crop model parameters
   - If training is unstable: Reduce batch size or learning rate
