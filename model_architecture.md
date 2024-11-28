# PINN Crop Model Architecture

## Model Overview
```mermaid
graph TD
    subgraph Input
        G[Genetic Features<br/>5 dimensions] --> GE
        E[Environmental Features<br/>5 dimensions] --> EE
        E --> PCM
    end

    subgraph Neural_Network[Neural Network Component]
        GE[Genetic Encoder<br/>5→64→32] --> C
        EE[Environmental Encoder<br/>5→64→32] --> C
        C[Concatenate<br/>64 dimensions] --> D
        D[Decoder<br/>64→32→1] --> YP
        YP[Yield Prediction]
    end

    subgraph Physics_Model[SIMPLE Crop Model]
        PCM[Physics Crop Model] --> PP
        PP[Physics Prediction]
    end

    subgraph Loss_Calculation[Loss Components]
        YP --> MSE
        PP --> PL
        GT[Ground Truth] --> MSE
        YP --> PL
        MSE[MSE Loss] --> TL
        PL[Physics Loss] --> TL
        TL[Total Loss<br/>α×MSE + (1-α)×Physics]
    end

    style Neural_Network fill:#e6f3ff,stroke:#4a90e2
    style Physics_Model fill:#ffe6e6,stroke:#e24a4a
    style Loss_Calculation fill:#e6ffe6,stroke:#4ae24a
```

## Detailed Component Breakdown

### 1. Input Features
```mermaid
graph LR
    subgraph Genetic_Features[Genetic Features]
        G1[Height Potential]
        G2[Growth Rate]
        G3[Drought Resistance]
        G4[Disease Resistance]
        G5[Nutrient Efficiency]
    end

    subgraph Environmental_Features[Environmental Features]
        E1[Max Temperature]
        E2[Min Temperature]
        E3[Solar Radiation]
        E4[CO2 Levels]
        E5[Relative Humidity]
    end
```

### 2. SIMPLE Crop Model Components
```mermaid
graph TD
    subgraph SIMPLE_Model[SIMPLE Crop Model]
        T[Temperature<br/>Response] --> B
        S[Solar Radiation<br/>Response] --> B
        C[CO2<br/>Response] --> B
        H[Heat Stress<br/>Response] --> B
        W[Water Stress<br/>Response] --> B
        B[Biomass<br/>Accumulation]
    end

    subgraph Response_Functions[Response Functions]
        T1[fTemp = 0 if T < Tbase<br/>Linear increase to Topt]
        S1[fSolar = Sigmoid function<br/>of thermal time]
        C1[fCO2 = Linear response<br/>350-700 ppm]
        H1[fHeat = Linear decline<br/>above Theat]
        W1[fWater = Function of<br/>soil water content]
    end
```

### 3. Loss Calculation Detail
```mermaid
graph LR
    subgraph Loss_Components[Loss Components]
        MSE[MSE Loss<br/>Data-driven] --> TL
        PL[Physics Loss<br/>Model-driven] --> TL
        TL[Total Loss<br/>Weighted Sum]
    end

    subgraph Weights[Loss Weights]
        A[α: MSE weight]
        B[1-α: Physics weight]
    end
```

## Training Process Flow
```mermaid
sequenceDiagram
    participant I as Input Data
    participant NN as Neural Network
    participant PM as Physics Model
    participant L as Loss Calculator
    participant O as Optimizer

    I->>NN: Feed features
    NN->>L: Yield prediction
    I->>PM: Environmental features
    PM->>L: Physics-based prediction
    L->>O: Calculate total loss
    O->>NN: Update weights
```

## Key Features
1. **Hybrid Architecture**
   - Combines data-driven (Neural Network) and physics-based (SIMPLE) approaches
   - Ensures predictions are both accurate and physically meaningful

2. **Dual Loss Function**
   - MSE Loss: Measures prediction accuracy against actual data
   - Physics Loss: Ensures compliance with crop growth principles
   - Weighted combination allows balance between data and physics

3. **Feature Processing**
   - Separate encoders for genetic and environmental features
   - Allows model to learn specific patterns in each domain
   - Combined decoder for integrated yield prediction

4. **Physics Integration**
   - SIMPLE model provides mechanistic constraints
   - Key processes: temperature response, CO2 fertilization, radiation use
   - Helps prevent physically impossible predictions
