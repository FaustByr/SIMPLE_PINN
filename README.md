# SIMPLE_PINN: Physics-Informed Neural Networks for Crop Modeling

A sophisticated implementation of the SIMPLE crop growth model using Physics-Informed Neural Networks (PINNs) for accurate crop yield prediction.

## Project Overview

This project combines traditional crop modeling with modern deep learning techniques by implementing a Physics-Informed Neural Network approach to crop yield prediction. The model incorporates both data-driven learning and physics-based constraints from the SIMPLE crop growth model.

## Features

- **Hybrid Architecture**: Combines neural networks with physics-based crop modeling
- **Dual Encoding**: Separate encoders for genetic and environmental features
- **Physics-Constrained Learning**: Incorporates SIMPLE crop model physics into loss function
- **Modular Design**: Easily modifiable architecture for different crop types and conditions

## Model Architecture

### Input Features

#### Genetic Features (5D)
- Height potential
- Growth rate
- Drought resistance
- Disease resistance
- Nutrient efficiency

#### Environmental Features (5D)
- Maximum temperature
- Minimum temperature
- Solar radiation
- CO2 levels
- Relative humidity

### Neural Network Structure
- Genetic Encoder: 5 → 64 → 32 neurons
- Environmental Encoder: 5 → 64 → 32 neurons
- Decoder: 64 → 32 → 1 neuron

## Physics Components

The model incorporates the following physical responses from the SIMPLE crop model:
- Temperature response
- Solar radiation response
- CO2 response
- Heat stress response
- Water stress response

## Installation

1. Clone the repository:
```bash
git clone https://github.com/FaustByr/SIMPLE_PINN.git
cd SIMPLE_PINN
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies

- numpy>=1.21.0
- pandas>=1.3.0
- torch>=1.9.0
- matplotlib>=3.4.0
- scikit-learn>=0.24.0
- jupyter>=1.0.0
- notebook>=6.4.0

## Documentation

- Detailed code documentation: See `documentation.md`
- Model architecture details: See `model_architecture.md`

## Performance Metrics

The model's performance is evaluated using:
- Mean Squared Error (MSE)
- Physics Loss
- R² Score
- Root Mean Square Error (RMSE)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- SIMPLE crop model developers
- Physics-Informed Neural Networks research community
- Contributors and maintainers

## Contact

Project Link: [https://github.com/FaustByr/SIMPLE_PINN](https://github.com/FaustByr/SIMPLE_PINN)
