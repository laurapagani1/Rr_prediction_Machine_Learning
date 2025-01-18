# Rr_prediction_Machine_Learning: Ship Resistance Calculator

## Overview
This application is a machine learning-powered tool designed to predict ship resistance (Rr) based on various hull design parameters. It provides naval architects and marine engineers with a quick and intuitive way to estimate a vessel's resistance through water using key hydrostatic parameters.

## Features
- Interactive slider-based input for six key ship parameters
- Real-time resistance (Rr) prediction using machine learning
- Radar chart visualization of input parameters
- Detailed explanations of all ship design parameters
- Responsive web interface built with Streamlit

## Technical Details

### Input Parameters
- **LC (Longitudinal Center of Buoyancy)**: Position of buoyancy center along ship length
- **PC (Prismatic Coefficient)**: Hull volume to prism volume ratio
- **L/D (Length-to-Displacement ratio)**: Indicates hull slenderness
- **B/Dr (Beam-to-Draft ratio)**: Width to depth ratio
- **L/B (Length-to-Beam ratio)**: Length to width ratio
- **Fr (Froude number)**: Dimensionless speed-length ratio

### Machine Learning Model
- Implemented using Decision Tree Regressor
- Trained on 308 hull data points
- Features standardized using StandardScaler
- Model performance metrics tracked (MAE, MSE, R²)

## Project Structure
```
├── main.py                 # Streamlit web application
├── train.py               # Model training script
├── assets/
│   └── style.css         # Custom styling
├── data/
│   └── yacht_hydro.csv   # Training dataset
└── model/
    ├── model.pkl         # Trained model
    └── scaler.pkl        # Feature scaler
```

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run main.py
```

## Usage

1. Launch the application
2. Use the sliders in the sidebar to adjust ship parameters
3. View the radar chart showing relative parameter values
4. Check the predicted resistance value
5. Read detailed parameter explanations in the main panel

## Dependencies
- streamlit
- pandas
- numpy
- plotly
- scikit-learn
- pickle

## Model Training

To retrain the model with new data:

1. Place your data in `data/yacht_hydro.csv`
2. Run the training script:
```bash
python train.py
```

## Data Format
The application expects CSV data with the following columns:
- LC: Longitudinal Center of Buoyancy
- PC: Prismatic Coefficient
- L/D: Length-to-Displacement ratio
- B/Dr: Beam-to-Draft ratio
- L/B: Length-to-Beam ratio
- Fr: Froude number
- Rr: Resistance (target variable)

## Technical Implementation

### Model Architecture
The application uses a Decision Tree Regressor model trained on standardized features. The model was chosen for its:
- Ability to capture non-linear relationships
- Good performance on the given dataset
- Interpretability of results

### Data Processing
- Features are standardized using StandardScaler
- Model input is dynamically scaled using saved scaler parameters
- Real-time predictions are made on user input

### Visualization
- Interactive radar chart using Plotly
- Normalized parameter visualization
- Real-time updates based on user input

## Limitations
- Predictions are based on training data range
- Model assumes conventional hull forms
- Results should be verified with computational fluid dynamics (CFD) or tank testing

## Future Improvements
- Integration with CFD validation
- Support for more hull parameters
- Advanced visualization options
- Batch prediction capability
- Model uncertainty estimation

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
[Your chosen license]

## Contact
[Your contact information]

---
*This tool is intended for preliminary design estimates only and should not be used as the sole basis for final design decisions.*
