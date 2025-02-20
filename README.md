# AI-powered Smart Farming Dashboard

A minor project (CSD0603) by Bhuvnesh Pal \

## Overview
This project develops an AI-powered dashboard to assist farmers with crop recommendations, yield predictions, and fertilizer optimization using historical IoT datasets. This prototype implements crop recommendation using Graph Neural Networks (GNNs).

## Setup
1. Place `crop_recommendation.csv` in the `data/` folder (download from Kaggle).
2. Install dependencies: `pip install -r requirements.txt`.
3. Train models: `python train.py`.
4. Get recommendations: `python predict.py`.

## Files
- `models/`: GCN and Embedding Predictor definitions.
- `utils/`: Data processing helpers.
- `train.py`: Trains GNN and auxiliary models.
- `predict.py`: Predicts crops for new inputs.

## Future Enhancements
- Yield prediction with TCNs/Transformers.
- Climate impact with Climformer + probabilistic forecasting.
- Fertilizer optimization with Genetic Algorithms.
- Adaptive learning via Federated Learning.
