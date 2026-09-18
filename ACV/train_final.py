import numpy as np
import pandas as pd
import joblib
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_data
from features import build_feature_matrix
from models import (
    LightGBMRanker, XGBoostRanker, ClassicalEnsemble, 
    UnsupervisedAnomalyDetector, SiameseRanker,
    train_all_models, evaluate_all_models, optimize_ensemble_weights,
    evaluate_ranking
)


def train_final_model():
    """Train final ensemble on all training data and save models."""
    print("=" * 60)
    print("ACV Refrigerant Leak Localisation - Final Training")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading data...")
    data = load_data()
    
    # Build features
    print("2. Building feature matrix...")
    X, y, groups, car_ids, feature_names, X_test, test_car_ids = build_feature_matrix(
        data['train_cases'], data['test_df']
    )
    print(f"   Train: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"   Faulty cars: {y.sum()} / {len(y)}")
    print(f"   Test: {X_test.shape[0]} cars")
    
    # Train all models on full data
    print("\n3. Training all models on full dataset...")
    models = train_all_models(X, y, groups, feature_names)
    
    # Evaluate on training data (for reference)
    print("\n4. Evaluating on training data...")
    eval_results = evaluate_all_models(models, X, y, groups)
    
    # Optimize ensemble weights
    print("\n5. Optimizing ensemble weights...")
    weights = optimize_ensemble_weights(eval_results, y, groups)
    
    # Save all models
    print("\n6. Saving models...")
    os.makedirs('models', exist_ok=True)
    
    for name, model in models.items():
        model.save(f'models/{name}_model.pkl')
    
    # Save ensemble weights and metadata
    metadata = {
        'weights': weights,
        'feature_names': feature_names,
        'model_names': list(models.keys()),
        'train_score': eval_results,
        'n_features': X.shape[1],
        'n_train_samples': X.shape[0]
    }
    joblib.dump(metadata, 'models/ensemble_metadata.pkl')
    
    # Save feature scaler info (already in each model)
    print("   Models saved to models/")
    
    # Generate test predictions
    print("\n7. Generating test predictions...")
    test_scores_dict = {}
    for name, model in models.items():
        test_scores_dict[name] = model.predict(X_test)
    
    # Ensemble predictions
    score_matrix = np.column_stack([test_scores_dict[name] for name in models.keys()])
    ensemble_scores = score_matrix @ weights
    
    # Rank cars for test case
    ranked_indices = np.argsort(-ensemble_scores)
    ranked_cars = [test_car_ids[i] for i in ranked_indices]
    
    print(f"   Test ensemble scores: {ensemble_scores}")
    print(f"   Ranked cars: {' | '.join(ranked_cars)}")
    
    # Save predictions
    pred_df = pd.DataFrame({
        'file_id': ['acv_test_case.xlsx'],
        'ranked_cars': [' | '.join(ranked_cars)]
    })
    pred_df.to_csv('acv_predictions.csv', index=False)
    print("   Predictions saved to acv_predictions.csv")
    
    # Also save detailed per-model rankings for analysis
    for name in models.keys():
        model_scores = test_scores_dict[name]
        model_ranked = np.argsort(-model_scores)
        model_cars = [test_car_ids[i] for i in model_ranked]
        print(f"   {name} ranking: {' | '.join(model_cars)}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    return models, weights, feature_names, ranked_cars


if __name__ == "__main__":
    train_final_model()