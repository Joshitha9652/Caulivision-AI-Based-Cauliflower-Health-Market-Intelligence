"""
train_model.py
End-to-end training pipeline for Cauliflower AI models:
1. Size Classification Model (Small, Medium, Large)
2. Quality Assessment Model (Grade A, Grade B, Grade C)
3. Mandi Price Prediction Model (Selling Price Regressor)
4. Calibrated Computer Vision Disease Classifier (Bacterial Spot Rot, Black Rot, Downy Mildew, Healthy Fruit)
"""

from model_utils import train_all_models

if __name__ == "__main__":
    print("=" * 70)
    print("STARTING CAULIFLOWER AI MODEL TRAINING SUITE")
    print("=" * 70)
    results = train_all_models()
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY & MODEL METRICS:")
    print("=" * 70)
    metrics = results["metrics"]
    print(f"[OK] Size Classifier Accuracy:    {metrics['size_acc'] * 100:.2f}%")
    print(f"[OK] Quality Classifier Accuracy: {metrics['quality_acc'] * 100:.2f}%")
    print(f"[OK] AP Mandi Price Regressor R2: {metrics['price_r2']:.4f}")
    print(f"[OK] Vision Disease Model Acc:    {metrics['disease_acc'] * 100:.2f}%")
    print("\nAll models and label encoders serialized and saved successfully in 'models/' and root.")
