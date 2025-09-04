from train import train_model

def test_training_runs():
    model, acc, scaler = train_model(epochs=1)
    assert acc > 0.5  # Sanity check: better than random
