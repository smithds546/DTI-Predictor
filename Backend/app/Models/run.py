import numpy as np
import pandas as pd
from basic import NeuralNetwork

def main():
    # --- Load prepped data ---
    X_drug_train = np.load("/Users/drs/Projects/DTI/Backend/app/data/prepped/drugs/drug_train.npy")
    X_prot_train = np.load("/Users/drs/Projects/DTI/Backend/app/data/prepped/proteins/prot_train.npy")
    y_train = pd.read_csv("/Users/drs/Projects/DTI/Backend/app/data/prepped/bindingdb/bindingdb_train.csv")["interaction"].values.reshape(-1, 1)

    X_drug_val = np.load("/Users/drs/Projects/DTI/Backend/app/data/prepped/drugs/drug_val.npy")
    X_prot_val = np.load("/Users/drs/Projects/DTI/Backend/app/data/prepped/proteins/prot_val.npy")
    y_val = pd.read_csv("/Users/drs/Projects/DTI/Backend/app/data/prepped/bindingdb/bindingdb_validation.csv")["interaction"].values.reshape(-1, 1)

    X_drug_test = np.load("//Users/drs/Projects/DTI/Backend/app/data/prepped/drugs/drug_test.npy")
    X_prot_test = np.load("/Users/drs/Projects/DTI/Backend/app/data/prepped/proteins/prot_test.npy")
    y_test = pd.read_csv("/Users/drs/Projects/DTI/Backend/app/data/prepped/bindingdb/bindingdb_test.csv")["interaction"].values.reshape(-1, 1)

    # --- Initialize network ---
    INPUT_SIZE_DRUG = X_drug_train.shape[1]
    INPUT_SIZE_PROTEIN = X_prot_train.shape[1]
    HIDDEN_SIZE = 32
    OUTPUT_SIZE = 1
    LEARNING_RATE = 0.5
    EPOCHS = 20000

    nn = NeuralNetwork(INPUT_SIZE_DRUG, INPUT_SIZE_PROTEIN, HIDDEN_SIZE, OUTPUT_SIZE, LEARNING_RATE)

    # --- Training loop with validation monitoring ---
    for epoch in range(EPOCHS):
        nn.forward(X_drug_train, X_prot_train)
        dW1, db1, dW2, db2 = nn.backward(y_train)
        nn.update_weights(dW1, db1, dW2, db2)

        if (epoch + 1) % 1000 == 0:
            train_loss = np.mean((nn.y_hat - y_train)**2)
            y_val_pred = nn.predict(X_drug_val, X_prot_val)
            val_loss = np.mean((y_val_pred - y_val)**2)
            print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    # --- Evaluate on test set ---
    y_test_pred = nn.predict(X_drug_test, X_prot_test)
    y_test_class = (y_test_pred > 0.5).astype(int)
    accuracy = np.mean(y_test_class == y_test)
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()