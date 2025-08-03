import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

# === Part 1: Data Loading and Preprocessing ===
def load_and_preprocess_data():
    user_place_data = pd.read_csv('https://raw.githubusercontent.com/NhatMinh2910/Pre-thesis-Datasets/refs/heads/main/rats.csv')
    user_features = pd.read_csv('https://raw.githubusercontent.com/NhatMinh2910/Pre-thesis-Datasets/refs/heads/main/ufeat.csv')

    # Keep only needed columns
    user_features = user_features[['user_id', 'Age', 'Gender', 'Budget','GroupComp']]

    # Fill missing values
    user_features.fillna(0, inplace=True)

    # Map Gender to numeric
    user_features['Gender'] = user_features['Gender'].map({'Male': 1, 'Female': 0})

    # One-hot encode GroupComp
    user_features = pd.get_dummies(user_features, columns=['GroupComp'])

    # Encode user_id and place_id
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    user_place_data['user_id'] = user_encoder.fit_transform(user_place_data['user_id'])
    user_place_data['place_id'] = item_encoder.fit_transform(user_place_data['place_id'])

    # Normalize rating
    if user_place_data['rating'].max() > 1:
        user_place_data['rating'] = user_place_data['rating'] / user_place_data['rating'].max()

    # Merge user features
    user_place_data = user_place_data.merge(user_features, on='user_id', how='left')

    return user_place_data, user_encoder, item_encoder, user_features

# === Part 2: Prepare Inputs and Train-Test Split ===
def prepare_data(user_place_data, user_features):
    X_user_place = user_place_data[['user_id', 'place_id']].values
    y = user_place_data['rating'].values

    # Extract user features columns except 'user_id'
    feature_cols = [c for c in user_features.columns if c != 'user_id']
    X_user_features = user_place_data[feature_cols].values.astype(np.float32)

    # Apply weights: Age (0.3), Gender (0.3), Budget (0.2), each GroupComp column (0.2 total split evenly)
    # First find indices of each attribute in X_user_features
    age_idx = feature_cols.index('Age')
    gender_idx = feature_cols.index('Gender')
    budget_idx = feature_cols.index('Budget')

    # GroupComp columns (all other columns except Age, Gender, Budget)
    groupcomp_indices = [i for i, c in enumerate(feature_cols) if c.startswith('GroupComp_')]

    # Number of GroupComp columns
    n_groupcomp = len(groupcomp_indices)
    if n_groupcomp == 0:
        raise ValueError("No GroupComp columns found after one-hot encoding.")

    # Define weights per feature
    weights = np.ones(X_user_features.shape[1], dtype=np.float32) * 0.0
    weights[age_idx] = 0.2
    weights[gender_idx] = 0.3
    weights[budget_idx] = 0.3
    # Distribute GroupComp total weight 0.2 evenly
    for idx in groupcomp_indices:
        weights[idx] = 0.2 / n_groupcomp

    # Apply weights by multiplying feature columns
    X_user_features_weighted = X_user_features * weights

    # Train test split
    X_train, X_test, y_train, y_test, user_feat_train, user_feat_test = train_test_split(
        X_user_place, y, X_user_features_weighted, test_size=0.2, random_state=42)

    # Convert inputs to proper dtypes
    X_train_user = np.array(X_train[:, 0], dtype=np.int32)
    X_train_place = np.array(X_train[:, 1], dtype=np.int32)
    user_feat_train = np.array(user_feat_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)

    X_test_user = np.array(X_test[:, 0], dtype=np.int32)
    X_test_place = np.array(X_test[:, 1], dtype=np.int32)
    user_feat_test = np.array(user_feat_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)

    return (X_train_user, X_train_place, user_feat_train, y_train,
            X_test_user, X_test_place, user_feat_test, y_test)

# === Part 3: Build the NCF Model ===
def build_ncf_model(num_users, num_items, user_feat_dim, embedding_size=32):
    user_input = Input(shape=(1,), name='user_input')
    place_input = Input(shape=(1,), name='place_input')
    user_features_input = Input(shape=(user_feat_dim,), name='user_features_input')

    user_embedding = Embedding(num_users, embedding_size,
                               embeddings_regularizer=tf.keras.regularizers.l2(1e-5),
                               name='user_embedding')(user_input)
    place_embedding = Embedding(num_items, embedding_size,
                                embeddings_regularizer=tf.keras.regularizers.l2(1e-5),
                                name='place_embedding')(place_input)

    user_flat = Flatten()(user_embedding)
    place_flat = Flatten()(place_embedding)

    # Concatenate embeddings and weighted user features
    combined = Concatenate()([user_flat, place_flat, user_features_input])

    x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-6))(combined)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-6))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='linear')(x)

    model = Model(inputs=[user_input, place_input, user_features_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.002), loss='mse')
    return model

# === Part 4: Main Execution ===
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

def compute_accuracy(y_true, y_pred, threshold=0.5):
    y_pred_bin = (y_pred.flatten() >= threshold).astype(int)
    y_true_bin = (y_true >= threshold).astype(int)
    return accuracy_score(y_true_bin, y_pred_bin)

def compute_precision(y_true, y_pred, threshold=0.5):
    y_pred_bin = (y_pred.flatten() >= threshold).astype(int)
    y_true_bin = (y_true >= threshold).astype(int)
    return precision_score(y_true_bin, y_pred_bin)

def compute_recall(y_true, y_pred, threshold=0.5):
    y_pred_bin = (y_pred.flatten() >= threshold).astype(int)
    y_true_bin = (y_true >= threshold).astype(int)
    return recall_score(y_true_bin, y_pred_bin)

def compute_f1_score(y_true, y_pred, threshold=0.5):
    y_pred_bin = (y_pred.flatten() >= threshold).astype(int)
    y_true_bin = (y_true >= threshold).astype(int)
    return f1_score(y_true_bin, y_pred_bin)

def main():
    user_place_data, user_encoder, item_encoder, user_features = load_and_preprocess_data()

    (X_train_user, X_train_place, user_feat_train, y_train,
     X_test_user, X_test_place, user_feat_test, y_test) = prepare_data(user_place_data, user_features)

    model = build_ncf_model(
        num_users=len(user_encoder.classes_),
        num_items=len(item_encoder.classes_),
        user_feat_dim=user_feat_train.shape[1]
    )


    model.fit([X_train_user, X_train_place, user_feat_train], y_train,
              epochs=10, batch_size=128, validation_split=0.1)

    # Predict on test set
    y_pred = model.predict([X_test_user, X_test_place, user_feat_test])

    # Calculate RMSE
    rmse = np.sqrt(np.mean((y_test - y_pred.flatten()) ** 2))
    print(f"Test RMSE: {rmse:.4f}")

    # Calculate Accuracy
    accuracy = compute_accuracy(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Calculate Precision
    precision = compute_precision(y_test, y_pred)
    print(f"Test Precision: {precision:.4f}")

    # Calculate Recall
    recall = compute_recall(y_test, y_pred)
    print(f"Test Recall: {recall:.4f}")

    # Calculate F1-Score
    f1 = compute_f1_score(y_test, y_pred)
    print(f"Test F1-Score: {f1:.4f}")

    # Example prediction
    user_id_example = 1
    place_id_example = 2

    # Prepare example user features vector with weights applied (adjust indices as needed)
    user_features_example = np.zeros((1, user_feat_train.shape[1]), dtype=np.float32)
    age_idx = user_features.columns.get_loc('Age') - 1
    gender_idx = user_features.columns.get_loc('Gender') - 1
    budget_idx = user_features.columns.get_loc('Budget') - 1
    groupcomp_cols = [i-1 for i, c in enumerate(user_features.columns) if c.startswith('GroupComp_')]

    user_features_example[0, age_idx] = 30 * 0.2
    user_features_example[0, gender_idx] = 1 * 0.3
    user_features_example[0, budget_idx] = 500 * 0.3
    for i in groupcomp_cols:
        user_features_example[0, i] = 0  # or set one group to 0.2/n_groupcomp

    pred = model.predict([np.array([user_id_example]), np.array([place_id_example]), user_features_example])
    print(f"Predicted Rating: {pred[0][0]:.4f}")

if __name__ == "__main__":
    main()
