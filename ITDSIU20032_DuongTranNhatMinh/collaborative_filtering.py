import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split

def preprocess_data(data, min_user_ratings=5, min_place_ratings=5):
    """Preprocess the dataset by filtering sparse data and normalizing ratings."""
    data.drop_duplicates(subset=['user_id', 'place_id'], keep='last', inplace=True)
    data.dropna(subset=['user_id', 'place_id', 'rating'], inplace=True)

    filtered_users = data['user_id'].value_counts()
    filtered_places = data['place_id'].value_counts()
    data = data[data['user_id'].isin(filtered_users[filtered_users >= min_user_ratings].index)]
    data = data[data['place_id'].isin(filtered_places[filtered_places >= min_place_ratings].index)]

    data['rating_normalized'] = (data['rating'] - data['rating'].min()) / (data['rating'].max() - data['rating'].min())
    return data

def create_user_item_matrix(data):
    """Create a user-item matrix."""
    return data.pivot_table(index='user_id', columns='place_id', values='rating_normalized', fill_value=0)

def calculate_similarity(matrix, method='cosine'):
    """Calculate similarity matrix."""
    if method == 'cosine':
        return pd.DataFrame(cosine_similarity(matrix), index=matrix.index, columns=matrix.index)
    else:
        raise ValueError("Unsupported similarity method")

def matrix_factorization(train_matrix, k=10):
    """Apply Singular Value Decomposition (SVD)."""
    u, sigma, vt = svds(train_matrix, k=k)
    sigma = np.diag(sigma)
    return np.dot(np.dot(u, sigma), vt)

def predict_ratings(user_id, place_id, train_matrix, similarity_matrix, k=10):
    """Predict ratings using a weighted similarity approach."""
    if place_id not in train_matrix.columns:
        return train_matrix.loc[user_id].mean()

    user_ratings = train_matrix.loc[user_id]
    similar_items = similarity_matrix[place_id] if place_id in similarity_matrix else pd.Series(dtype=float)

    rated_items = user_ratings[user_ratings > 0].index
    similarities = similar_items.loc[rated_items]
    top_k_items = similarities.nlargest(k)

    numerator = sum(user_ratings[item] * similarities[item] for item in top_k_items.index)
    denominator = sum(abs(similarities[item]) for item in top_k_items.index)

    return numerator / denominator if denominator != 0 else user_ratings.mean()

def evaluate_model(train_matrix, test_data, similarity_matrix, k=10, threshold=0.5):
    """Evaluate the model with RMSE, MAE, accuracy, precision, recall, and F1 score."""
    predictions, actuals = [], []
    for _, row in test_data.iterrows():
        user_id, place_id, actual_rating = row['user_id'], row['place_id'], row['rating_normalized']
        if user_id in train_matrix.index:
            predicted_rating = predict_ratings(user_id, place_id, train_matrix, similarity_matrix, k=k)
        else:
            predicted_rating = train_matrix.mean().mean()
        predictions.append(predicted_rating)
        actuals.append(actual_rating)

    # RMSE and MAE
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)

    # Accuracy (considering difference <= 0.5 as accurate)
    accuracy = np.mean(np.abs(np.array(actuals) - np.array(predictions)) <= 0.5) * 100

    # Precision, Recall, and F1-Score
    # Convert ratings to binary (above threshold = 1, below threshold = 0)
    predicted_binary = [1 if p > threshold else 0 for p in predictions]
    actual_binary = [1 if a > threshold else 0 for a in actuals]

    precision = precision_score(actual_binary, predicted_binary)
    recall = recall_score(actual_binary, predicted_binary)
    f1 = f1_score(actual_binary, predicted_binary)

    return rmse, mae, accuracy, precision, recall, f1

# Load the dataset
data = pd.read_csv('https://raw.githubusercontent.com/NhatMinh2910/Pre-thesis-Datasets/refs/heads/main/rats.csv')
data = preprocess_data(data)

# Split data into training and testing sets
train, test = train_test_split(data[['user_id', 'place_id', 'rating_normalized']], test_size=0.3, random_state=42)
train_matrix = create_user_item_matrix(train)

# Train the model using matrix factorization
train_matrix_filled = matrix_factorization(train_matrix.fillna(0).values, k=10)
train_matrix_filled = pd.DataFrame(train_matrix_filled, index=train_matrix.index, columns=train_matrix.columns)

# Calculate similarity matrix
similarity_matrix = calculate_similarity(train_matrix.T, method='cosine')

# Evaluate the model
rmse, mae, accuracy, precision, recall, f1 = evaluate_model(train_matrix, test, similarity_matrix, k=10)

print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
