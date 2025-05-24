# Tourism Destination Recommender

A Streamlit web application that recommends tourism destinations using Neural Collaborative Filtering (NCF) with user authentication, PostgreSQL database integration, and destination names.

## Features

- User authentication (registration, login, profile management)
- PostgreSQL database for persistent storage
- User profile input (age, gender, budget, and group composition)
- Rating system for known destinations
- Personalized recommendations based on user ratings and profile
- Clean, text-based interface with styled cards
- Interactive UI

## Installation

1. Clone this repository:
   ```bash
   git clone <your-repository-url>
   cd <repository-folder>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up PostgreSQL database:
   - Install PostgreSQL if you don't have it already
   - Create a database user with appropriate permissions
   - Set up environment variables by creating a `.env` file:
     ```
     DATABASE_URL=postgresql://username:password@localhost:5432/tourism_recommender
     ```
   - Run the database setup script:
     ```bash
     python setup_db.py
     ```

4. Set up Unsplash API (optional but recommended for real destination images):
   - Register for a free Unsplash developer account at https://unsplash.com/developers
   - Create a new application to get your API keys
   - Add your API keys to the `.env` file:
     ```
     UNSPLASH_ACCESS_KEY=your_access_key_here
     UNSPLASH_SECRET_KEY=your_secret_key_here
     ```
   - Without API keys, the app will use placeholder images

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

2. Open your browser and navigate to the displayed URL (typically http://localhost:8501)

3. Register a new account or log in with existing credentials

4. Fill out your user profile in the sidebar

5. Rate tourism destinations in the "Rate Tourism Destinations" section

6. Click "Generate Recommendations" to get personalized recommendations

## How It Works

The app uses a Neural Collaborative Filtering (NCF) model built with TensorFlow. The model:

1. Combines user embeddings, place embeddings, and user features
2. Processes these through dense neural network layers
3. Outputs a predicted rating for each user-place pair
4. Recommends the highest-rated unvisited places

## Database Schema

The application uses the following database schema:

- **Users**: User information including profile data
- **Places**: Tourism destination information
- **Ratings**: User ratings for destinations (many-to-many relationship)

## Data

The app uses two datasets:
- User-place ratings: Contains user IDs, place IDs, and ratings
- User features: Contains demographic information about users

For demo purposes, the app uses sample data from public repositories.

## Model Details

The NCF model architecture:
- Separate embedding layers for users and places
- Integration of user features (age, gender, budget, group type)
- Two dense layers (128 and 64 neurons) with ReLU activation
- Dropout layers (0.5) for regularization
- Linear output layer for rating prediction

## Key Algorithm Components

### Data Loading and Preprocessing
- Loading user-place interactions and user feature data
- Cleaning and normalizing data (filling missing values, encoding categorical features)
- **Rating normalization**: Converting ratings to 0-1 range for training
- Gender mapping to numeric values ('Male': 1, 'Female': 0)
- One-hot encoding of group composition features
- Label encoding for user and place IDs

### Feature Weighting
- Age: 30% weight
- Gender: 30% weight
- Budget: 20% weight
- Group Composition: 20% weight (distributed evenly among group types)

### Model Architecture
- Embedding layers for users and places with L2 regularization
- Concatenation of embeddings with weighted user features
- Dense neural network layers with ReLU activation
- Dropout for preventing overfitting
- Linear output layer for rating prediction

### Web App Optimizations
- BatchNormalization for better training stability
- Lower dropout rate (0.3 vs 0.5)
- Smaller network architecture (64→32 vs 128→64)
- Higher L2 regularization in embeddings (1e-5 vs 1e-6)
- Higher learning rate (0.002 vs 0.001)
- Smaller embedding size (32 vs 50)
- Early stopping during training
- Caching for improved performance
- Batch prediction for recommendations
- **Fixed rating conversion**: Proper scaling from normalized (0-1) to display (1-5) range
- **Fixed feature scaling**: User features now properly normalized to match training data range
- **Improved user ID mapping**: Hash-based user ID assignment for personalized embeddings

## Deployment

To deploy in production:
1. Set up a secure PostgreSQL database
2. Configure environment variables for production
3. Use a more secure authentication method (e.g., JWT tokens)
4. Deploy using Streamlit Sharing, Heroku, or any other cloud provider 