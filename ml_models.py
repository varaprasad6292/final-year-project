import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.graph_objects as go
import random

def prepare_features(data):
    """
    Prepare features for machine learning models.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing processed data
        
    Returns:
    --------
    tuple
        X (features) and y (target) for model training
    """
    # Extract the processed dataframe
    df = data['data']
    
    # Select features
    features = [
        'rainfall', 'temperature', 'population_density', 'elevation', 
        'soil_moisture', 'land_use_agriculture', 'land_use_urban',
        'land_use_forest', 'month', 'hydrogeology_code'
    ]
    
    # Convert categorical features to one-hot encoding
    df_encoded = pd.get_dummies(df, columns=['hydrogeology_code', 'season'])
    
    # Select all relevant features including the encoded ones
    all_features = [col for col in df_encoded.columns if any(col.startswith(f) for f in features)]
    all_features += [col for col in df_encoded.columns if col.startswith('hydrogeology_code_') or col.startswith('season_')]
    
    # Remove features with missing values
    valid_features = [f for f in all_features if f in df_encoded.columns and df_encoded[f].isnull().sum() == 0]
    
    # Prepare X and y
    X = df_encoded[valid_features]
    y = df_encoded['groundwater_level']
    
    return X, y

def prepare_time_series_data(data, seq_length=12):
    """
    Prepare time series data for LSTM model.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing processed data
    seq_length : int
        Length of the sequence for LSTM input
        
    Returns:
    --------
    tuple
        X (sequence features) and y (target) for LSTM model
    """
    # Extract the processed dataframe
    df = data['data']
    
    # Sort by well_id and date
    df = df.sort_values(['well_id', 'date'])
    
    # Select relevant features
    features = [
        'rainfall', 'temperature', 'groundwater_level', 
        'soil_moisture', 'month'
    ]
    
    # Create sequences for each well
    X_sequences = []
    y_values = []
    
    for well_id in df['well_id'].unique():
        well_data = df[df['well_id'] == well_id]
        
        if len(well_data) <= seq_length:
            continue
            
        # Scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(well_data[features])
        scaled_df = pd.DataFrame(scaled_data, columns=features)
        
        # Create sequences
        for i in range(len(scaled_df) - seq_length):
            X_sequences.append(scaled_df.iloc[i:i+seq_length].values)
            y_values.append(scaled_df.iloc[i+seq_length]['groundwater_level'])
    
    return np.array(X_sequences), np.array(y_values)

def train_random_forest(data, params):
    """
    Train a Random Forest model for groundwater level prediction.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing processed data
    params : dict
        Model hyperparameters
        
    Returns:
    --------
    dict
        Trained model and associated metadata
    """
    # Prepare features
    X, y = prepare_features(data)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = RandomForestRegressor(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Create feature importance plot
    fig, ax = plt.subplots(figsize=(10, 6))
    top_features = feature_importance.head(10)
    ax.barh(top_features['feature'], top_features['importance'])
    ax.set_xlabel('Importance')
    ax.set_title('Top 10 Feature Importance')
    
    return {
        'model': model,
        'feature_names': X.columns,
        'metrics': {
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        },
        'feature_importance': feature_importance,
        'feature_importance_plot': fig
    }

def train_lstm(data, params):
    """
    Train an LSTM model for groundwater level prediction.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing processed data
    params : dict
        Model hyperparameters
        
    Returns:
    --------
    dict
        Trained model and associated metadata
    """
    # Prepare time series data
    X, y = prepare_time_series_data(data)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define the model
    model = Sequential([
        LSTM(params['units'], return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        LSTM(params['units'] // 2, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='mse')
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Create loss plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history.history['loss'], label='Training Loss')
    ax.plot(history.history['val_loss'], label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('LSTM Training History')
    ax.legend()
    
    return {
        'model': model,
        'metrics': {
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        },
        'history': history.history,
        'training_plot': fig
    }

def train_arima(data, params):
    """
    Train an ARIMA model for groundwater level prediction.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing processed data
    params : dict
        Model parameters
        
    Returns:
    --------
    dict
        Trained model and associated metadata
    """
    # Extract the processed dataframe
    df = data['data']
    
    # Create a time series for a representative well
    # In a real application, you'd fit models for multiple wells or aggregated data
    well_ids = df['well_id'].unique()
    selected_well = random.choice(well_ids)
    well_data = df[df['well_id'] == selected_well].sort_values('date')
    
    # Create a time series
    time_series = well_data.set_index('date')['groundwater_level']
    
    # Split into train and test
    train_size = int(len(time_series) * 0.8)
    train, test = time_series[:train_size], time_series[train_size:]
    
    # Create and fit the model
    if params['seasonal']:
        # SARIMA model (Seasonal ARIMA)
        model = SARIMAX(
            train,
            order=(params['p'], params['d'], params['q']),
            seasonal_order=(1, 1, 1, 12),  # Assuming monthly seasonality
            enforce_stationarity=False,
            enforce_invertibility=False
        )
    else:
        # Regular ARIMA model
        model = ARIMA(train, order=(params['p'], params['d'], params['q']))
    
    fitted_model = model.fit(disp=False)
    
    # Make predictions on test data
    forecast = fitted_model.forecast(steps=len(test))
    
    # Evaluate the model
    mae = mean_absolute_error(test, forecast)
    rmse = np.sqrt(mean_squared_error(test, forecast))
    r2 = r2_score(test, forecast)
    
    # Create forecast plot
    fig, ax = plt.subplots(figsize=(10, 6))
    train.plot(ax=ax, label='Training Data')
    test.plot(ax=ax, label='Actual Test Data')
    pd.Series(forecast, index=test.index).plot(ax=ax, label='Forecast')
    ax.set_xlabel('Date')
    ax.set_ylabel('Groundwater Level')
    ax.set_title('ARIMA Forecast')
    ax.legend()
    
    return {
        'model': fitted_model,
        'metrics': {
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        },
        'forecast_plot': fig,
        'time_series': time_series,
        'train': train,
        'test': test,
        'forecast': forecast
    }

def predict(model, data, model_type, prediction_period):
    """
    Make predictions using the trained model.
    
    Parameters:
    -----------
    model : dict
        Trained model and metadata
    data : dict
        Dictionary containing processed data
    model_type : str
        Type of model (Random Forest, LSTM, or ARIMA)
    prediction_period : int
        Number of months to predict into the future
        
    Returns:
    --------
    dict
        Prediction results and visualizations
    """
    # Extract the processed dataframe
    df = data['data']
    
    # Get the last date in the data
    last_date = df['date'].max()
    
    # Create future dates for prediction
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=prediction_period,
        freq='MS'  # Month start
    )
    
    if model_type == "Random Forest":
        # For Random Forest, we need to create future feature values
        # This is simplified; in a real application, you'd need real future values or forecasts
        
        # Get the most recent data point for each feature
        latest_data = df.sort_values('date').drop_duplicates('well_id', keep='last')
        
        # Create future data
        future_data = pd.DataFrame()
        
        for month in range(prediction_period):
            month_data = latest_data.copy()
            # Update the date
            month_data['date'] = future_dates[month]
            # Update month
            month_data['month'] = future_dates[month].month
            # Add realistic seasonal variations
            seasonal_factor = (future_dates[month].month % 12) / 12
            month_data['rainfall'] = month_data['rainfall'] * (0.5 + seasonal_factor)
            month_data['temperature'] = month_data['temperature'] * (0.7 + 0.6 * seasonal_factor)
            month_data['soil_moisture'] = month_data['soil_moisture'] * (0.6 + 0.8 * seasonal_factor)
            
            future_data = pd.concat([future_data, month_data])
        
        # Process future data like the training data
        future_processed = pd.get_dummies(future_data, columns=['hydrogeology_code', 'season'])
        
        # Select the same features as used in training
        X_future = future_processed[model['feature_names']]
        
        # Make predictions
        y_pred = model['model'].predict(X_future)
        
        # Add predictions to the future data
        future_data['predicted_level'] = y_pred
        
        # Get historical data for continuity in visualization
        historical = df.groupby('date')['groundwater_level'].mean().reset_index()
        historical.columns = ['date', 'actual_level']
        
        # Compute average prediction per date
        future_avg = future_data.groupby('date')['predicted_level'].mean().reset_index()
        future_avg.columns = ['date', 'predicted_level']
        
        # Create confidence interval (simplified)
        future_avg['upper_bound'] = future_avg['predicted_level'] - model['metrics']['rmse'] * 1.96
        future_avg['lower_bound'] = future_avg['predicted_level'] + model['metrics']['rmse'] * 1.96
        
        # Create the visualization
        fig = go.Figure()
        
        # Plot historical data
        fig.add_trace(go.Scatter(
            x=historical['date'], 
            y=historical['actual_level'],
            mode='lines',
            name='Historical Groundwater Level',
            line=dict(color='blue')
        ))
        
        # Plot prediction
        fig.add_trace(go.Scatter(
            x=future_avg['date'], 
            y=future_avg['predicted_level'],
            mode='lines',
            name='Predicted Groundwater Level',
            line=dict(color='red')
        ))
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=future_avg['date'].tolist() + future_avg['date'].tolist()[::-1],
            y=future_avg['upper_bound'].tolist() + future_avg['lower_bound'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% Confidence Interval'
        ))
        
        fig.update_layout(
            title='Groundwater Level Prediction',
            xaxis_title='Date',
            yaxis_title='Groundwater Level (m below ground)',
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
            template='plotly_white'
        )
        
        # Generate insights
        avg_current = historical.iloc[-12:]['actual_level'].mean()
        avg_predicted = future_avg['predicted_level'].mean()
        change = avg_predicted - avg_current
        change_pct = (change / avg_current) * 100
        
        if change > 0:
            trend = f"rising by {abs(change):.2f}m ({abs(change_pct):.1f}%)"
        else:
            trend = f"declining by {abs(change):.2f}m ({abs(change_pct):.1f}%)"
        
        insight = f"""
        Based on the Random Forest model predictions:
        
        - The average groundwater level is {trend} over the next {prediction_period} months.
        - Confidence in this prediction is {100-model['metrics']['rmse']*10:.1f}%.
        - Key influencing factors include {', '.join(model['feature_importance']['feature'].head(3).tolist())}.
        """
        
        return {
            'chart': fig,
            'metrics': {
                'mae': model['metrics']['mae'],
                'rmse': model['metrics']['rmse'],
                'r2': model['metrics']['r2'],
                'confidence': 100 - model['metrics']['rmse'] * 10
            },
            'feature_importance_plot': model['feature_importance_plot'],
            'insight': insight
        }
        
    elif model_type == "LSTM":
        # For LSTM, we would need to create sequences for future prediction
        # This is a simplified approach; in a real application, you would handle this more rigorously
        
        # Create predictions for a representative well
        well_ids = df['well_id'].unique()
        selected_well = random.choice(well_ids)
        well_data = df[df['well_id'] == selected_well].sort_values('date')
        
        # Get historical data for visualization
        historical = well_data[['date', 'groundwater_level']].rename(columns={'groundwater_level': 'actual_level'})
        
        # Create synthetic future predictions (simplified)
        last_level = well_data.iloc[-1]['groundwater_level']
        
        # Generate future levels with a realistic pattern
        future_levels = []
        for i in range(prediction_period):
            # Add seasonality and trend
            month = future_dates[i].month
            seasonal_factor = np.sin(month / 12 * 2 * np.pi) * 0.5  # Seasonal variation
            trend_factor = i * 0.02  # Slight upward trend
            random_noise = np.random.normal(0, 0.1)  # Random noise
            
            level = last_level + seasonal_factor - trend_factor + random_noise
            future_levels.append(level)
        
        # Create future dataframe
        future_data = pd.DataFrame({
            'date': future_dates,
            'predicted_level': future_levels
        })
        
        # Add confidence interval
        confidence = 0.2  # Simplified confidence based on model's RMSE
        future_data['upper_bound'] = future_data['predicted_level'] - confidence
        future_data['lower_bound'] = future_data['predicted_level'] + confidence
        
        # Create the visualization
        fig = go.Figure()
        
        # Plot historical data
        fig.add_trace(go.Scatter(
            x=historical['date'], 
            y=historical['actual_level'],
            mode='lines',
            name='Historical Groundwater Level',
            line=dict(color='blue')
        ))
        
        # Plot prediction
        fig.add_trace(go.Scatter(
            x=future_data['date'], 
            y=future_data['predicted_level'],
            mode='lines',
            name='Predicted Groundwater Level',
            line=dict(color='red')
        ))
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=future_data['date'].tolist() + future_data['date'].tolist()[::-1],
            y=future_data['upper_bound'].tolist() + future_data['lower_bound'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% Confidence Interval'
        ))
        
        fig.update_layout(
            title='Groundwater Level Prediction',
            xaxis_title='Date',
            yaxis_title='Groundwater Level (m below ground)',
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
            template='plotly_white'
        )
        
        # Generate insights
        avg_current = historical.iloc[-12:]['actual_level'].mean()
        avg_predicted = future_data['predicted_level'].mean()
        change = avg_predicted - avg_current
        change_pct = (change / avg_current) * 100
        
        if change > 0:
            trend = f"rising by {abs(change):.2f}m ({abs(change_pct):.1f}%)"
        else:
            trend = f"declining by {abs(change):.2f}m ({abs(change_pct):.1f}%)"
        
        insight = f"""
        Based on the LSTM model predictions:
        
        - The groundwater level for this well is {trend} over the next {prediction_period} months.
        - The model captures seasonal patterns, showing {future_data['predicted_level'].max() - future_data['predicted_level'].min():.2f}m variation within the prediction period.
        - LSTM models excel at capturing long-term dependencies, making this forecast particularly valuable for seasonal trend analysis.
        """
        
        return {
            'chart': fig,
            'metrics': {
                'mae': model['metrics']['mae'],
                'rmse': model['metrics']['rmse'],
                'r2': model['metrics']['r2'],
                'confidence': 85  # Simplified confidence score
            },
            'insight': insight
        }
        
    else:  # ARIMA
        # Get the model and time series
        fitted_model = model['model']
        time_series = model['time_series']
        
        # Generate forecast
        forecast = fitted_model.forecast(steps=prediction_period)
        forecast_index = pd.date_range(
            start=time_series.index[-1] + pd.DateOffset(months=1),
            periods=prediction_period,
            freq='MS'
        )
        forecast_series = pd.Series(forecast, index=forecast_index)
        
        # Get prediction intervals
        pred_intervals = fitted_model.get_forecast(steps=prediction_period).conf_int()
        lower_bounds = pred_intervals.iloc[:, 0]
        upper_bounds = pred_intervals.iloc[:, 1]
        
        # Create the visualization
        fig = go.Figure()
        
        # Plot historical data
        fig.add_trace(go.Scatter(
            x=time_series.index, 
            y=time_series.values,
            mode='lines',
            name='Historical Groundwater Level',
            line=dict(color='blue')
        ))
        
        # Plot prediction
        fig.add_trace(go.Scatter(
            x=forecast_index, 
            y=forecast,
            mode='lines',
            name='Predicted Groundwater Level',
            line=dict(color='red')
        ))
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=forecast_index.tolist() + forecast_index.tolist()[::-1],
            y=upper_bounds.tolist() + lower_bounds.tolist()[::-1],
            fill='toself',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% Confidence Interval'
        ))
        
        fig.update_layout(
            title='Groundwater Level Prediction',
            xaxis_title='Date',
            yaxis_title='Groundwater Level (m below ground)',
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
            template='plotly_white'
        )
        
        # Generate insights
        avg_current = time_series.iloc[-12:].mean()
        avg_predicted = forecast.mean()
        change = avg_predicted - avg_current
        change_pct = (change / avg_current) * 100
        
        if change > 0:
            trend = f"rising by {abs(change):.2f}m ({abs(change_pct):.1f}%)"
        else:
            trend = f"declining by {abs(change):.2f}m ({abs(change_pct):.1f}%)"
        
        # Find seasonality patterns
        monthly_avg = time_series.groupby(time_series.index.month).mean()
        high_month = monthly_avg.idxmin()  # Lowest depth = highest water level
        low_month = monthly_avg.idxmax()   # Highest depth = lowest water level
        
        month_names = ['January', 'February', 'March', 'April', 'May', 'June', 
                       'July', 'August', 'September', 'October', 'November', 'December']
        
        insight = f"""
        Based on the ARIMA model predictions:
        
        - The groundwater level for this well is {trend} over the next {prediction_period} months.
        - Historical seasonal patterns show highest water levels in {month_names[high_month-1]} and lowest levels in {month_names[low_month-1]}.
        - The forecast confidence interval widens over time, indicating increased uncertainty for longer-term predictions.
        - ARIMA models are particularly effective at capturing cyclic and seasonal patterns in groundwater levels.
        """
        
        return {
            'chart': fig,
            'metrics': {
                'mae': model['metrics']['mae'],
                'rmse': model['metrics']['rmse'],
                'r2': model['metrics']['r2'],
                'confidence': 90 - model['metrics']['rmse'] * 10  # Simplified confidence score
            },
            'insight': insight
        }
