import pandas as pd
import numpy as np
from datetime import datetime
from sample_data.groundwater_data import generate_groundwater_data

def load_data(state="All States", district="All Districts", start_date=None, end_date=None):
    """
    Load groundwater data from the data source.
    In a real-world scenario, this would connect to CGWB APIs or databases.
    For this application, we use generated sample data.
    
    Parameters:
    -----------
    state : str
        The state to filter data for
    district : str
        The district to filter data for
    start_date : datetime
        The start date for the data
    end_date : datetime
        The end date for the data
        
    Returns:
    --------
    pd.DataFrame
        Dataframe containing the groundwater data
    """
    # In a real application, this would fetch data from CGWB or other sources
    # For this demo, we generate simulated data that mimics real groundwater patterns
    data = generate_groundwater_data(state, district, start_date, end_date)
    
    return data

def preprocess_data(data):
    """
    Preprocess the groundwater data to prepare it for analysis and modeling.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Raw groundwater data
        
    Returns:
    --------
    dict
        Dictionary containing processed data and metadata
    """
    # Check for missing values
    missing_values = data.isnull().sum().sum()
    has_missing_values = missing_values > 0
    
    # Calculate key metrics
    avg_level = data['groundwater_level'].mean()
    well_count = len(data['well_id'].unique())
    
    # Calculate level changes
    first_month_avg = data[data['date'] <= data['date'].min() + pd.DateOffset(months=1)]['groundwater_level'].mean()
    last_month_avg = data[data['date'] >= data['date'].max() - pd.DateOffset(months=1)]['groundwater_level'].mean()
    level_change = last_month_avg - first_month_avg
    
    # Calculate declining trend areas
    declining_wells = 0
    total_wells = 0
    
    for well_id in data['well_id'].unique():
        well_data = data[data['well_id'] == well_id].sort_values('date')
        if len(well_data) >= 2:
            total_wells += 1
            if well_data.iloc[-1]['groundwater_level'] > well_data.iloc[0]['groundwater_level']:
                declining_wells += 1
    
    declining_percentage = (declining_wells / total_wells * 100) if total_wells > 0 else 0
    
    # Calculate previous declining percentage (for trend)
    half_point = data['date'].min() + (data['date'].max() - data['date'].min()) / 2
    older_data = data[data['date'] <= half_point]
    
    older_declining_wells = 0
    older_total_wells = 0
    
    for well_id in older_data['well_id'].unique():
        well_data = older_data[older_data['well_id'] == well_id].sort_values('date')
        if len(well_data) >= 2:
            older_total_wells += 1
            if well_data.iloc[-1]['groundwater_level'] > well_data.iloc[0]['groundwater_level']:
                older_declining_wells += 1
    
    older_declining_percentage = (older_declining_wells / older_total_wells * 100) if older_total_wells > 0 else 0
    declining_change = declining_percentage - older_declining_percentage
    
    # Calculate rainfall metrics
    avg_rainfall = data['rainfall'].mean()
    first_year_rainfall = data[data['date'].dt.year == data['date'].dt.year.min()]['rainfall'].mean()
    last_year_rainfall = data[data['date'].dt.year == data['date'].dt.year.max()]['rainfall'].mean()
    rainfall_change = last_year_rainfall - first_year_rainfall
    
    # Add more derived features for analysis
    processed_data = data.copy()
    
    # Create seasonal features
    processed_data['month'] = processed_data['date'].dt.month
    processed_data['year'] = processed_data['date'].dt.year
    processed_data['season'] = pd.cut(
        processed_data['month'],
        bins=[0, 3, 6, 9, 12],
        labels=['Winter', 'Spring', 'Summer', 'Autumn'],
        include_lowest=True
    )
    
    # Calculate rolling averages for smoothing
    for well_id in processed_data['well_id'].unique():
        mask = processed_data['well_id'] == well_id
        processed_data.loc[mask, 'level_3m_avg'] = processed_data.loc[mask, 'groundwater_level'].rolling(window=3, min_periods=1).mean()
        processed_data.loc[mask, 'level_6m_avg'] = processed_data.loc[mask, 'groundwater_level'].rolling(window=6, min_periods=1).mean()
        processed_data.loc[mask, 'level_12m_avg'] = processed_data.loc[mask, 'groundwater_level'].rolling(window=12, min_periods=1).mean()
    
    # Calculate rate of change
    processed_data['level_change_rate'] = processed_data.groupby('well_id')['groundwater_level'].diff()
    
    # Return processed data and metadata
    return {
        'data': processed_data,
        'has_missing_values': has_missing_values,
        'missing_value_count': missing_values,
        'avg_level': avg_level,
        'well_count': well_count,
        'level_change': level_change,
        'declining_percentage': declining_percentage,
        'declining_change': declining_change,
        'avg_rainfall': avg_rainfall,
        'rainfall_change': rainfall_change
    }
