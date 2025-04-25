import pandas as pd
import numpy as np
from scipy import interpolate
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import matplotlib.pyplot as plt

def interpolate_missing_data(data):
    """
    Interpolate missing values in groundwater data.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing processed data
        
    Returns:
    --------
    dict
        Dictionary with interpolated data
    """
    # Extract the processed dataframe
    df = data['data'].copy()
    
    # Check for missing values
    missing_columns = df.columns[df.isnull().any()].tolist()
    
    if not missing_columns:
        # No missing values to interpolate
        return data
    
    # Separate numerical and categorical columns
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Handle missing values in numerical columns
    for well_id in df['well_id'].unique():
        well_mask = df['well_id'] == well_id
        well_data = df.loc[well_mask]
        
        for col in numerical_cols:
            if col in missing_columns:
                # Check if there are missing values for this well
                if well_data[col].isnull().any():
                    # If we have enough non-null values, use interpolation
                    if well_data[col].notna().sum() >= 2:
                        # Sort by date for proper time-series interpolation
                        sorted_data = well_data.sort_values('date')
                        
                        # For groundwater level and related variables, use cubic interpolation if possible
                        if col in ['groundwater_level', 'level_3m_avg', 'level_6m_avg', 'level_12m_avg']:
                            if sorted_data[col].notna().sum() >= 4:  # Cubic requires at least 4 points
                                df.loc[well_mask, col] = sorted_data[col].interpolate(method='cubic')
                            else:
                                df.loc[well_mask, col] = sorted_data[col].interpolate(method='linear')
                        else:
                            # For other numerical variables, use linear interpolation
                            df.loc[well_mask, col] = sorted_data[col].interpolate(method='linear')
                    
                    # If we don't have enough data points, use forward/backward fill
                    else:
                        df.loc[well_mask, col] = df.loc[well_mask, col].fillna(method='ffill').fillna(method='bfill')
    
    # Check if there are still missing values in numerical columns
    still_missing_numerical = df[numerical_cols].isnull().any().any()
    
    if still_missing_numerical:
        # Use KNN imputation for remaining missing numerical values
        numerical_data = df[numerical_cols]
        
        # Use KNN to impute the remaining missing values
        imputer = KNNImputer(n_neighbors=5)
        
        # Imputation requires no NaN values in the reference columns
        # We'll use only columns with complete data for reference
        complete_cols = [col for col in numerical_cols if df[col].notna().all()]
        
        if complete_cols:
            for col in numerical_cols:
                if df[col].isnull().any():
                    # Create a reference dataframe with the column to impute and complete columns
                    ref_cols = complete_cols.copy()
                    if col in ref_cols:
                        ref_cols.remove(col)
                    
                    if ref_cols:  # Ensure we have reference columns
                        ref_df = df[ref_cols + [col]].copy()
                        # Apply KNN imputation
                        imputed_data = imputer.fit_transform(ref_df)
                        # Update only the column we're imputing
                        df[col] = imputed_data[:, -1]
    
    # Handle missing values in categorical columns
    for col in categorical_cols:
        if col in missing_columns:
            # Use most frequent value for each group
            for group in df.groupby('well_id'):
                well_id = group[0]
                group_data = group[1]
                
                if group_data[col].isnull().any():
                    most_frequent = group_data[col].mode()[0] if not group_data[col].mode().empty else None
                    
                    if most_frequent is not None:
                        df.loc[df['well_id'] == well_id, col] = df.loc[df['well_id'] == well_id, col].fillna(most_frequent)
            
            # Fill any remaining missing values with the overall most frequent value
            most_frequent_overall = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
            df[col] = df[col].fillna(most_frequent_overall)
    
    # Update the data dictionary
    data['data'] = df
    data['has_missing_values'] = df.isnull().any().any()
    data['missing_value_count'] = df.isnull().sum().sum()
    
    return data

def compare_interpolation_methods(data, column='groundwater_level'):
    """
    Compare different interpolation methods for a specific column.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing processed data
    column : str
        Column to interpolate
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure comparing interpolation methods
    """
    # Extract the processed dataframe
    df = data['data'].copy()
    
    # Select a well with missing values in the specified column
    wells_with_missing = []
    
    for well_id in df['well_id'].unique():
        well_data = df[df['well_id'] == well_id]
        if well_data[column].isnull().any():
            wells_with_missing.append(well_id)
    
    if not wells_with_missing:
        # Create artificial missing values for demonstration
        selected_well = df['well_id'].iloc[0]
        well_data = df[df['well_id'] == selected_well].sort_values('date')
        
        # Create a copy with artificially removed values
        well_data_copy = well_data.copy()
        
        # Remove some values
        indices_to_remove = np.random.choice(well_data.index[1:-1], size=int(len(well_data) * 0.2), replace=False)
        well_data_copy.loc[indices_to_remove, column] = np.nan
        
        # Save the true values for comparison
        true_values = well_data.loc[indices_to_remove, column]
    else:
        # Use a well with actual missing values
        selected_well = wells_with_missing[0]
        well_data = df[df['well_id'] == selected_well].sort_values('date')
        well_data_copy = well_data.copy()
        
        # Save indices with missing values
        missing_indices = well_data[well_data[column].isnull()].index
        
        # There are no true values to compare in this case
        true_values = None
    
    # Apply different interpolation methods
    methods = {
        'Linear': well_data_copy[column].interpolate(method='linear'),
        'Cubic': well_data_copy[column].interpolate(method='cubic'),
        'Time': well_data_copy[column].interpolate(method='time'),
        'Spline': well_data_copy[column].interpolate(method='spline', order=3),
        'Polynomial': well_data_copy[column].interpolate(method='polynomial', order=2)
    }
    
    # Create a plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot original data
    ax.scatter(well_data['date'], well_data[column], color='black', label='Original Data', alpha=0.5)
    
    # Plot missing values (if we have true values)
    if true_values is not None:
        ax.scatter(well_data.loc[indices_to_remove, 'date'], true_values, 
                   color='red', label='Removed Values (True)', marker='x', s=100)
    
    # Plot interpolated values with different methods
    colors = ['blue', 'green', 'orange', 'purple', 'brown']
    for (method_name, interpolated_series), color in zip(methods.items(), colors):
        ax.plot(well_data['date'], interpolated_series, label=f'{method_name} Interpolation', color=color, alpha=0.7)
    
    ax.set_title(f'Comparison of Interpolation Methods for {column} (Well ID: {selected_well})')
    ax.set_xlabel('Date')
    ax.set_ylabel(f'{column} (m below ground)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig
