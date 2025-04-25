import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

def create_trend_chart(data):
    """
    Create a trend chart for groundwater levels.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing processed data
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    # Extract the processed dataframe
    df = data['data']
    
    # Calculate monthly averages
    df['year_month'] = df['date'].dt.to_period('M')
    monthly_avg = df.groupby('year_month')['groundwater_level'].mean().reset_index()
    monthly_avg['date'] = monthly_avg['year_month'].dt.to_timestamp()
    
    # Calculate 12-month moving average
    monthly_avg['12_month_ma'] = monthly_avg['groundwater_level'].rolling(window=12, min_periods=1).mean()
    
    # Create figure
    fig = go.Figure()
    
    # Add monthly averages
    fig.add_trace(go.Scatter(
        x=monthly_avg['date'],
        y=monthly_avg['groundwater_level'],
        mode='lines',
        name='Monthly Average',
        line=dict(color='rgba(0, 119, 182, 0.6)')
    ))
    
    # Add moving average
    fig.add_trace(go.Scatter(
        x=monthly_avg['date'],
        y=monthly_avg['12_month_ma'],
        mode='lines',
        name='12-Month Moving Average',
        line=dict(color='red', width=3)
    ))
    
    # Add trend line
    z = np.polyfit(range(len(monthly_avg)), monthly_avg['groundwater_level'], 1)
    p = np.poly1d(z)
    
    fig.add_trace(go.Scatter(
        x=monthly_avg['date'],
        y=p(range(len(monthly_avg))),
        mode='lines',
        name='Long-term Trend',
        line=dict(color='green', dash='dash')
    ))
    
    # Calculate rainfall (on second y-axis)
    monthly_rainfall = df.groupby('year_month')['rainfall'].mean().reset_index()
    monthly_rainfall['date'] = monthly_rainfall['year_month'].dt.to_timestamp()
    
    # Add rainfall as bar chart on secondary y-axis
    fig.add_trace(go.Bar(
        x=monthly_rainfall['date'],
        y=monthly_rainfall['rainfall'],
        name='Rainfall',
        marker_color='rgba(0, 180, 216, 0.3)',
        yaxis='y2'
    ))
    
    # Update layout
    fig.update_layout(
        title='Groundwater Level Trends and Rainfall',
        xaxis_title='Date',
        yaxis=dict(
            title='Groundwater Level (m below ground)',
            titlefont=dict(color='blue'),
            tickfont=dict(color='blue')
        ),
        yaxis2=dict(
            title='Rainfall (mm)',
            titlefont=dict(color='rgba(0, 180, 216, 1)'),
            tickfont=dict(color='rgba(0, 180, 216, 1)'),
            anchor='x',
            overlaying='y',
            side='right'
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig

def create_forecasting_chart(forecast_results):
    """
    Create a chart for forecasted groundwater levels.
    
    Parameters:
    -----------
    forecast_results : dict
        Dictionary containing forecast results
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    # Extract forecast data
    historical = forecast_results['historical']
    forecast = forecast_results['forecast']
    prediction_interval = forecast_results['prediction_interval']
    
    # Create figure
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=historical.index,
        y=historical.values,
        mode='lines',
        name='Historical Data',
        line=dict(color='blue')
    ))
    
    # Add forecast
    fig.add_trace(go.Scatter(
        x=forecast.index,
        y=forecast.values,
        mode='lines',
        name='Forecast',
        line=dict(color='red')
    ))
    
    # Add prediction interval
    fig.add_trace(go.Scatter(
        x=forecast.index.tolist() + forecast.index.tolist()[::-1],
        y=prediction_interval['upper'].tolist() + prediction_interval['lower'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.2)',
        line=dict(color='rgba(255, 255, 255, 0)'),
        name='Prediction Interval'
    ))
    
    # Update layout
    fig.update_layout(
        title=f'Groundwater Level Forecast for Well ID: {forecast_results["well_id"]}',
        xaxis_title='Date',
        yaxis_title='Groundwater Level (m below ground)',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig

def create_correlation_heatmap(data, selected_factors):
    """
    Create a correlation heatmap for selected factors.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing processed data
    selected_factors : list
        List of factors to include in the heatmap
        
    Returns:
    --------
    matplotlib.figure.Figure
        Matplotlib figure object
    """
    # Extract the processed dataframe
    df = data['data']
    
    # Map selected factors to actual dataframe columns
    factor_columns = {
        'Rainfall': 'rainfall',
        'Temperature': 'temperature',
        'Population Density': 'population_density',
        'Elevation': 'elevation',
        'Soil Moisture': 'soil_moisture',
        'Land Use': ['land_use_agriculture', 'land_use_urban', 'land_use_forest'],
        'Hydrogeology': 'hydrogeology_code'
    }
    
    # Collect columns to include in correlation
    columns_to_include = ['groundwater_level']
    
    for factor in selected_factors:
        if factor in factor_columns:
            if isinstance(factor_columns[factor], list):
                columns_to_include.extend(factor_columns[factor])
            else:
                columns_to_include.append(factor_columns[factor])
    
    # Filter columns that actually exist in the dataframe
    existing_columns = [col for col in columns_to_include if col in df.columns]
    
    # Create correlation matrix
    correlation = df[existing_columns].corr()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.zeros_like(correlation)
    mask[np.triu_indices_from(mask)] = True
    
    sns.heatmap(
        correlation,
        mask=mask,
        annot=True,
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=.5,
        cbar_kws={'shrink': .7},
        ax=ax
    )
    
    ax.set_title('Correlation Heatmap of Groundwater Influencing Factors')
    
    return fig
