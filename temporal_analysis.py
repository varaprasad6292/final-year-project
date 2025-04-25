import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import random

def analyze_temporal_patterns(data, analysis_type="Seasonal Patterns", params=None):
    """
    Analyze temporal patterns in groundwater data.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing processed data
    analysis_type : str
        Type of temporal analysis to perform
    params : dict
        Parameters for the analysis
        
    Returns:
    --------
    dict
        Analysis results including charts and insights
    """
    # Extract the processed dataframe
    df = data['data']
    
    if analysis_type == "Seasonal Patterns":
        # Default parameter
        if params is None or 'aggregation' not in params:
            aggregation = "Monthly"
        else:
            aggregation = params['aggregation']
        
        # Aggregate data by time period
        if aggregation == "Monthly":
            df['period'] = df['date'].dt.to_period('M')
            period_format = "%b %Y"
        elif aggregation == "Quarterly":
            df['period'] = df['date'].dt.to_period('Q')
            period_format = "Q%q %Y"
        else:  # Yearly
            df['period'] = df['date'].dt.to_period('Y')
            period_format = "%Y"
        
        # Group by period
        grouped = df.groupby('period')['groundwater_level'].mean().reset_index()
        grouped['period_str'] = grouped['period'].dt.strftime(period_format)
        
        # Create seasonal pattern chart
        fig = px.line(
            grouped,
            x='period_str',
            y='groundwater_level',
            markers=True,
            labels={
                'period_str': 'Time Period',
                'groundwater_level': 'Average Groundwater Level (m below ground)'
            },
            title=f'{aggregation} Average Groundwater Level'
        )
        
        # Add trend line
        z = np.polyfit(range(len(grouped)), grouped['groundwater_level'], 1)
        p = np.poly1d(z)
        trend_line = p(range(len(grouped)))
        
        fig.add_trace(go.Scatter(
            x=grouped['period_str'],
            y=trend_line,
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Trend Line'
        ))
        
        fig.update_layout(
            xaxis_tickangle=-45,
            template='plotly_white'
        )
        
        # Identify seasonal patterns
        if aggregation == "Monthly":
            # Get monthly averages across all years
            df['month'] = df['date'].dt.month
            monthly_avg = df.groupby('month')['groundwater_level'].mean().reset_index()
            
            # Find months with highest and lowest groundwater levels
            highest_month = monthly_avg.loc[monthly_avg['groundwater_level'].idxmin()]  # Lowest depth = highest water level
            lowest_month = monthly_avg.loc[monthly_avg['groundwater_level'].idxmax()]   # Highest depth = lowest water level
            
            month_names = ['January', 'February', 'March', 'April', 'May', 'June', 
                          'July', 'August', 'September', 'October', 'November', 'December']
            
            # Calculate average trend
            trend_per_year = z[0] * 12  # Convert monthly trend to yearly
            
            if trend_per_year > 0:
                trend_description = f"declining at a rate of {trend_per_year:.2f}m per year (water table dropping)"
            else:
                trend_description = f"rising at a rate of {abs(trend_per_year):.2f}m per year (water table rising)"
            
            insight = f"""
            **Seasonal Patterns in Groundwater Levels:**
            
            - The groundwater table is typically highest (minimum depth) in {month_names[highest_month['month']-1]}, with an average depth of {highest_month['groundwater_level']:.2f}m below ground.
            - The groundwater table is typically lowest (maximum depth) in {month_names[lowest_month['month']-1]}, with an average depth of {lowest_month['groundwater_level']:.2f}m below ground.
            - This represents a seasonal variation of {lowest_month['groundwater_level'] - highest_month['groundwater_level']:.2f}m.
            - The long-term trend shows groundwater levels are {trend_description}.
            
            These patterns are likely influenced by seasonal rainfall, irrigation practices, and the natural groundwater recharge cycle.
            """
        else:
            # Simplify insights for quarterly or yearly data
            if z[0] > 0:
                trend_description = f"declining at a rate of {z[0]:.4f}m per {aggregation.lower()[:-2]} (water table dropping)"
            else:
                trend_description = f"rising at a rate of {abs(z[0]):.4f}m per {aggregation.lower()[:-2]} (water table rising)"
                
            insight = f"""
            **Temporal Patterns in Groundwater Levels:**
            
            - The {aggregation.lower()} data shows a clear trend with groundwater levels {trend_description}.
            - The average groundwater level over this period is {grouped['groundwater_level'].mean():.2f}m below ground.
            - The range of variation in the {aggregation.lower()} averages is {grouped['groundwater_level'].max() - grouped['groundwater_level'].min():.2f}m.
            
            This trend may be influenced by long-term climate patterns, changes in extraction rates, or land use changes.
            """
        
    elif analysis_type == "Long-term Trends":
        # Default parameter
        if params is None or 'trend_period' not in params:
            trend_period = 5
        else:
            trend_period = params['trend_period']
        
        # Calculate yearly averages
        df['year'] = df['date'].dt.year
        yearly_avg = df.groupby('year')['groundwater_level'].mean().reset_index()
        
        # Calculate moving averages
        yearly_avg[f'{trend_period}yr_mavg'] = yearly_avg['groundwater_level'].rolling(window=trend_period, min_periods=1).mean()
        
        # Create long-term trend chart
        fig = go.Figure()
        
        # Add yearly data points
        fig.add_trace(go.Scatter(
            x=yearly_avg['year'],
            y=yearly_avg['groundwater_level'],
            mode='markers',
            name='Yearly Average',
            marker=dict(color='blue')
        ))
        
        # Add moving average line
        fig.add_trace(go.Scatter(
            x=yearly_avg['year'],
            y=yearly_avg[f'{trend_period}yr_mavg'],
            mode='lines',
            name=f'{trend_period}-Year Moving Average',
            line=dict(color='red', width=3)
        ))
        
        # Add trend line
        z = np.polyfit(yearly_avg['year'], yearly_avg['groundwater_level'], 1)
        p = np.poly1d(z)
        trend_line = p(yearly_avg['year'])
        
        fig.add_trace(go.Scatter(
            x=yearly_avg['year'],
            y=trend_line,
            mode='lines',
            line=dict(color='green', dash='dash'),
            name='Linear Trend'
        ))
        
        fig.update_layout(
            title=f'Long-term Groundwater Level Trends ({min(yearly_avg["year"])}-{max(yearly_avg["year"])})',
            xaxis_title='Year',
            yaxis_title='Average Groundwater Level (m below ground)',
            template='plotly_white'
        )
        
        # Calculate trend rate
        annual_trend = z[0]
        
        # Generate insight
        if annual_trend > 0:
            trend_description = f"declining at a rate of {annual_trend:.4f}m per year (water table dropping)"
            if annual_trend > 0.5:
                severity = "a severe rate of depletion"
            elif annual_trend > 0.2:
                severity = "a concerning rate of depletion"
            else:
                severity = "a moderate rate of depletion"
        else:
            trend_description = f"rising at a rate of {abs(annual_trend):.4f}m per year (water table rising)"
            if abs(annual_trend) > 0.5:
                severity = "a significant rate of recovery"
            elif abs(annual_trend) > 0.2:
                severity = "a moderate rate of recovery"
            else:
                severity = "a slight rate of recovery"
        
        # Calculate significant turning points
        turning_points = []
        for i in range(2, len(yearly_avg) - 2):
            prev_trend = yearly_avg.iloc[i]['groundwater_level'] - yearly_avg.iloc[i-2]['groundwater_level']
            next_trend = yearly_avg.iloc[i+2]['groundwater_level'] - yearly_avg.iloc[i]['groundwater_level']
            
            if (prev_trend * next_trend < 0) and (abs(prev_trend) > 0.5 or abs(next_trend) > 0.5):
                turning_points.append((yearly_avg.iloc[i]['year'], yearly_avg.iloc[i]['groundwater_level']))
        
        turning_points_text = ""
        if turning_points:
            turning_points_text = "Significant turning points in the trend occurred in "
            turning_points_text += ", ".join([f"{year} ({level:.2f}m)" for year, level in turning_points])
            turning_points_text += "."
        
        insight = f"""
        **Long-term Groundwater Level Trends:**
        
        - Over the period from {min(yearly_avg['year'])} to {max(yearly_avg['year'])}, groundwater levels are {trend_description}.
        - This represents {severity} and suggests that {'extraction may be exceeding natural recharge' if annual_trend > 0 else 'natural recharge may be exceeding extraction'}.
        - The {trend_period}-year moving average helps filter out short-term fluctuations, revealing the underlying trend.
        - If the current trend continues, groundwater levels could {'rise' if annual_trend < 0 else 'decline'} by approximately {abs(annual_trend * 10):.2f}m over the next decade.
        
        {turning_points_text}
        
        This long-term trend may be influenced by climate change, changing extraction patterns, land use changes, or groundwater management policies.
        """
        
    else:  # Anomaly Detection
        # Default parameter
        if params is None or 'sensitivity' not in params:
            sensitivity = 3
        else:
            sensitivity = params['sensitivity']
        
        # Select a representative well for anomaly detection
        well_ids = df['well_id'].unique()
        selected_well = random.choice(well_ids)
        well_data = df[df['well_id'] == selected_well].sort_values('date')
        
        # Set up time series for analysis
        ts = well_data.set_index('date')['groundwater_level']
        
        # Try to decompose the time series (if enough data)
        try:
            decomposition = seasonal_decompose(ts, model='additive', period=12)
            trend = decomposition.trend
            seasonal = decomposition.seasonal
            residual = decomposition.resid
            
            # Identify anomalies based on residuals
            std_dev = residual.std()
            threshold = sensitivity * std_dev
            
            anomalies = ts[(residual > threshold) | (residual < -threshold)]
            
            # Create anomaly detection chart
            fig = go.Figure()
            
            # Add original time series
            fig.add_trace(go.Scatter(
                x=ts.index,
                y=ts.values,
                mode='lines',
                name='Groundwater Level',
                line=dict(color='blue')
            ))
            
            # Add trend
            fig.add_trace(go.Scatter(
                x=trend.index,
                y=trend.values,
                mode='lines',
                name='Trend',
                line=dict(color='green')
            ))
            
            # Add anomalies
            fig.add_trace(go.Scatter(
                x=anomalies.index,
                y=anomalies.values,
                mode='markers',
                name='Anomalies',
                marker=dict(color='red', size=10, symbol='circle-open')
            ))
            
            fig.update_layout(
                title=f'Anomaly Detection in Groundwater Levels (Well ID: {selected_well})',
                xaxis_title='Date',
                yaxis_title='Groundwater Level (m below ground)',
                template='plotly_white'
            )
            
            # Generate insight
            if len(anomalies) > 0:
                anomaly_pct = (len(anomalies) / len(ts)) * 100
                most_recent = max(anomalies.index)
                most_severe_idx = (residual.abs()).idxmax()
                most_severe_date = most_severe_idx
                most_severe_value = ts[most_severe_idx]
                expected_value = most_severe_value - residual[most_severe_idx]
                
                insight = f"""
                **Anomaly Detection Results:**
                
                - {len(anomalies)} anomalies were detected, representing {anomaly_pct:.1f}% of all measurements.
                - The most recent anomaly occurred on {most_recent.strftime('%Y-%m-%d')}.
                - The most severe anomaly occurred on {most_severe_date.strftime('%Y-%m-%d')}, with a groundwater level of {most_severe_value:.2f}m (expected: {expected_value:.2f}m).
                - Anomalies may be caused by extreme weather events, sudden changes in extraction rates, measurement errors, or other unusual factors.
                
                The sensitivity setting for anomaly detection is currently {sensitivity}, where higher values detect only the most extreme anomalies.
                """
            else:
                insight = f"""
                **Anomaly Detection Results:**
                
                No significant anomalies were detected in the groundwater level data for Well ID {selected_well}.
                
                This suggests that groundwater levels have followed expected seasonal and trend patterns without major disruptions.
                
                The sensitivity setting for anomaly detection is currently {sensitivity}. Consider lowering this value to detect more subtle anomalies.
                """
        
        except:
            # If decomposition fails, use a simpler approach
            # Calculate rolling mean and standard deviation
            rolling_mean = ts.rolling(window=12, min_periods=1).mean()
            rolling_std = ts.rolling(window=12, min_periods=1).std()
            
            # Identify anomalies
            upper_bound = rolling_mean + sensitivity * rolling_std
            lower_bound = rolling_mean - sensitivity * rolling_std
            
            anomalies = ts[(ts > upper_bound) | (ts < lower_bound)]
            
            # Create anomaly detection chart
            fig = go.Figure()
            
            # Add original time series
            fig.add_trace(go.Scatter(
                x=ts.index,
                y=ts.values,
                mode='lines',
                name='Groundwater Level',
                line=dict(color='blue')
            ))
            
            # Add rolling mean
            fig.add_trace(go.Scatter(
                x=rolling_mean.index,
                y=rolling_mean.values,
                mode='lines',
                name='12-Month Moving Average',
                line=dict(color='green')
            ))
            
            # Add bounds
            fig.add_trace(go.Scatter(
                x=upper_bound.index,
                y=upper_bound.values,
                mode='lines',
                name='Upper Bound',
                line=dict(color='rgba(255,0,0,0.3)', dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=lower_bound.index,
                y=lower_bound.values,
                mode='lines',
                name='Lower Bound',
                line=dict(color='rgba(255,0,0,0.3)', dash='dash'),
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.1)'
            ))
            
            # Add anomalies
            fig.add_trace(go.Scatter(
                x=anomalies.index,
                y=anomalies.values,
                mode='markers',
                name='Anomalies',
                marker=dict(color='red', size=10, symbol='circle-open')
            ))
            
            fig.update_layout(
                title=f'Anomaly Detection in Groundwater Levels (Well ID: {selected_well})',
                xaxis_title='Date',
                yaxis_title='Groundwater Level (m below ground)',
                template='plotly_white'
            )
            
            # Generate insight
            if len(anomalies) > 0:
                anomaly_pct = (len(anomalies) / len(ts)) * 100
                insight = f"""
                **Anomaly Detection Results:**
                
                - {len(anomalies)} anomalies were detected, representing {anomaly_pct:.1f}% of all measurements.
                - Anomalies are defined as measurements that deviate more than {sensitivity} standard deviations from the 12-month moving average.
                - Anomalies may be caused by extreme weather events, sudden changes in extraction rates, measurement errors, or other unusual factors.
                
                The sensitivity setting for anomaly detection is currently {sensitivity}, where higher values detect only the most extreme anomalies.
                """
            else:
                insight = f"""
                **Anomaly Detection Results:**
                
                No significant anomalies were detected in the groundwater level data for Well ID {selected_well}.
                
                This suggests that groundwater levels have followed expected patterns without major disruptions.
                
                The sensitivity setting for anomaly detection is currently {sensitivity}. Consider lowering this value to detect more subtle anomalies.
                """
    
    return {
        'chart': fig,
        'insight': insight
    }

def forecast_future_levels(data, period=6):
    """
    Forecast future groundwater levels.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing processed data
    period : int
        Number of months to forecast
        
    Returns:
    --------
    dict
        Forecast results including charts and insights
    """
    # Extract the processed dataframe
    df = data['data']
    
    # Choose a representative well with sufficient data
    well_counts = df.groupby('well_id').size()
    eligible_wells = well_counts[well_counts >= 24].index  # At least 2 years of data
    
    if len(eligible_wells) > 0:
        selected_well = random.choice(eligible_wells)
    else:
        selected_well = random.choice(df['well_id'].unique())
    
    well_data = df[df['well_id'] == selected_well].sort_values('date')
    
    # Create time series
    ts = well_data.set_index('date')['groundwater_level']
    
    # Train Holt-Winters model
    model = ExponentialSmoothing(
        ts,
        trend='add',
        seasonal='add',
        seasonal_periods=12  # Assuming monthly data with annual seasonality
    )
    
    fitted_model = model.fit()
    
    # Forecast future values
    forecast = fitted_model.forecast(period)
    
    # Create prediction intervals (simplified)
    pred_interval = fitted_model.get_prediction_interval(forecast)
    
    # Create forecast chart
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=ts.index,
        y=ts.values,
        mode='lines',
        name='Historical Data',
        line=dict(color='blue')
    ))
    
    # Add forecasted values
    fig.add_trace(go.Scatter(
        x=forecast.index,
        y=forecast.values,
        mode='lines',
        name='Forecast',
        line=dict(color='red')
    ))
    
    # Add prediction intervals
    fig.add_trace(go.Scatter(
        x=forecast.index.tolist() + forecast.index.tolist()[::-1],
        y=pred_interval['upper'].tolist() + pred_interval['lower'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(255,0,0,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% Prediction Interval'
    ))
    
    fig.update_layout(
        title=f'Groundwater Level Forecast for Well ID: {selected_well}',
        xaxis_title='Date',
        yaxis_title='Groundwater Level (m below ground)',
        template='plotly_white'
    )
    
    # Generate forecast insight
    last_value = ts.iloc[-1]
    forecast_end = forecast.iloc[-1]
    change = forecast_end - last_value
    
    if change > 0:
        change_direction = f"decline by {change:.2f}m (water table dropping)"
    else:
        change_direction = f"rise by {abs(change):.2f}m (water table rising)"
    
    # Seasonal pattern in forecast
    if period >= 12:
        forecast_monthly = forecast.groupby(forecast.index.month).mean()
        highest_month = forecast_monthly.idxmin()  # Lowest depth = highest water level
        lowest_month = forecast_monthly.idxmax()   # Highest depth = lowest water level
        
        month_names = ['January', 'February', 'March', 'April', 'May', 'June', 
                      'July', 'August', 'September', 'October', 'November', 'December']
        
        seasonal_insight = f"""
        The forecast shows seasonal patterns with:
        - Highest water table (minimum depth) expected in {month_names[highest_month-1]}
        - Lowest water table (maximum depth) expected in {month_names[lowest_month-1]}
        """
    else:
        seasonal_insight = ""
    
    forecast_insight = f"""
    **Groundwater Level Forecast:**
    
    Based on historical patterns, the groundwater level for Well ID {selected_well} is expected to {change_direction} over the next {period} months.
    
    {seasonal_insight}
    
    The forecast takes into account historical seasonal patterns, trends, and cyclical components in the data. The prediction interval indicates the range of likely values, with wider intervals representing greater uncertainty in the forecast.
    """
    
    return {
        'well_id': selected_well,
        'historical': ts,
        'forecast': forecast,
        'prediction_interval': pred_interval,
        'forecast_chart': fig,
        'forecast_insight': forecast_insight
    }
