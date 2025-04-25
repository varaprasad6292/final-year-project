import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from utils.data_processor import load_data, preprocess_data, load_and_preprocess_data, get_related_datasets
from utils.visualizations import plot_groundwater_heatmap, plot_groundwater_timeseries

# Configure page
st.set_page_config(
    page_title="India Groundwater Analysis",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
st.sidebar.title("💧 Groundwater Level Analysis")
st.sidebar.info(
    "This application provides analysis and forecasting of groundwater levels across India "
    "using machine learning models and spatiotemporal analysis."
)

# Navigation
pages = {
    "Dashboard": "Dashboard",
    "Data Explorer": "pages/data_explorer.py",
    "Time Series Analysis": "pages/time_series_analysis.py",
    "Spatial Analysis": "pages/spatial_analysis.py",
    "Forecasting": "pages/forecast.py"
}

# Main content
st.title("India Groundwater Level Analysis and Forecasting")

# Introduction
st.markdown("""
This application integrates groundwater data from the Central Ground Water Board (CGWB) 
and other agencies, with capabilities to analyze and forecast groundwater levels 
across India. The system leverages machine learning models like LSTM, Random Forest, 
and ARIMA to provide spatiotemporal predictions.
""")

# Data source selection
st.sidebar.header("Data Source")
data_source = st.sidebar.radio(
    "Select Data Source",
    ["Central Ground Water Board (CGWB)", "Other Agencies"],
    index=0
)

# Filters
st.sidebar.header("Data Filters")
# Date filters - default to last 4 years
end_date = datetime.now()
start_date = end_date - timedelta(days=365*4)

start_date_input = st.sidebar.date_input("Start Date", start_date)
end_date_input = st.sidebar.date_input("End Date", end_date)

if start_date_input > end_date_input:
    st.sidebar.error("Error: Start date must be before end date.")
    
# Load data
with st.spinner("Loading groundwater data..."):
    try:
        # Convert date inputs to datetime
        start_date_dt = datetime.combine(start_date_input, datetime.min.time())
        end_date_dt = datetime.combine(end_date_input, datetime.max.time())
        
        # Use our combined load and preprocess function with filters
        data_dict = load_and_preprocess_data(
            state="All States", 
            district="All Districts", 
            start_date=start_date_dt, 
            end_date=end_date_dt
        )
        
        if data_dict and 'processed_data' in data_dict and not data_dict['processed_data'].empty:
            # Extract the processed data
            df_processed = data_dict['processed_data']
            
            # Get related datasets
            related_data = get_related_datasets()
            
            # Display success message
            st.success(f"Successfully loaded data with {len(df_processed)} records from {len(data_dict['wells'])} wells.")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Wells", len(data_dict['wells']))
            
            with col2:
                st.metric("States Covered", len(data_dict['states']))
            
            with col3:
                avg_level = round(df_processed['water_level'].mean(), 2)
                st.metric("Avg. Water Level (m)", avg_level)
            
            with col4:
                year_range = f"{df_processed['date'].min().year} - {df_processed['date'].max().year}"
                st.metric("Year Range", year_range)
            
            # Add more detailed metrics
            st.subheader("Key Insights")
            col1, col2 = st.columns(2)
            
            with col1:
                # Calculate overall trend
                overall_trend = df_processed['trend'].mean()
                trend_icon = "🔺" if overall_trend > 0 else "🔻"
                st.metric(
                    "Overall Trend", 
                    f"{abs(overall_trend):.2f} m/year {trend_icon}", 
                    delta=None,
                    help="Positive values indicate increasing depth to water (depletion), negative values indicate water level rise"
                )
                
                # States with most depletion
                if 'trend' in df_processed.columns:
                    state_trends = df_processed.groupby('state')['trend'].mean().sort_values(ascending=False)
                    worst_state = state_trends.index[0]
                    worst_trend = state_trends.iloc[0]
                    st.metric(
                        "State with Highest Depletion", 
                        f"{worst_state}",
                        delta=f"{worst_trend:.2f} m/year",
                        delta_color="inverse"
                    )
            
            with col2:
                # Calculate seasonal variation
                if 'seasonal_deviation' in df_processed.columns:
                    max_seasonal_dev = df_processed.groupby('season')['seasonal_deviation'].mean().abs().max()
                    st.metric("Seasonal Variation", f"{max_seasonal_dev:.2f} m")
                
                # States with least depletion
                if 'trend' in df_processed.columns:
                    best_state = state_trends.index[-1]
                    best_trend = state_trends.iloc[-1]
                    st.metric(
                        "State with Lowest Depletion",
                        f"{best_state}",
                        delta=f"{best_trend:.2f} m/year",
                        delta_color="inverse"
                    )
            
            # Visualization section
            st.header("Groundwater Level Visualization")
            
            # 1. Map visualization
            st.subheader("Geographical Distribution")
            fig1 = plot_groundwater_heatmap(df_processed)
            st.plotly_chart(fig1, use_container_width=True)
            
            # 2. Time series visualization
            st.subheader("Temporal Trends")
            
            # State selector with all states from the data
            states = sorted(data_dict['states'])
            selected_state = st.selectbox("Select a state", states)
            
            # If a state is selected, allow selection of districts in that state
            if selected_state and selected_state in data_dict['districts']:
                districts = ["All Districts"] + sorted(data_dict['districts'][selected_state])
                selected_district = st.selectbox("Select a district", districts)
                
                # Filter data based on selection
                if selected_district and selected_district != "All Districts":
                    filtered_data = df_processed[(df_processed['state'] == selected_state) & 
                                                 (df_processed['district'] == selected_district)]
                else:
                    filtered_data = df_processed[df_processed['state'] == selected_state]
                
                # Plot time series
                fig2 = plot_groundwater_timeseries(filtered_data)
                st.plotly_chart(fig2, use_container_width=True)
                
                # Show statistics for the selected area
                st.subheader(f"Statistics for {selected_state}" + 
                           (f", {selected_district}" if selected_district != "All Districts" else ""))
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Number of wells
                    wells_count = len(filtered_data['well_id'].unique())
                    st.metric("Number of Wells", wells_count)
                
                with col2:
                    # Average water level
                    avg_level = round(filtered_data['water_level'].mean(), 2)
                    st.metric("Average Water Level", f"{avg_level} m")
                
                with col3:
                    # Trend
                    if 'trend' in filtered_data.columns:
                        trend = filtered_data['trend'].mean()
                        trend_text = f"{abs(trend):.2f} m/year"
                        if trend > 0:
                            trend_text += " (depletion)"
                        elif trend < 0:
                            trend_text += " (recharge)"
                        else:
                            trend_text += " (stable)"
                        st.metric("Trend", trend_text)
            
            # 3. Additional information
            if related_data and 'rainfall' in related_data and not related_data['rainfall'].empty:
                st.subheader("Rainfall Data")
                rainfall_data = related_data['rainfall']
                
                # Filter rainfall data for the selected state
                if selected_state:
                    state_rainfall = rainfall_data[rainfall_data['state'] == selected_state]
                    
                    # Calculate annual rainfall
                    annual_rainfall = state_rainfall.groupby(state_rainfall['date'].dt.year)['rainfall_mm'].mean()
                    
                    # Display a bar chart of annual rainfall
                    st.bar_chart(annual_rainfall)
                    
                    # Calculate correlation with groundwater levels
                    if not filtered_data.empty:
                        # Prepare data for correlation
                        gw_monthly = filtered_data.groupby(filtered_data['date'].dt.to_period('M'))['water_level'].mean()
                        rf_monthly = state_rainfall.groupby(state_rainfall['date'].dt.to_period('M'))['rainfall_mm'].mean()
                        
                        # Convert period index to string for joining
                        gw_monthly.index = gw_monthly.index.astype(str)
                        rf_monthly.index = rf_monthly.index.astype(str)
                        
                        # Find common months
                        common_months = set(gw_monthly.index).intersection(set(rf_monthly.index))
                        
                        if common_months:
                            # Filter to common months
                            gw_common = gw_monthly[gw_monthly.index.isin(common_months)]
                            rf_common = rf_monthly[rf_monthly.index.isin(common_months)]
                            
                            # Calculate correlation
                            correlation = gw_common.corr(rf_common)
                            st.metric("Correlation with Rainfall", f"{correlation:.2f}")
                            
                            # Interpretation
                            if correlation < -0.5:
                                st.info("Strong negative correlation: Higher rainfall is associated with lower groundwater levels (higher recharge).")
                            elif correlation > 0.5:
                                st.warning("Strong positive correlation: Higher rainfall is associated with higher groundwater levels (potential issues with recharge).")
                            else:
                                st.info("Weak correlation: Rainfall and groundwater levels do not show a strong relationship in this region.")
            
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Please ensure the data source is available. Using limited functionality.")

# Resources section
st.markdown("---")
st.subheader("Resources")
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    - [Central Ground Water Board](https://cgwb.gov.in/index.html)
    - [India Water Portal](https://www.indiawaterportal.org/)
    - [National Water Informatics Centre](https://nwic.gov.in/)
    """)

with col2:
    st.markdown("""
    - [Groundwater Level Data Repository](https://indiawris.gov.in/wris/)
    - [Rainfall Data - IMD](https://mausam.imd.gov.in/)
    - [Land Use Data - Bhuvan](https://bhuvan.nrsc.gov.in/)
    """)

# Footer
st.markdown("---")
st.markdown("#### About")
st.markdown("""
This application is designed for analyzing and forecasting groundwater levels in India.
It integrates data from multiple sources and employs advanced machine learning techniques
to provide insights and predictions about groundwater trends.
""")
