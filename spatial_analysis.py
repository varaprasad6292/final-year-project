import folium
from folium.plugins import HeatMap, MarkerCluster
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
import random
from folium.plugins import HeatMapWithTime

def create_spatial_map(data, map_type="Groundwater Level"):
    """
    Create a spatial map visualization of groundwater data.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing processed data
    map_type : str
        Type of map to create (Groundwater Level, Groundwater Depletion, or Risk Assessment)
        
    Returns:
    --------
    folium.Map
        Interactive folium map
    """
    # Extract the processed dataframe
    df = data['data']
    
    # Get the latest data for each well
    latest_data = df.sort_values('date').drop_duplicates('well_id', keep='last')
    
    # Center the map on India
    map_center = [20.5937, 78.9629]
    
    # Create a map
    m = folium.Map(location=map_center, zoom_start=5, tiles='CartoDB positron')
    
    if map_type == "Groundwater Level":
        # Create a MarkerCluster for well locations
        marker_cluster = MarkerCluster().add_to(m)
        
        # Add markers for each well
        for _, row in latest_data.iterrows():
            # Define color based on groundwater level
            if row['groundwater_level'] < 5:
                color = 'blue'
                risk = 'High water table'
            elif row['groundwater_level'] < 10:
                color = 'green'
                risk = 'Moderate water table'
            elif row['groundwater_level'] < 20:
                color = 'orange'
                risk = 'Low water table'
            elif row['groundwater_level'] < 30:
                color = 'red'
                risk = 'Very low water table'
            else:
                color = 'darkred'
                risk = 'Critical water table'
            
            # Create popup content
            popup_content = f"""
            <strong>Well ID:</strong> {row['well_id']}<br>
            <strong>Groundwater Level:</strong> {row['groundwater_level']:.2f} m below ground<br>
            <strong>Status:</strong> {risk}<br>
            <strong>Rainfall:</strong> {row['rainfall']:.1f} mm<br>
            <strong>Soil Moisture:</strong> {row['soil_moisture']:.1f}%<br>
            <strong>Land Use:</strong> {row['predominant_land_use']}<br>
            <strong>Hydrogeology:</strong> {row['hydrogeology']}
            """
            
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=folium.Popup(popup_content, max_width=300),
                icon=folium.Icon(color=color, icon='tint', prefix='fa')
            ).add_to(marker_cluster)
        
        # Add a heatmap layer
        heat_data = [[row['latitude'], row['longitude'], row['groundwater_level']] 
                    for _, row in latest_data.iterrows()]
        HeatMap(heat_data, radius=15, blur=10, gradient={
            0.2: 'blue', 0.4: 'cyan', 0.6: 'lime', 0.8: 'yellow', 1: 'red'
        }).add_to(m)
        
    elif map_type == "Groundwater Depletion":
        # Calculate depletion rate for each well
        depletion_rates = []
        
        for well_id in df['well_id'].unique():
            well_data = df[df['well_id'] == well_id].sort_values('date')
            
            if len(well_data) < 2:
                continue
                
            # Calculate yearly depletion rate
            first_year = well_data.iloc[0]['date'].year
            last_year = well_data.iloc[-1]['date'].year
            years_diff = max(1, last_year - first_year)
            
            first_level = well_data.iloc[0]['groundwater_level']
            last_level = well_data.iloc[-1]['groundwater_level']
            level_diff = last_level - first_level
            
            annual_rate = level_diff / years_diff
            
            # Get the latest data for this well
            latest = well_data.iloc[-1]
            
            depletion_rates.append({
                'well_id': well_id,
                'latitude': latest['latitude'],
                'longitude': latest['longitude'],
                'annual_depletion_rate': annual_rate,
                'current_level': last_level,
                'hydrogeology': latest['hydrogeology'],
                'predominant_land_use': latest['predominant_land_use']
            })
        
        depletion_df = pd.DataFrame(depletion_rates)
        
        # Create a MarkerCluster
        marker_cluster = MarkerCluster().add_to(m)
        
        # Add markers for each well
        for _, row in depletion_df.iterrows():
            # Define color based on depletion rate
            if row['annual_depletion_rate'] <= 0:
                color = 'blue'
                status = 'Improving'
            elif row['annual_depletion_rate'] < 0.1:
                color = 'green'
                status = 'Minimal depletion'
            elif row['annual_depletion_rate'] < 0.3:
                color = 'lime'
                status = 'Low depletion'
            elif row['annual_depletion_rate'] < 0.5:
                color = 'orange'
                status = 'Moderate depletion'
            elif row['annual_depletion_rate'] < 1.0:
                color = 'red'
                status = 'High depletion'
            else:
                color = 'darkred'
                status = 'Severe depletion'
            
            # Create popup content
            popup_content = f"""
            <strong>Well ID:</strong> {row['well_id']}<br>
            <strong>Annual Depletion Rate:</strong> {row['annual_depletion_rate']:.2f} m/year<br>
            <strong>Status:</strong> {status}<br>
            <strong>Current Level:</strong> {row['current_level']:.2f} m below ground<br>
            <strong>Land Use:</strong> {row['predominant_land_use']}<br>
            <strong>Hydrogeology:</strong> {row['hydrogeology']}
            """
            
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=folium.Popup(popup_content, max_width=300),
                icon=folium.Icon(color=color, icon='tint', prefix='fa')
            ).add_to(marker_cluster)
        
        # Add a heatmap layer for depletion rate
        heat_data = [[row['latitude'], row['longitude'], max(0, row['annual_depletion_rate'] * 5)] 
                    for _, row in depletion_df.iterrows()]
        HeatMap(heat_data, radius=15, blur=10, gradient={
            0.2: 'blue', 0.4: 'lime', 0.6: 'yellow', 0.8: 'orange', 1: 'red'
        }).add_to(m)
        
    else:  # Risk Assessment
        # Define risk factors and calculate risk score
        risk_scores = []
        
        for well_id in df['well_id'].unique():
            well_data = df[df['well_id'] == well_id].sort_values('date')
            
            if len(well_data) < 2:
                continue
                
            # Get the latest data for this well
            latest = well_data.iloc[-1]
            
            # Calculate depletion rate
            first_year = well_data.iloc[0]['date'].year
            last_year = well_data.iloc[-1]['date'].year
            years_diff = max(1, last_year - first_year)
            
            first_level = well_data.iloc[0]['groundwater_level']
            last_level = well_data.iloc[-1]['groundwater_level']
            level_diff = last_level - first_level
            
            annual_rate = level_diff / years_diff
            
            # Calculate risk factors
            level_risk = min(1, latest['groundwater_level'] / 30)  # Normalize to 0-1
            depletion_risk = min(1, max(0, annual_rate) / 1.0)  # Normalize to 0-1
            rainfall_risk = 1 - min(1, latest['rainfall'] / 1000)  # Normalize to 0-1
            population_risk = min(1, latest['population_density'] / 1000)  # Normalize to 0-1
            
            # Calculate overall risk (weighted average)
            overall_risk = (
                level_risk * 0.3 +
                depletion_risk * 0.4 +
                rainfall_risk * 0.2 +
                population_risk * 0.1
            )
            
            risk_scores.append({
                'well_id': well_id,
                'latitude': latest['latitude'],
                'longitude': latest['longitude'],
                'risk_score': overall_risk,
                'current_level': last_level,
                'annual_depletion_rate': annual_rate,
                'rainfall': latest['rainfall'],
                'population_density': latest['population_density'],
                'hydrogeology': latest['hydrogeology'],
                'predominant_land_use': latest['predominant_land_use']
            })
        
        risk_df = pd.DataFrame(risk_scores)
        
        # Create a MarkerCluster
        marker_cluster = MarkerCluster().add_to(m)
        
        # Add markers for each well
        for _, row in risk_df.iterrows():
            # Define color based on risk score
            if row['risk_score'] < 0.2:
                color = 'blue'
                risk_level = 'Very Low Risk'
            elif row['risk_score'] < 0.4:
                color = 'green'
                risk_level = 'Low Risk'
            elif row['risk_score'] < 0.6:
                color = 'orange'
                risk_level = 'Moderate Risk'
            elif row['risk_score'] < 0.8:
                color = 'red'
                risk_level = 'High Risk'
            else:
                color = 'darkred'
                risk_level = 'Critical Risk'
            
            # Create popup content
            popup_content = f"""
            <strong>Well ID:</strong> {row['well_id']}<br>
            <strong>Risk Level:</strong> {risk_level} ({row['risk_score']:.2f})<br>
            <strong>Current Level:</strong> {row['current_level']:.2f} m below ground<br>
            <strong>Annual Depletion Rate:</strong> {row['annual_depletion_rate']:.2f} m/year<br>
            <strong>Rainfall:</strong> {row['rainfall']:.1f} mm<br>
            <strong>Population Density:</strong> {row['population_density']:.1f} people/km²<br>
            <strong>Land Use:</strong> {row['predominant_land_use']}<br>
            <strong>Hydrogeology:</strong> {row['hydrogeology']}
            """
            
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=folium.Popup(popup_content, max_width=300),
                icon=folium.Icon(color=color, icon='exclamation-triangle', prefix='fa')
            ).add_to(marker_cluster)
        
        # Add a heatmap layer for risk score
        heat_data = [[row['latitude'], row['longitude'], row['risk_score']] 
                    for _, row in risk_df.iterrows()]
        HeatMap(heat_data, radius=15, blur=10, gradient={
            0.2: 'blue', 0.4: 'green', 0.6: 'yellow', 0.8: 'orange', 1: 'red'
        }).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m

def analyze_spatial_patterns(data):
    """
    Analyze spatial patterns in groundwater data.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing processed data
        
    Returns:
    --------
    dict
        Analysis results including charts and insights
    """
    # Extract the processed dataframe
    df = data['data']
    
    # Get the latest data for each well
    latest_data = df.sort_values('date').drop_duplicates('well_id', keep='last')
    
    # Hydrogeological analysis
    hydro_data = latest_data.groupby('hydrogeology')['groundwater_level'].agg(['mean', 'std', 'count']).reset_index()
    hydro_data = hydro_data.sort_values('mean')
    
    hydro_fig = px.bar(
        hydro_data,
        x='hydrogeology',
        y='mean',
        error_y='std',
        color='count',
        labels={
            'hydrogeology': 'Hydrogeological Formation',
            'mean': 'Mean Groundwater Level (m below ground)',
            'count': 'Number of Wells'
        },
        title='Groundwater Level by Hydrogeological Formation',
        color_continuous_scale='Viridis'
    )
    
    hydro_insight = f"""
    Hydrogeological formations significantly impact groundwater levels across regions:
    
    - {hydro_data.iloc[0]['hydrogeology']} formations show the highest water table (shallowest depth) at {hydro_data.iloc[0]['mean']:.2f}m below ground on average.
    - {hydro_data.iloc[-1]['hydrogeology']} formations have the lowest water table (deepest depth) at {hydro_data.iloc[-1]['mean']:.2f}m below ground on average.
    - The variation (standard deviation) is greatest in {hydro_data.sort_values('std', ascending=False).iloc[0]['hydrogeology']} formations at {hydro_data.sort_values('std', ascending=False).iloc[0]['std']:.2f}m, indicating high spatial heterogeneity.
    """
    
    # Land use impact analysis
    land_use_data = latest_data.groupby('predominant_land_use')['groundwater_level'].agg(['mean', 'min', 'max', 'count']).reset_index()
    land_use_data['range'] = land_use_data['max'] - land_use_data['min']
    land_use_data = land_use_data.sort_values('mean')
    
    land_use_fig = px.bar(
        land_use_data,
        x='predominant_land_use',
        y='mean',
        color='range',
        labels={
            'predominant_land_use': 'Land Use Type',
            'mean': 'Mean Groundwater Level (m below ground)',
            'range': 'Range (m)'
        },
        title='Groundwater Level by Land Use Type',
        color_continuous_scale='RdYlBu_r'
    )
    
    land_use_insight = f"""
    Land use patterns show clear relationships with groundwater levels:
    
    - {land_use_data.iloc[0]['predominant_land_use']} areas have the highest water table (shallowest depth) at {land_use_data.iloc[0]['mean']:.2f}m below ground on average.
    - {land_use_data.iloc[-1]['predominant_land_use']} areas show the lowest water table (deepest depth) at {land_use_data.iloc[-1]['mean']:.2f}m below ground on average.
    - The largest range in groundwater levels is observed in {land_use_data.sort_values('range', ascending=False).iloc[0]['predominant_land_use']} areas at {land_use_data.sort_values('range', ascending=False).iloc[0]['range']:.2f}m.
    - Urban areas typically show deeper groundwater levels than rural or forested areas, likely due to factors such as increased groundwater extraction, reduced recharge due to impervious surfaces, and possible contamination issues.
    """
    
    return {
        'hydrogeological_chart': hydro_fig,
        'hydrogeological_insight': hydro_insight,
        'land_use_chart': land_use_fig,
        'land_use_insight': land_use_insight
    }
