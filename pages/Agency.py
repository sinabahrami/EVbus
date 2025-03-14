import streamlit as st
import os
import zipfile
import pandas as pd
from datetime import datetime, timedelta
import folium
from folium import plugins
from streamlit_folium import st_folium
from geopy.distance import geodesic
from collections import Counter
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Cache the data processing to improve performance
@st.cache_data
def load_gtfs_data(zip_file_path):
    """Load and process GTFS data from a zip file."""
    dataframes = {}
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        for file_name in zip_ref.namelist():
            if file_name.endswith('.txt'):
                with zip_ref.open(file_name) as file:
                    df_name = file_name.split(".")[0]
                    dataframes[df_name] = pd.read_csv(file)
    
    return dataframes

def convert_to_datetime_over_24(time_str):
    """Convert GTFS time strings (including those over 24 hours) to timedelta."""
    if pd.isna(time_str) or time_str.lower() == "nan":
        return np.nan  # Return NaN instead of processing
    else:
        hours, minutes, seconds = map(int, time_str.split(":"))
        if hours >= 24:
            days = hours // 24
            hours = hours % 24
            time_str = f"{days} days {hours:02}:{minutes:02}:{seconds:02}"
        return pd.to_timedelta(time_str)

def compute_range_tracking(distances, time_gaps, end_id_lists, bus_range, charging_power, energy_usage, min_stoppage_time, top_end_stop_ids):
    """Compute the range tracking for a bus over a sequence of trips."""
    range_tracking = [bus_range]  # Initialize list with bus_range
    current_range = bus_range  # Initialize range tracking variable
    
    for i in range(len(distances)):
        # Decrease range by traveled distance
        current_range -= distances[i]
        range_tracking.append(current_range)

        # Increase range based on charging time
        if i < len(time_gaps):  # Ensure time_gaps is available
            if time_gaps[i] > min_stoppage_time and end_id_lists[i] in top_end_stop_ids:
                charge_added = (charging_power * (time_gaps[i]-min_stoppage_time)) / energy_usage
            else:
                charge_added = 0
            current_range = min(bus_range, current_range + charge_added)
            range_tracking.append(current_range)
    
    return range_tracking

def create_bus_electrification_map(shapes_df, routes_df, trips_df, proposed_locations_df, wireless_track_shape_df, center_lat, center_lon):
    """Create a folium map with routes and charging locations."""
    # Create the map with a specific location and zoom level
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles=None)
    
    # Add OpenStreetMap tile layer
    folium.TileLayer(
        tiles='https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
        attr='OpenStreetMap',
        name='Open Street Map',
        overlay=False,
        opacity=0.5,
        control=True
    ).add_to(m)
    
    # Create a feature group for all routes
    all_routes = folium.FeatureGroup(name="All Routes", show=True)
    
    # Get unique route IDs
    unique_routes = trips_df['route_id'].unique()
    num_routes = len(unique_routes)
    
    # Generate colors for routes using a colormap
    colormap = matplotlib.colormaps.get_cmap('tab20b')  # Only pass the colormap name
    colors = [mcolors.to_hex(colormap(i / (num_routes - 1))) for i in range(num_routes)]  # Normalize indices
    route_colors = dict(zip(unique_routes, colors))
    
    # Add route lines to the map
    for route_id in unique_routes:
        # Get shape IDs for this route
        route_shapes = trips_df[trips_df['route_id'] == route_id]['shape_id'].unique()
        
        # Create a feature group for this route
        route_group = folium.FeatureGroup(name=f"Route {route_id}", show=False)
        
        # Get the route color if available, otherwise use the generated color
        try:
            #route_color_info = routes_df[routes_df['route_id'] == route_id]['route_color'].iloc[0]
            #route_color = f"#{route_color_info}" if not pd.isna(route_color_info) else route_colors[route_id]
            route_color = route_colors[route_id]
        except (IndexError, KeyError):
            route_color = route_colors[route_id]
        
        # Add each shape for this route
        for shape_id in route_shapes:
            shape_points = shapes_df[shapes_df['shape_id'] == shape_id].sort_values('shape_pt_sequence')
            coordinates = shape_points[['shape_pt_lat', 'shape_pt_lon']].values.tolist()
            
            if len(coordinates) > 1:  # Ensure we have at least 2 points to make a line
                # Add to the route-specific group
                folium.PolyLine(
                    coordinates,
                    color=route_color,
                    weight=4,
                    #opacity=1,
                    tooltip=f"Route {route_id} - Shape {shape_id}"
                ).add_to(route_group)
                
                # Also add to the "All Routes" group
                folium.PolyLine(
                    coordinates,
                    color=route_color,
                    weight=4,
                    #opacity=1
                ).add_to(all_routes)
        
        # Add the route group to the map
        route_group.add_to(m)
    
    # Add the "All Routes" group to the map
    all_routes.add_to(m)

    if not wireless_track_shape_df.empty:
        counter=1
        for shape_id in wireless_track_shape_df['shape_id'].unique():
            wireless_track_group = folium.FeatureGroup(name=f"Wireless track {counter}")
            shape_data = wireless_track_shape_df[wireless_track_shape_df['shape_id'] == shape_id].sort_values(by='target_shape_pt_sequence')
            shape_coords = shape_data[['shape_pt_lat', 'shape_pt_lon']].values.tolist()
            folium.PolyLine(shape_coords, color="green", weight=4).add_to(wireless_track_group)
            counter+=1
            wireless_track_group.add_to(m)

    # Create a feature group for charging locations
    charging_locations = folium.FeatureGroup(name="Proposed Charging Locations", show=True)
    
    # Add markers for charging locations
    for idx, row in proposed_locations_df.iterrows():
        folium.Marker(
            location=[row["stop_lat"], row["stop_lon"]],
            icon=folium.Icon(color="blue", icon="plug", prefix="fa"),
            popup=f"Charging Location ID: {row['stop_id']}",
            tooltip="Charging Location"
        ).add_to(charging_locations)
    
    # Add the charging locations group to the map
    charging_locations.add_to(m)
    
    # Add layer control to toggle visibility
    folium.LayerControl().add_to(m)
    
    return m

def compute_shape_distances(df):
    df = df.sort_values(by=["shape_id", "shape_pt_sequence"]).reset_index(drop=True)

    df.loc[df["shape_pt_sequence"] == 1, "shape_dist_traveled"] = 0  # First point is 0

    # Iterate over rows and calculate distance
    for i in range(1, len(df)):  # Start from the second row
        prev_row = df.iloc[i - 1]
        current_row = df.iloc[i]

        if pd.isna(current_row["shape_dist_traveled"]):  # Only calculate if missing
            prev_coords = (prev_row["shape_pt_lat"], prev_row["shape_pt_lon"])
            current_coords = (current_row["shape_pt_lat"], current_row["shape_pt_lon"])
            distance = geodesic(prev_coords, current_coords).meters  # Distance in meters

            df.loc[i, "shape_dist_traveled"] = prev_row["shape_dist_traveled"] + distance  # Cumulative sum

    return df
    
def find_nearest_stop(lat, lon, stops_df):
    """Find the nearest stop based on latitude and longitude."""
    stops_df['distance'] = stops_df.apply(lambda row: geodesic((lat, lon), (row['stop_lat'], row['stop_lon'])).meters, axis=1)
    nearest_stop = stops_df.loc[stops_df['distance'].idxmin()]
    return nearest_stop['stop_id']
    
def compute_range_tracking_lane(distances, time_gaps,end_id_lists,shapeids,speeds,bus_range, charging_power,dynamic_wireless_charging_power, energy_usage, min_stoppage_time, top_end_stop_ids, wireless_track_shapeids,wireless_track_shape,shapes):
    range_tracking = [bus_range]  # Initialize list with Bus_range
    current_range = bus_range  # Initialize range tracking variable
    
    for i in range(len(distances)):
        # Decrease range by traveled distance
        current_range -= distances[i]
        if shapeids[i] in wireless_track_shapeids:
            shape_dummy=shapes[shapes["shape_id"] == shapeids[i]].copy()
            mask = shape_dummy[
                shape_dummy.apply(
                    lambda row: (row["shape_pt_lat"] in wireless_track_shape["shape_pt_lat"].values) and 
                    (row["shape_pt_lon"] in wireless_track_shape["shape_pt_lon"].values),
                    axis=1
                )
            ]

            shape_track_length=0
            for j in range(1, len(mask)):  
                if mask.iloc[j]["shape_pt_sequence"] - mask.iloc[j - 1]["shape_pt_sequence"] == 1:
                    shape_track_length +=(mask.iloc[j]["shape_dist_traveled"] - mask.iloc[j - 1]["shape_dist_traveled"])

            shape_track_length/=1609

            charge_added = float((((shape_track_length/speeds[i])*60)*dynamic_wireless_charging_power)/energy_usage)
        else:
            charge_added =0
        current_range = min(bus_range, current_range + charge_added)
        range_tracking.append(current_range)
        
        # Increase range based on charging time
        if i < len(time_gaps):  # Ensure time_gaps is available
            if time_gaps[i]>min_stoppage_time and end_id_lists[i] in top_end_stop_ids:
                charge_added = (charging_power * time_gaps[i]) / energy_usage
            else:
                charge_added =0
            current_range = min(bus_range, current_range + charge_added)
            range_tracking.append(current_range)
    
    return range_tracking

def compute_group_counts(row):
    shape_counts = row['shape_counts']
    return {tuple(group): sum(shape_counts.get(shape, 0) for shape in group) for group in row['shape_id_groups']}

def extract_shape_counts(row):
    shape_counts = {}
    for group, count in row.items():
        for shape in group:
            shape_counts[shape] = shape_counts.get(shape, 0) + count
    return shape_counts

def find_best_matching_segment(shapes, target_shape_id, input_distance, filtered_blocks,wireless_track_shape):
    target_shape = shapes[shapes["shape_id"] == target_shape_id].copy()
    target_shape = target_shape.rename(columns={"shape_pt_sequence": "target_shape_pt_sequence",
                                                "shape_dist_traveled": "target_shape_dist_traveled"})

    if not wireless_track_shape.empty:
        mask = target_shape[["shape_pt_lat", "shape_pt_lon"]].apply(
            lambda row: (row["shape_pt_lat"] in wireless_track_shape["shape_pt_lat"].values) and 
                    (row["shape_pt_lon"] in wireless_track_shape["shape_pt_lon"].values), axis=1
        )
        # Remove matching rows
        target_shape = target_shape[~mask].reset_index(drop=True)
    
    if input_distance>target_shape["target_shape_dist_traveled"].max():
        input_distance=target_shape["target_shape_pt_sequence"].max()
    
    indices=np.linspace(0, target_shape["target_shape_pt_sequence"].max(), 21)  
    start_indices=indices[:-1].astype(int)

    # Generate sub-segments starting from these selected points
    segments = []
    segment_distances=[]
    for start_idx in start_indices:
        for end_idx in range(start_idx + 1, len(target_shape)):
            segment = target_shape.iloc[start_idx:end_idx + 1]
            total_distance = segment["target_shape_dist_traveled"].iloc[-1] - segment["target_shape_dist_traveled"].iloc[0]

            if total_distance-input_distance>0:
                segments.append(segment)
                segment_distances.append(total_distance)
                break  # Stop early to avoid unnecessary longer segments

    
    if not segments:
        return None, []  # No valid segment found
    
    # Loop through each other shape_id to measure overlap
    max_overlap = 0
    best_segment = None
    best_overlapping_shapes = []
    added_distance=0

    for i in range(len(segments)):
        segment=segments[i]
        segment_overlap = 0
        overlapping_shapes = []  # Stores shape IDs that overlap with this segment

        for shape_id in set(shapes["shape_id"]):
            overlap_shape = shapes[shapes["shape_id"] == shape_id].rename(
                columns={"shape_pt_sequence": "overlap_shape_pt_sequence",
                         "shape_dist_traveled": "overlap_shape_dist_traveled"}
            )

            # Find matching points in overlap shape
            overlap = pd.merge(segment[["shape_pt_lat", "shape_pt_lon", "target_shape_pt_sequence"]],
                               overlap_shape[["shape_pt_lat", "shape_pt_lon", "overlap_shape_pt_sequence"]],
                               on=["shape_pt_lat", "shape_pt_lon"],
                               how="inner")

            # Sort and keep only consecutive points
            overlap = overlap.sort_values(by="target_shape_pt_sequence").reset_index(drop=True)

            if len(overlap) > 1:
                mask = (overlap["target_shape_pt_sequence"].diff() == 1) & (overlap["overlap_shape_pt_sequence"].diff() == 1)
                overlap = overlap[mask].reset_index(drop=True)

            # If there is overlap, add shape_id to list
            if not overlap.empty:
                overlapping_shapes.append(shape_id)
                segment_overlap += len(overlap)  # Count overlapping points

        # Choose the segment with max overlap
        if segment_overlap > max_overlap:
            max_overlap = segment_overlap
            best_segment = segment
            added_distance=segment_distances[i]
            best_overlapping_shapes = overlapping_shapes  # Store the best segment's overlapping shape IDs

    return best_segment, best_overlapping_shapes,added_distance



def main():
    # Set page config for a cleaner interface
    st.set_page_config(
        page_title="Bus Electrification Analysis",
        page_icon="🚌",
        layout="wide"
    )
    
    # List of allowed agency zip files
    agencies = ["BATA (Traverse City)", "CATA (Lansing)", "DDOT (Detroit)", "MAX (Holland)", "Smart (Detroit)", "The Rapid (Grand Rapids)","TheRide (Ann Arbor-Ypsilanti)", "UMich"]
    
    # Application UI
    st.title("🚌 Bus System Electrification Analysis")
    st.markdown("*Internal Testing Release*")
    
    # Create sidebar for settings
    with st.sidebar:
        st.header("Configuration")
        
        # Allow user to select agency
        selected_agency = st.selectbox("Select the agency", agencies)
        
        # Energy and range parameters
        #st.subheader("Electric Bus Parameters")
        bus_range = st.number_input("Electric bus range (miles)", min_value=5, value=150, step=10)
        charging_power = st.number_input("Stationary Charging power (kW)", min_value=0, value=250, step=50)
        dynamic_wireless_charging_power = st.number_input("Dynamic Charging power (kW)", min_value=0, value=0, step=10)

        # Advanced parameters with expander to keep interface clean
        with st.expander("Advanced Parameters"):
            min_stoppage_time = st.number_input("Stationary charging setup time (min)", min_value=0, value=0, step=1)
            energy_usage = st.number_input("Bus energy usage (kWmin/mile)", min_value=5, value=150, step=10)
            #critical_range = st.number_input("Critical range threshold (miles)", min_value=5, value=20, step=5)
            
        critical_range=20
        
        # Run analysis button
        analyze_button = st.button("Run Analysis", use_container_width=True)
    
    # Main content
    if analyze_button:
        st.session_state.clear()
        msg1 = st.empty()
        msg2 = st.empty()
        msg3 = st.empty()
        with st.spinner("Processing GTFS data and analyzing bus routes..."):
            # Load data
            zip_file_path = f"{selected_agency}_GTFS.zip"
            
            try:
                # Load GTFS data
                dataframes = load_gtfs_data(zip_file_path)
                
                # Extract dataframes
                stops = dataframes.get('stops')
                trips = dataframes.get('trips')
                routes = dataframes.get('routes')
                shapes = dataframes.get('shapes')
                stop_times = dataframes.get('stop_times')
                calendar = dataframes.get('calendar', None)
                
                if any(df is None or df.empty for df in [stops, trips, routes, shapes, stop_times]):
                    st.error("Missing required GTFS files. Please check the ZIP file.")
                    return

                missing_rows = shapes.loc[pd.isna(shapes["shape_dist_traveled"])]
                if not missing_rows.empty:
                    shapes = compute_shape_distances(shapes)
                else:
                    # Select a sample shape_id
                    sample_shape_id = shapes['shape_id'].iloc[0]  # Pick the first shape_id
                    # Get the first two points of that shape_id
                    sample_shape = shapes[shapes['shape_id'] == sample_shape_id].sort_values(by='shape_pt_sequence')
                    # Create new columns for reported and computed distances
                    sample_shape['reported_distance'] = None
                    sample_shape['computed_distance_meter'] = None
                    sample_shape['shape_dist_flag'] = None
                    for i in range(len(sample_shape) - 1):
                        reported_distance = sample_shape.iloc[i + 1]['shape_dist_traveled'] - sample_shape.iloc[i]['shape_dist_traveled']
                        # Extract coordinates
                        lat1, lon1 = sample_shape.iloc[i]['shape_pt_lat'], sample_shape.iloc[i]['shape_pt_lon']
                        lat2, lon2 = sample_shape.iloc[i + 1]['shape_pt_lat'], sample_shape.iloc[i + 1]['shape_pt_lon']  
                        # Compute geodesic distance (meters)
                        computed_distance_meter = geodesic((lat1, lon1), (lat2, lon2)).meters
                
                        sample_shape.loc[sample_shape.index[i], 'reported_distance'] = reported_distance
                        sample_shape.loc[sample_shape.index[i], 'computed_distance_meter'] = computed_distance_meter
                
                        if abs(computed_distance_meter - reported_distance) < 0.5:  # Small threshold for rounding errors
                            sample_shape.loc[sample_shape.index[i], 'shape_dist_flag'] = 1
                        elif abs(computed_distance_meter*3.281 - reported_distance) < 0.5:
                            sample_shape.loc[sample_shape.index[i], 'shape_dist_flag'] = 2
                        elif abs(computed_distance_meter/1000 - reported_distance) < 0.5:
                            sample_shape.loc[sample_shape.index[i], 'shape_dist_flag'] = 3
                        elif abs(computed_distance_meter/1609 - reported_distance) < 0.5:
                            sample_shape.loc[sample_shape.index[i], 'shape_dist_flag'] = 4
                    shape_dist_flag=sample_shape['shape_dist_flag'].mode().iloc[0] #shape_dist_flag=1 #[1: meter, 2:feet, 3:km, 4: mile]
                    if shape_dist_flag==2:
                        shapes["shape_dist_traveled"]/=3.281
                    elif shape_dist_flag==3:
                        shapes["shape_dist_traveled"]*=1000
                    elif shape_dist_flag==4:
                        shapes["shape_dist_traveled"]*=1609
                
                # Clean and prepare data
                stop_times = stop_times.sort_values(by=['trip_id', 'stop_sequence']).reset_index(drop=True)
                stop_times['stop_sequence'] = stop_times.groupby('trip_id').cumcount() + 1
                shapes = shapes.sort_values(by=['shape_id', 'shape_pt_sequence']).reset_index(drop=True)
                shapes['shape_pt_sequence'] = shapes.groupby('shape_id').cumcount() + 1

                shape_route = trips[['route_id', 'shape_id']].drop_duplicates()
                shape_route = shape_route.merge(routes[['route_id','route_type','route_color']], on='route_id', how='left')

                
                # Get service ID for weekdays
                if calendar is not None:
                    weekday_services = calendar[
                        (calendar["monday"] == 1) &
                        (calendar["tuesday"] == 1) &
                        (calendar["wednesday"] == 1) &
                        (calendar["thursday"] == 1)
                    ]["service_id"]
                    
                    if not weekday_services.empty:
                        weekday_service_id = weekday_services.iloc[0]  # Take the first one
                    else:
                        weekday_service_id = trips['service_id'].value_counts().index[0]
                else:
                    # If no calendar.txt, use a default service_id
                    weekday_service_id = trips['service_id'].value_counts().index[0]
                
                # Filter trips for weekday service
                weekday_trips = trips[trips['service_id'] == weekday_service_id]
                
                # Calculate shape distances
                shape_distances = shapes.groupby('shape_id').last()[['shape_dist_traveled']]
                shape_distances.columns = ['shape_distance_meters']  # Rename the column to match the expected name     
                shape_distances['shape_distance_km'] = shape_distances['shape_distance_meters'] / 1000
                shape_distances['shape_distance_miles'] = shape_distances['shape_distance_km'] * 0.621371
                
                # Process trip times
                stop_times['departure_time'] = stop_times['departure_time'].astype(str)
                stop_times['departure_time'] = stop_times['departure_time'].apply(convert_to_datetime_over_24)
                
                # Get trip start and end times
                trip_start_times = stop_times.groupby('trip_id')['departure_time'].min().reset_index()
                trip_start_times.rename(columns={'departure_time': 'trip_start_time'}, inplace=True)
                
                trip_end_times = stop_times.groupby('trip_id')['departure_time'].max().reset_index()
                trip_end_times.rename(columns={'departure_time': 'trip_end_time'}, inplace=True)
                
                # Combine trip times
                trip_times = pd.merge(trip_start_times, trip_end_times, on='trip_id')
                
                # Merge with trips
                trips_with_times = trips.merge(trip_times, on='trip_id', how='left')
                weekday_trips = trips_with_times[trips_with_times['service_id'] == weekday_service_id].copy()
                
                # Merge with distances
                trip_distances = weekday_trips.merge(shape_distances, on='shape_id', how='left')
                trip_distances = trip_distances.merge(routes[['route_id', 'route_short_name']], on='route_id', how='left')
                weekday_trips = weekday_trips.merge(trip_distances[["trip_id", "shape_distance_miles"]], on="trip_id", how="left")
                
                # Get start and end points
                start_points = shapes[shapes["shape_pt_sequence"] == 1][["shape_id", "shape_pt_lat", "shape_pt_lon"]]
                start_points = start_points.rename(columns={"shape_pt_lat": "start_lat", "shape_pt_lon": "start_lon"})

                start_points = start_points.merge(
                    stops[['stop_id', 'stop_lat', 'stop_lon']],
                    left_on=['start_lat', 'start_lon'],
                    right_on=['stop_lat', 'stop_lon'],
                    how='left'
                ).rename(columns={'stop_id': 'start_stop_id'})
                
                # Find closest stop if no exact match
                missing_stops = start_points['start_stop_id'].isna()
                start_points.loc[missing_stops, 'start_stop_id'] = start_points[missing_stops].apply(
                    lambda row: find_nearest_stop(row['start_lat'], row['start_lon'], stops) if pd.isna(row['start_stop_id']) else row['start_stop_id'], axis=1
                )
                
                # # Drop redundant stop_lat and stop_lon
                start_points.drop(columns=['stop_lat', 'stop_lon'], inplace=True)
                
                end_points = shapes.loc[shapes.groupby("shape_id")["shape_pt_sequence"].idxmax(), ["shape_id", "shape_pt_lat", "shape_pt_lon"]]
                end_points = end_points.rename(columns={"shape_pt_lat": "end_lat", "shape_pt_lon": "end_lon"})

                end_points = end_points.merge(
                    stops[['stop_id', 'stop_lat', 'stop_lon']],
                    left_on=['end_lat', 'end_lon'],
                    right_on=['stop_lat', 'stop_lon'],
                    how='left'
                ).rename(columns={'stop_id': 'end_stop_id'})
                
                # Find closest stop if no exact match
                missing_stops = end_points['end_stop_id'].isna()
                end_points.loc[missing_stops, 'end_stop_id'] = end_points[missing_stops].apply(
                    lambda row: find_nearest_stop(row['end_lat'], row['end_lon'], stops) if pd.isna(row['end_stop_id']) else row['end_stop_id'], axis=1
                )
                
                # Drop redundant stop_lat and stop_lon
                end_points.drop(columns=['stop_lat', 'stop_lon'], inplace=True)
                
                # Merge points with trips
                weekday_trips = weekday_trips.merge(start_points, on="shape_id", how="left")
                weekday_trips = weekday_trips.merge(end_points, on="shape_id", how="left")
                
                # Sort and calculate time gaps
                weekday_trips = weekday_trips.sort_values(by=['block_id', 'trip_start_time'])
                weekday_trips['next_trip_start_time'] = weekday_trips.groupby('block_id')['trip_start_time'].shift(-1)
                weekday_trips['time_gap'] = (weekday_trips['next_trip_start_time'] - weekday_trips['trip_end_time']).dt.total_seconds() / 60
                weekday_trips["trip_speed"] = weekday_trips['shape_distance_miles']/((weekday_trips['trip_end_time'] - weekday_trips['trip_start_time']).dt.total_seconds() / 3600) #mile per hour
                
                # # Get start and end stop IDs
                # weekday_trips = weekday_trips.merge(
                #     stops[['stop_id', 'stop_lat', 'stop_lon']],
                #     left_on=['start_lat', 'start_lon'],
                #     right_on=['stop_lat', 'stop_lon'],
                #     how='left'
                # ).rename(columns={'stop_id': 'start_stop_id'})
                
                # weekday_trips.drop(columns=['stop_lat', 'stop_lon'], inplace=True)
                
                # weekday_trips = weekday_trips.merge(
                #     stops[['stop_id', 'stop_lat', 'stop_lon']],
                #     left_on=['end_lat', 'end_lon'],
                #     right_on=['stop_lat', 'stop_lon'],
                #     how='left'
                # ).rename(columns={'stop_id': 'end_stop_id'})
                
                # weekday_trips.drop(columns=['stop_lat', 'stop_lon'], inplace=True)

                
            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
                return            
    
        msg1.success("✅ GTFS data processed successfully.")
        with st.spinner("Optimizing stationary charging locations..."):    
            try: 
                
                # Calculate block distances
                block_distances = weekday_trips.groupby('block_id').agg({
                    'shape_distance_miles': 'sum',
                    'route_id': 'unique',
                    'shape_id': 'unique'
                }).reset_index()
                
                block_distances.columns = ['block_id', 'total_distance_miles', 'routes_in_block_id', 'routes_in_block_shapes']
                
                # Group trips by block
                block_trip_routes = weekday_trips.groupby('block_id').agg(
                    trips_by_route=('shape_id', lambda x: list(x)),
                    time_gaps=('time_gap', lambda x: list(x)),
                    distances_list=('shape_distance_miles', lambda x: list(x)),
                    avg_speed_list=('trip_speed', lambda x:list(x)),
                    start_lat_list=('start_lat', lambda x: list(x)),
                    end_lat_list=('end_lat', lambda x: list(x)),
                    start_lon_list=('start_lon', lambda x: list(x)),
                    end_lon_list=('end_lon', lambda x: list(x)),
                    start_id_list=('start_stop_id', lambda x: list(x)),
                    end_id_list=('end_stop_id', lambda x: list(x))
                ).reset_index()
                
                # Merge block info
                block_general = pd.merge(block_distances, block_trip_routes, on='block_id', how='outer')
                block_general['time_gaps_sum'] = block_general['time_gaps'].apply(lambda x: np.nansum(x) if isinstance(x, list) else np.nan)
                block_general["time_gaps"] = block_general["time_gaps"].apply(lambda lst: [x for x in lst if not pd.isna(x)])
                block_general = block_general.sort_values(by=['total_distance_miles', 'time_gaps_sum'])
                
                # Initialize charging location selection
                top_end_stop_ids = []
                
                # Apply function to calculate range tracking
                block_general["range_tracking"] = block_general.apply(
                    lambda row: compute_range_tracking(
                        row["distances_list"], 
                        row["time_gaps"], 
                        row["end_id_list"],
                        bus_range,
                        charging_power,
                        energy_usage,
                        min_stoppage_time,
                        top_end_stop_ids
                    ),
                    axis=1
                )
                
                min_range_without_charging = int(np.ceil(block_general["total_distance_miles"].max()))
                
                # Identify infeasible blocks
                infeasible_blocks = block_general[block_general["range_tracking"].apply(lambda rt: any(x < 0 for x in rt) if rt else False)]["block_id"].tolist()
                blocks_below_critical = block_general[block_general["range_tracking"].apply(lambda rt: any(x < critical_range for x in rt) if rt else False)]["block_id"].tolist()
            


                # Iteratively select charging locations
                iteration_count = 0
                
                while infeasible_blocks and iteration_count < stops['stop_id'].nunique(): 
                    previous_infeasible_count = len(infeasible_blocks)
                    
                    # Filter blocks
                    filtered_blocks = block_general[block_general["block_id"].isin(infeasible_blocks)].copy()
                    
                    # Process end stops
                    filtered_blocks["end_id_list_trimmed"] = filtered_blocks["end_id_list"].apply(lambda x: x[:-1])
                    filtered_blocks["valid_end_ids"] = filtered_blocks.apply(
                        lambda row: [end_id for end_id, gap in zip(row["end_id_list_trimmed"], row["time_gaps"]) 
                                   if gap > min_stoppage_time and pd.notna(end_id)], 
                        axis=1
                    )
                    
                    # Count end stop frequencies
                    all_end_ids = [end_id for sublist in filtered_blocks["valid_end_ids"] for end_id in sublist if pd.notna(end_id)]
                    end_id_counts = Counter(all_end_ids)
                    
                    # Find missing IDs
                    missing_ids_per_row = filtered_blocks["valid_end_ids"].apply(lambda row: [eid for eid in row if eid not in top_end_stop_ids and pd.notna(eid)])
                    missing_id_counts = Counter([eid for row in missing_ids_per_row for eid in row if pd.notna(eid)])
                    
                    # Sort by frequency
                    sorted_missing_ids = sorted(missing_id_counts.keys(), key=lambda x: missing_id_counts[x], reverse=True)
                    
                    # Select new charging locations
                    added_ids = set()
                    
                    for idx, row in enumerate(missing_ids_per_row):
                        if not any(eid in top_end_stop_ids or eid in added_ids for eid in row if pd.notna(eid)):
                            for new_id in sorted_missing_ids:
                                if new_id in row:
                                    added_ids.add(new_id)
                                    break
                    
                    # Update charging locations
                    top_end_stop_ids = list(set(top_end_stop_ids).union(added_ids))
                    
                    # Recalculate range tracking
                    block_general["range_tracking"] = block_general.apply(
                        lambda row: compute_range_tracking(
                            row["distances_list"], 
                            row["time_gaps"], 
                            row["end_id_list"],
                            bus_range,
                            charging_power,
                            energy_usage,
                            min_stoppage_time,
                            top_end_stop_ids
                        ),
                        axis=1
                    )
                    
                    # Update infeasible blocks
                    infeasible_blocks = block_general[block_general["range_tracking"].apply(lambda rt: any(x < 0 for x in rt) if rt else False)]["block_id"].tolist()
                    blocks_below_critical = block_general[block_general["range_tracking"].apply(lambda rt: any(x < critical_range for x in rt) if rt else False)]["block_id"].tolist()
                    
                    # Break if no improvement
                    if len(infeasible_blocks) >= previous_infeasible_count and len(infeasible_blocks) > 0:
                        break
                    
                    iteration_count += 1

            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
                return
        
        msg2.success("✅ Stationary charging locations are optimized.")
        with st.spinner("Optimizing dynamic track locations..."):  
            try:
                
                wireless_track_shapeids=set()
                wireless_track_length=0
                wireless_track_shape = pd.DataFrame()
                
                if dynamic_wireless_charging_power>0:
                    while len(infeasible_blocks)>0:
                        # Filter block_general to keep only rows where block_id is in infeasible_blocks
                        filtered_blocks = block_general[block_general["block_id"].isin(infeasible_blocks)].copy()
                        
                        filtered_blocks["estimate_required_length"]=-(
                            filtered_blocks["range_tracking"].apply(min) * energy_usage * filtered_blocks["avg_speed_list"].apply(max)
                        ) / (dynamic_wireless_charging_power * 60)
                
                        # Count occurrences
                        filtered_blocks["shape_counts"] = filtered_blocks["trips_by_route"].apply(Counter) 
                        filtered_blocks["shape_id_groups"] = None  # Initialize column
                        filtered_blocks = filtered_blocks.astype({"shape_id_groups": "object"})  # Force dtype to object
                
                        for index, ids in filtered_blocks["routes_in_block_shapes"].items():
                            groups=[]
                            for shape_id in ids:
                                direction_id = trips.loc[trips["shape_id"] == shape_id, "direction_id"].iloc[0]
                                route_id = trips.loc[trips["shape_id"] == shape_id, "route_id"].iloc[0]
                                found_group = False
                                for group in groups:
                                    # Get the first shape_id in the group
                                    first_shape_id = group[0]
                                    # Get its direction_id and route_id
                                    first_direction_id = trips.loc[trips["shape_id"] == first_shape_id, "direction_id"].iloc[0]
                                    first_route_id = trips.loc[trips["shape_id"] == first_shape_id, "route_id"].iloc[0]
                            
                                    # If both direction_id and route_id match, add the shape_id to this group
                                    if direction_id == first_direction_id and route_id == first_route_id:
                                        group.append(shape_id)
                                        found_group = True
                                        break
                        
                                # If no group was found, create a new group for this shape_id
                                if not found_group:
                                    groups.append([shape_id])       
                            filtered_blocks.at[index, "shape_id_groups"] = groups
                        
                        filtered_blocks['group_counts'] = filtered_blocks.apply(compute_group_counts, axis=1)
                
                        filtered_blocks['flattened_counts'] = filtered_blocks['group_counts'].apply(extract_shape_counts)
                
                
                        # Step 2: Find common shape IDs across all rows
                        common_shapes = set(filtered_blocks['flattened_counts'].iloc[0].keys()-wireless_track_shapeids)
                        
                
                        for i in range(1, len(filtered_blocks)):
                            common_shapes.intersection_update(filtered_blocks['flattened_counts'].iloc[i].keys()-wireless_track_shapeids)
                
                
                        # Step 3: If common shape IDs exist, find the one with the highest count sum
                        if common_shapes:
                            best_shape = max(common_shapes, key=lambda shape: sum(filtered_blocks['flattened_counts'].iloc[i][shape] for i in range(len(filtered_blocks))))
                            track_shape_id = {best_shape}
                        else:
                            # If no common shape ID, return the highest count shape from each row
                            highest_shapes = {max(row, key=row.get) for row in filtered_blocks['flattened_counts']}
                            track_shape_id = {next(iter(highest_shapes))}
                         
                   
                
                        # Compute the sum of counts for the selected shape IDs in each row
                        filtered_blocks['track_shape_count'] = filtered_blocks['flattened_counts'].apply(lambda row: sum(row[shape] for shape in track_shape_id if shape in row))
                
                        filtered_blocks['estimate_length-per_shape']=filtered_blocks["estimate_required_length"]/filtered_blocks["track_shape_count"]
                
                        
                
                        new_track_shape, new_track_shapeids,new_distance=find_best_matching_segment(shapes, list(track_shape_id)[0], max(filtered_blocks.loc[filtered_blocks['track_shape_count']>0,'estimate_length-per_shape'])*1609, filtered_blocks,wireless_track_shape)
                        
                        wireless_track_length= wireless_track_length+new_distance/1609
                        wireless_track_shape = pd.concat([wireless_track_shape, new_track_shape], ignore_index=True)
                        wireless_track_shapeids.update(new_track_shapeids)
 
                
                        if len(infeasible_blocks)>0:   
                            filtered_blocks["range_tracking"] = filtered_blocks.apply(
                                lambda row: compute_range_tracking_lane(row["distances_list"], row["time_gaps"], row["end_id_list"], row["trips_by_route"], row["avg_speed_list"],bus_range, charging_power,dynamic_wireless_charging_power, energy_usage, min_stoppage_time, top_end_stop_ids, wireless_track_shapeids,wireless_track_shape,shapes),
                                axis=1
                            )
                            infeasible_blocks = filtered_blocks[filtered_blocks["range_tracking"].apply(lambda rt: any(x < 0 for x in rt) if rt else False)]["block_id"].tolist()
                        
                        if new_distance==0:
                            break

            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
                return
        
        msg3.success("✅ Dynamic track locations are optimized.")
        with st.spinner("Checking for further improvments & preparing results..."):  
            try:
                
                for id in top_end_stop_ids[:]:  # Iterate over a copy
                    top_end_stop_ids.remove(id)  # Remove safely

                    block_general["range_tracking"] = block_general.apply(
                        lambda row: compute_range_tracking_lane(row["distances_list"], row["time_gaps"], row["end_id_list"], row["trips_by_route"], row["avg_speed_list"],bus_range, charging_power,dynamic_wireless_charging_power, energy_usage, min_stoppage_time, top_end_stop_ids, wireless_track_shapeids,wireless_track_shape,shapes),
                        axis=1
                    )
                    infeasible_blocks_copy = block_general[block_general["range_tracking"].apply(lambda rt: any(x < 0 for x in rt) if rt else False)]["block_id"].tolist()
                    
                    if len(infeasible_blocks_copy) > len(infeasible_blocks):
                        top_end_stop_ids.append(id)  # Add back if necessary

                # Apply function to blocks with total_distance_miles > Bus_range
                block_general["range_tracking"] = block_general.apply(
                    lambda row: compute_range_tracking_lane(row["distances_list"], row["time_gaps"], row["end_id_list"], row["trips_by_route"], row["avg_speed_list"],bus_range, charging_power,dynamic_wireless_charging_power, energy_usage, min_stoppage_time, top_end_stop_ids, wireless_track_shapeids,wireless_track_shape,shapes),
                    axis=1
                )

                
                # Identify infeasible block_ids where any range_tracking value is negative
                infeasible_blocks = block_general[block_general["range_tracking"].apply(lambda rt: any(x < 0 for x in rt) if rt else False)]["block_id"].tolist()

                # Identify block_ids where any range_tracking value is below critical_range
                blocks_below_critical = block_general[block_general["range_tracking"].apply(lambda rt: any(x < critical_range for x in rt) if rt else False)]["block_id"].tolist() 

                                
                # Get proposed charging locations
                selected_stops = pd.DataFrame(top_end_stop_ids, columns=["stop_id"])
                proposed_locations = selected_stops.merge(stops[["stop_id", "stop_lat", "stop_lon"]], on="stop_id", how="left")
                proposed_locations = proposed_locations.drop_duplicates().reset_index(drop=True)
                
                # Calculate map center
                center_lat = shapes['shape_pt_lat'].mean()
                center_lon = shapes['shape_pt_lon'].mean()
                
                # Create map
                bus_map = create_bus_electrification_map(
                    shapes,
                    routes,
                    trips,
                    proposed_locations,
                    wireless_track_shape,
                    center_lat,
                    center_lon
                )
                
                # Store results
                st.session_state["map"] = bus_map
                st.session_state["routes_count"] = len(trips['route_id'].unique())
                st.session_state["stops_count"] = stops['stop_id'].nunique()
                st.session_state["blocks_count"] = len(block_general)
                st.session_state["infeasible_blocks_count"] = len(infeasible_blocks)
                st.session_state["critical_blocks_count"] = len(blocks_below_critical)
                st.session_state["minimum_range_without_charger"] = min_range_without_charging
                st.session_state["num_locs"] = len(proposed_locations)
                st.session_state["wirelesslength"]= round(wireless_track_length,1)

                msg1.empty()
                msg2.empty()
                msg3.empty()
                
            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
                return
    
    # Display results if available
    if "map" in st.session_state:
        # Create metrics display
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Routes", st.session_state['routes_count'])
            
        with col2:
            st.metric("Total Stops", st.session_state['stops_count'])

        with col3:
            st.metric("Total Blocks", st.session_state['blocks_count'])
            #st.metric("Infeasible Blocks", st.session_state['infeasible_blocks_count'])
            #st.metric("Proposed Charging Locations", st.session_state['num_locs'])
        
        # Display additional information
        st.subheader("Analysis Results")
        st.write(f"With the selected configurations:")
        if st.session_state['infeasible_blocks_count']==0:
            st.success("✅ All blocks can be electrified.")
        else:
            if st.session_state['infeasible_blocks_count'] ==1:
                st.success(f"✅ {st.session_state['blocks_count']-st.session_state['infeasible_blocks_count']} blocks can be electrified.")
                st.error(f"❌ {st.session_state['infeasible_blocks_count']} block cannot be electrified.")
            else:
                st.success(f"✅ {st.session_state['blocks_count']-st.session_state['infeasible_blocks_count']} blocks can be electrified.")
                st.error(f"❌ {st.session_state['infeasible_blocks_count']} blocks cannot be electrified.")
            
        if st.session_state['num_locs'] >1: 
            st.write(f"- {st.session_state['num_locs']} stationary charging locations are needed.")
        elif st.session_state['num_locs'] ==1: 
            st.write(f"- {st.session_state['num_locs']} stationary charging location is needed.")
        else:
            st.write(f"- No stationary charging location is needed.")
            
        st.write(f"- The total length of the dynamic wireless track is {st.session_state['wirelesslength']} miles.")
        
        # Display map
        st.subheader("Route Map with Proposed Charging Locations")
        st_folium(st.session_state["map"], width=800, height=600, returned_objects=[])

if __name__ == "__main__":
    main()
