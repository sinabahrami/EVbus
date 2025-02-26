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

def create_bus_electrification_map(shapes_df, routes_df, trips_df, proposed_locations_df, center_lat, center_lon):
    """Create a folium map with routes and charging locations."""
    # Create the map with a specific location and zoom level
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
    
    # Add OpenStreetMap tile layer
    folium.TileLayer(
        tiles='https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
        attr='OpenStreetMap',
        name='OpenStreetMap',
        overlay=False,
        opacity=0.6,
        control=True
    ).add_to(m)
    
    # Create a feature group for all routes
    all_routes = folium.FeatureGroup(name="All Routes", show=True)
    
    # Get unique route IDs
    unique_routes = trips_df['route_id'].unique()
    num_routes = len(unique_routes)
    
    # Generate colors for routes using a colormap
    colormap = plt.cm.get_cmap('tab20b', num_routes)
    colors = [mcolors.to_hex(colormap(i)) for i in range(num_routes)]
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
                    opacity=1,
                    tooltip=f"Route {route_id} - Shape {shape_id}"
                ).add_to(route_group)
                
                # Also add to the "All Routes" group
                folium.PolyLine(
                    coordinates,
                    color=route_color,
                    weight=4,
                    opacity=1
                ).add_to(all_routes)
        
        # Add the route group to the map
        route_group.add_to(m)
    
    # Add the "All Routes" group to the map
    all_routes.add_to(m)
    
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

def main():
    # Set page config for a cleaner interface
    st.set_page_config(
        page_title="Bus Electrification Analysis",
        page_icon="🚌",
        layout="wide"
    )
    
    # List of allowed agency zip files
    agencies = ["TheRide (Ann Arbor-Ypsilanti)", "BATA", "Detroit Department of Transportation", "JATA", "MAX", "Smart", "UMich"]
    
    # Application UI
    st.title("🚌 Bus System Electrification Analysis")
    st.markdown("*Internal Testing Release*")
    
    # Create sidebar for settings
    with st.sidebar:
        st.header("Configuration")
        
        # Allow user to select agency
        selected_agency = st.selectbox("Select the agency", agencies)
        
        # Energy and range parameters
        st.subheader("Electric Bus Parameters")
        bus_range = st.number_input("Bus range (miles)", min_value=0, value=150, step=10)
        charging_power = st.number_input("Charging power (kW)", min_value=0, value=250, step=50)
        
        # Advanced parameters with expander to keep interface clean
        with st.expander("Advanced Parameters"):
            energy_usage = st.number_input("Energy usage (kWmin/mile)", min_value=50, value=150, step=10)
            critical_range = st.number_input("Critical range threshold (miles)", min_value=5, value=20, step=5)
            min_stoppage_time = st.number_input("Minimum stoppage time for charging (min)", min_value=0, value=5, step=1)
        
        # Run analysis button
        analyze_button = st.button("Run Analysis", use_container_width=True)
    
    # Main content
    if analyze_button:
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
                
                # Clean and prepare data
                stop_times = stop_times.sort_values(by=['trip_id', 'stop_sequence']).reset_index(drop=True)
                stop_times['stop_sequence'] = stop_times.groupby('trip_id').cumcount() + 1
                shapes = shapes.sort_values(by=['shape_id', 'shape_pt_sequence']).reset_index(drop=True)
                shapes['shape_pt_sequence'] = shapes.groupby('shape_id').cumcount() + 1
                
                # Get service ID for weekdays
                if calendar is not None:
                    weekday_service_id = calendar[
                        (calendar["monday"] == 1) &
                        (calendar["tuesday"] == 1) &
                        (calendar["wednesday"] == 1) &
                        (calendar["thursday"] == 1) #&
                        #(calendar["friday"] == 1)
                    ]["service_id"].iloc[0]
                else:
                    # If no calendar.txt, use a default service_id
                    weekday_service_id = trips['service_id'].value_counts().index[0]
                
                # Filter trips for weekday service
                weekday_trips = trips[trips['service_id'] == weekday_service_id]
                
                # Calculate shape distances
                shape_distances = shapes.groupby('shape_id').last()[['shape_dist_traveled']]
                shape_distances.columns = ['shape_distance_meters']
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
                
                end_points = shapes.loc[shapes.groupby("shape_id")["shape_pt_sequence"].idxmax(), ["shape_id", "shape_pt_lat", "shape_pt_lon"]]
                end_points = end_points.rename(columns={"shape_pt_lat": "end_lat", "shape_pt_lon": "end_lon"})
                
                # Merge points with trips
                weekday_trips = weekday_trips.merge(start_points, on="shape_id", how="left")
                weekday_trips = weekday_trips.merge(end_points, on="shape_id", how="left")
                
                # Sort and calculate time gaps
                weekday_trips = weekday_trips.sort_values(by=['block_id', 'trip_start_time'])
                weekday_trips['next_trip_start_time'] = weekday_trips.groupby('block_id')['trip_start_time'].shift(-1)
                weekday_trips['time_gap'] = (weekday_trips['next_trip_start_time'] - weekday_trips['trip_end_time']).dt.total_seconds() / 60
                
                # Get start and end stop IDs
                weekday_trips = weekday_trips.merge(
                    stops[['stop_id', 'stop_lat', 'stop_lon']],
                    left_on=['start_lat', 'start_lon'],
                    right_on=['stop_lat', 'stop_lon'],
                    how='left'
                ).rename(columns={'stop_id': 'start_stop_id'})
                
                weekday_trips.drop(columns=['stop_lat', 'stop_lon'], inplace=True)
                
                weekday_trips = weekday_trips.merge(
                    stops[['stop_id', 'stop_lat', 'stop_lon']],
                    left_on=['end_lat', 'end_lon'],
                    right_on=['stop_lat', 'stop_lon'],
                    how='left'
                ).rename(columns={'stop_id': 'end_stop_id'})
                
                weekday_trips.drop(columns=['stop_lat', 'stop_lon'], inplace=True)
                
                # Calculate block distances
                block_distances = weekday_trips.groupby('block_id').agg({
                    'shape_distance_miles': 'sum',
                    'route_id': 'unique',
                    'shape_id': 'unique'
                }).reset_index()
                
                block_distances.columns = ['block_id', 'total_distance_miles', 'routes_in_block_id', 'routes_in_block_shapes']
                
                # Group trips by block
                block_trip_routes = weekday_trips.groupby('block_id').agg(
                    trips_by_route=('route_id', lambda x: list(x)),
                    time_gaps=('time_gap', lambda x: list(x)),
                    distances_list=('shape_distance_miles', lambda x: list(x)),
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
                max_iterations = 50  # Prevent infinite loops
                
                while infeasible_blocks and iteration_count < stops['stop_id'].nunique(): #max_iterations:
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
                    if len(sorted_missing_ids) > 0:
                        added_ids.add(sorted_missing_ids[0])
                    
                    # for idx, row in enumerate(missing_ids_per_row):
                    #     if not any(eid in top_end_stop_ids or eid in added_ids for eid in row if pd.notna(eid)):
                    #         for new_id in sorted_missing_ids:
                    #             if new_id in row:
                    #                 added_ids.add(new_id)
                    #                 break
                    
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
                    if len(infeasible_blocks) == previous_infeasible_count:
                        top_end_stop_ids = list(set(top_end_stop_ids)-added_ids)
                    elif len(infeasible_blocks) >= previous_infeasible_count and len(infeasible_blocks) > 0:
                        break
                    
                    iteration_count += 1
                
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
        st.metric("Min Range Required to cover all blocks without charging", f"{st.session_state['minimum_range_without_charger']} miles")
        st.write(f"With the selected bus range of {bus_range} miles and charging power of {charging_power} kW:")
        
        if st.session_state['infeasible_blocks_count'] > 0:
            st.warning(f"⚠️ {st.session_state['infeasible_blocks_count']} blocks cannot be served with the current configuration.")
        else:
            st.success("✅ All blocks can be served with the current configuration.")
        
        st.write(f"- {st.session_state['critical_blocks_count']} blocks have range dropping below the critical threshold of {critical_range} miles")
        st.write(f"- {st.session_state['num_locs']} charging locations are needed")
        
        # Display map
        st.subheader("Route Map with Proposed Charging Locations")
        st_folium(st.session_state["map"], width=1000, height=600, returned_objects=[])

if __name__ == "__main__":
    main()
