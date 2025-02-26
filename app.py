import streamlit as st
import os
import zipfile
import pandas as pd
from datetime import datetime, timedelta
import folium
from fpdf import FPDF
from PIL import Image
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import folium
from folium import plugins
from geopy.distance import geodesic
from streamlit_folium import st_folium 
from collections import Counter 

def compute_range_tracking(distances, time_gaps,end_id_lists):
    range_tracking = [Bus_range]  # Initialize list with Bus_range
    current_range = Bus_range  # Initialize range tracking variable
    
    for i in range(len(distances)):
        # Decrease range by traveled distance
        current_range -= distances[i]
        range_tracking.append(current_range)

        # Increase range based on charging time
        if i < len(time_gaps):  # Ensure time_gaps is available
            if time_gaps[i]>min_stoppage_time and end_id_lists[i] in top_end_stop_ids:
                charge_added = (charging_power * time_gaps[i]) / energy_usage
            else:
                charge_added =0
            current_range = min(Bus_range, current_range + charge_added)
            range_tracking.append(current_range)
    
    return range_tracking

def convert_to_datetime_over_24(time_str):
    hours, minutes, seconds = map(int, time_str.split(":"))
    if hours >= 24:
        days = hours // 24
        hours = hours % 24  # Convert hours greater than 24 into the correct hour part
        time_str = f"{days} days {hours:02}:{minutes:02}:{seconds:02}"
    return pd.to_timedelta(time_str)

# List of allowed agency zip files
agencies=["The Ride","Detroit Department of Transportation", "Smart"]

# Streamlit UI to select file and input parameters
st.title("**Bus System Electrification Analysis** [Internal Testing Release]")

# Allow user to select a ZIP file from the allowed files
selected_agency = st.selectbox("Select the agency", agencies)

# Request Bus range and charging power inputs
Bus_range = st.number_input("Enter bus range (miles)", min_value=0, value=150, step=10)
charging_power = st.number_input("Enter wireless charging power (kW)", min_value=0, value=250, step=50)

# Define parameters
energy_usage = 150  # kWmin/mile
critical_range = 20  # Define the critical range threshold in miles
min_stoppage_time=0 #min

if st.button("Run Analysis"):
    selected_file = selected_agency+'_GTFS.zip'
    # Open and read the ZIP file
    with zipfile.ZipFile(selected_file, 'r') as zip_ref:
        dataframes = {}
        for file_name in zip_ref.namelist():
            if file_name.endswith('.txt'):
                with zip_ref.open(file_name) as file:
                    df_name = file_name.split(".")[0]
                    dataframes[df_name] = pd.read_csv(file)
                    globals()[df_name] = dataframes[df_name]
    
    stop_times = stop_times.sort_values(by=['trip_id', 'stop_sequence']).reset_index(drop=True)
    # Recount stop_sequence for each trip_id
    stop_times['stop_sequence'] = stop_times.groupby('trip_id').cumcount() + 1
    shapes = shapes.sort_values(by=['shape_id', 'shape_pt_sequence']).reset_index(drop=True)
    shapes['shape_pt_sequence'] = shapes.groupby('shape_id').cumcount() + 1

    shape_route = trips[['route_id', 'shape_id']].drop_duplicates()
    shape_route = shape_route.merge(routes[['route_id','route_type','route_color']], on='route_id', how='left')

    #Total_number_routes = len(routes)
    Num_stop = stops['stop_id'].nunique()

    # Define a mapping for the route_type values to their corresponding transportation modes
    route_type_mapping = {
        0: "Streetcar",
        1: "Subway",
        2: "Rail",
        3: "Bus",
        4: "Ferry",
        5: "Cable Tram",
        6: "Aerial Lift",
        7: "Funicular",
        11: "Trolleybus",
        12: "Monorail"
    }

    route_type_mapping = {
        0: "Streetcar",
        1: "Subway",
        2: "Rail",
        3: "Bus",
        4: "Ferry",
        5: "Cable Tram",
        6: "Aerial Lift",
        7: "Funicular",
        11: "Trolleybus",
        12: "Monorail"
    }
    shape_route['modes'] = shape_route['route_type'].map(route_type_mapping)

    if "calendar" in globals():
        weekday_service_id = calendar[
            (calendar["monday"] == 1) &
            (calendar["tuesday"] == 1) &
            (calendar["wednesday"] == 1) &
            (calendar["thursday"] == 1) &
            (calendar["friday"] == 1)
        ]["service_id"]
        weekday_service_id=weekday_service_id.iloc[0]
    else:
        weekday_service_id =2

    # Filter trips for weekday_service_id
    weekday_trips = trips[trips['service_id'] == weekday_service_id]

    # Extract the last row for each shape_id, which contains the total distance
    shape_distances = shapes.groupby('shape_id').last()[['shape_dist_traveled']]  # Extract the last row for each group
    shape_distances.columns = ['shape_distance_meters']  # Rename the column to match the expected name

    # Convert the distance from meters to kilometers (or miles if you prefer)
    shape_distances['shape_distance_km'] = shape_distances['shape_distance_meters'] / 1000  # Convert to km
    shape_distances['shape_distance_miles'] = shape_distances['shape_distance_km'] * 0.621371  # Convert to miles

    # Merge the routes data with trips to get route_short_name and other necessary columns
    trip_distances = weekday_trips.merge(shape_distances, on='shape_id', how='left')
    trip_distances = trip_distances.merge(routes[['route_id', 'route_short_name']], on='route_id', how='left')

    # Calculate block distances only for weekdays
    block_distances = trip_distances.groupby('block_id').agg({
        'shape_distance_km': 'sum',  # Sum distances for all trips in a block
        'route_id': 'unique',  # Collect unique route IDs (this will create an array, you can change to use only the first if needed)
        'shape_id': 'unique'  # Collect unique shape names (same as above)
    }).reset_index() 

    # If you want to flatten the unique route IDs and names into single values or strings:
    block_distances['routes_in_block_id'] = block_distances['route_id'].apply(lambda x: x[0] if len(x) == 1 else x)  # Use first element if only one, else keep all
    block_distances['routes_in_block_shapes'] = block_distances['shape_id'].apply(lambda x: x[0] if len(x) == 1 else x)  # Same for route names 

    # Convert distances from km to miles
    block_distances['total_distance_miles'] = block_distances['shape_distance_km'] * 0.621371 

    # Rename and reorder columns for clarity
    block_distances = block_distances[['block_id', 'total_distance_miles', 'route_id', 'shape_id']]
    block_distances.columns = ['block_id', 'total_distance_miles', 'routes_in_block_id', 'routes_in_block_shapes']

    # Ensure departure_time is treated as a string
    stop_times['departure_time'] = stop_times['departure_time'].astype(str)

    # Apply function to fix times
    #stop_times['departure_time'] = stop_times['departure_time'].apply(fix_gtfs_time)

    # Get the first departure time for each trip_id
    trip_start_times = stop_times.groupby('trip_id')['departure_time'].min().reset_index()
    trip_start_times.rename(columns={'departure_time': 'trip_start_time'}, inplace=True)

    # Merge with trips dataset
    trips_with_times = trips.merge(trip_start_times, on='trip_id', how='left')

    # Apply the function to the 'departure_time' column
    stop_times['departure_time'] = stop_times['departure_time'].apply(convert_to_datetime_over_24)

    trip_times = stop_times.groupby('trip_id')['departure_time'].min().reset_index()
    trip_times.rename(columns={'departure_time': 'trip_start_time'}, inplace=True)

    # Get the last departure time for each trip_id (latest stop)
    trip_end_times = stop_times.groupby('trip_id')['departure_time'].max().reset_index()
    trip_end_times.rename(columns={'departure_time': 'trip_end_time'}, inplace=True)

    # Merge both start and end times into a single DataFrame
    trip_times = pd.merge(trip_times, trip_end_times, on='trip_id')

    # Merge trip start and end times with trips dataset
    trips_with_times = trips.merge(trip_times, on='trip_id', how='left') 

    # Filter for weekday_service_id 
    weekday_trips = trips_with_times[trips_with_times['service_id'] == weekday_service_id].copy()
    weekday_trips = weekday_trips.merge(trip_distances[["trip_id", "shape_distance_miles"]], on="trip_id", how="left")

    # Find start point (where shape_pt_sequence is 1)
    start_points = shapes[shapes["shape_pt_sequence"] == 1][["shape_id", "shape_pt_lat", "shape_pt_lon"]]
    start_points = start_points.rename(columns={"shape_pt_lat": "start_lat", "shape_pt_lon": "start_lon"})

    # Find end point (where shape_pt_sequence is the max for each shape_id)
    end_points = shapes.loc[shapes.groupby("shape_id")["shape_pt_sequence"].idxmax(), ["shape_id", "shape_pt_lat", "shape_pt_lon"]]
    end_points = end_points.rename(columns={"shape_pt_lat": "end_lat", "shape_pt_lon": "end_lon"})

    # Merge start and end points into weekday_trips
    weekday_trips = weekday_trips.merge(start_points, on="shape_id", how="left")
    weekday_trips = weekday_trips.merge(end_points, on="shape_id", how="left")

    # Sort by block_id and trip start time
    weekday_trips = weekday_trips.sort_values(by=['block_id', 'trip_start_time']) 

    # Calculate the time gap between the current trip's trip_end_time and the next trip's trip_start_time in the same block
    weekday_trips['next_trip_start_time'] = weekday_trips.groupby('block_id')['trip_start_time'].shift(-1)
    weekday_trips['time_gap'] = (weekday_trips['next_trip_start_time'] - weekday_trips['trip_end_time']).dt.total_seconds() / 60  # Convert time gap to minutes

    # Merge for start_stop_id
    weekday_trips = weekday_trips.merge(
        stops[['stop_id', 'stop_lat', 'stop_lon']],
        left_on=['start_lat', 'start_lon'],
        right_on=['stop_lat', 'stop_lon'],
        how='left'
    ).rename(columns={'stop_id': 'start_stop_id'})

    # Drop redundant stop_lat and stop_lon
    weekday_trips.drop(columns=['stop_lat', 'stop_lon'], inplace=True)


    # Merge for end_stop_id
    weekday_trips = weekday_trips.merge(
        stops[['stop_id', 'stop_lat', 'stop_lon']],
        left_on=['end_lat', 'end_lon'],
        right_on=['stop_lat', 'stop_lon'],
        how='left'
    ).rename(columns={'stop_id': 'end_stop_id'})

    # Drop redundant stop_lat and stop_lon
    weekday_trips.drop(columns=['stop_lat', 'stop_lon'], inplace=True)

    block_trip_routes = weekday_trips.groupby('block_id').agg(
        trips_by_route=('route_id', lambda x: list(x)),
        time_gaps=('time_gap', lambda x: list(x)),
        distances_list=('shape_distance_miles', lambda x:list(x)),
        start_lat_list=('start_lat', lambda x: list(x)),  # Store unique latitudes
        end_lat_list=('end_lat', lambda x: list(x)),  # Store unique end latitudes
        start_lon_list=('start_lon', lambda x: list(x)),  # Store unique longitudes
        end_lon_list=('end_lon', lambda x: list(x)),  # Store unique end longitudes
        start_id_list=('start_stop_id', lambda x: list(x)),
        end_id_list=('end_stop_id', lambda x: list(x))
    ).reset_index()

    block_general = pd.merge(block_distances, block_trip_routes, on='block_id', how='outer') 
    # Calculate the sum of the time_gaps list, ignoring NaN values, and add it as a new column
    block_general['time_gaps_sum'] = block_general['time_gaps'].apply(lambda x: np.nansum(x) if isinstance(x, list) else np.nan)
    #del block_general["routes_in_block_name"]
    block_general["time_gaps"] = block_general["time_gaps"].apply(lambda lst: [x for x in lst if not pd.isna(x)])
    block_general=block_general.sort_values(by=['total_distance_miles','time_gaps_sum'])

    
    
    top_end_stop_ids = []

    # Apply function to blocks with total_distance_miles > Bus_range
    block_general["range_tracking"] = block_general.apply(
        lambda row: compute_range_tracking(row["distances_list"], row["time_gaps"], row["end_id_list"]),
        axis=1
    )

    min_range_without_charging =int(np.ceil(block_general["total_distance_miles"].iloc[-1]))

    # Identify infiseable block_ids where any range_tracking value is negative
    infiseable_blocks = block_general[block_general["range_tracking"].apply(lambda rt: any(x < 0 for x in rt) if rt else False)]["block_id"].tolist()

    # Identify block_ids where any range_tracking value is below critical_range
    blocks_below_critical = block_general[block_general["range_tracking"].apply(lambda rt: any(x < critical_range for x in rt) if rt else False)]["block_id"].tolist()

    flag=len(infiseable_blocks)+1
    while len(infiseable_blocks)>0 and flag>len(infiseable_blocks):
        flag=len(infiseable_blocks)
        # Filter block_general to keep only rows where block_id is in infiseable_blocks
        filtered_blocks = block_general[block_general["block_id"].isin(infiseable_blocks)].copy()

        # Ensure end_id_list and time_gaps have the same length by removing the last element from end_id_list
        filtered_blocks["end_id_list_trimmed"] = filtered_blocks["end_id_list"].apply(lambda x: x[:-1])

        # Filter end_id_list_trimmed based on time_gaps > min_stoppage_time
        filtered_blocks["valid_end_ids"] = filtered_blocks.apply(
        lambda row: [end_id for end_id, gap in zip(row["end_id_list_trimmed"], row["time_gaps"]) if gap > min_stoppage_time], 
        axis=1
        )
        all_end_ids = [end_id for sublist in filtered_blocks["valid_end_ids"] for end_id in sublist]

        # Flatten the list of filtered_end_ids and count occurrences of each ID
        end_id_counts = Counter(all_end_ids)

        # Find missing IDs that are not in top_end_stop_ids
        missing_ids_per_row = filtered_blocks["valid_end_ids"].apply(lambda row: [eid for eid in row if eid not in top_end_stop_ids])

        # Count frequency of missing IDs
        missing_id_counts = Counter([eid for row in missing_ids_per_row for eid in row])

        # Sort missing IDs by highest frequency
        sorted_missing_ids = sorted(missing_id_counts.keys(), key=lambda x: missing_id_counts[x], reverse=True)

        # Set to store newly added IDs
        added_ids = set()

        # Iterate through rows and add the minimum number of new IDs needed
        for idx, row in missing_ids_per_row.items():
            if not any(eid in top_end_stop_ids or eid in added_ids for eid in row):  
                # Find the most frequent missing ID and add it
                for new_id in sorted_missing_ids:
                    if new_id in row:
                        added_ids.add(new_id)
                        break  # Stop after adding one ID per row

        # Update top_end_stop_ids with the newly added IDs
        top_end_stop_ids = list(set(top_end_stop_ids).union(added_ids))

        # Apply function to blocks with total_distance_miles > Bus_range
        block_general["range_tracking"] = block_general.apply(
            lambda row: compute_range_tracking(row["distances_list"], row["time_gaps"], row["end_id_list"]),
            axis=1
        )
        # Identify infiseable block_ids where any range_tracking value is negative
        infiseable_blocks = block_general[block_general["range_tracking"].apply(lambda rt: any(x < 0 for x in rt) if rt else False)]["block_id"].tolist()

        # Identify block_ids where any range_tracking value is below critical_range
        blocks_below_critical = block_general[block_general["range_tracking"].apply(lambda rt: any(x < critical_range for x in rt) if rt else False)]["block_id"].tolist() 

    # Convert top_end_stop_ids to a DataFrame for merging
    selected_stops = pd.DataFrame(top_end_stop_ids, columns=["stop_id"])

    # Merge with stops dataset to get lat/lon
    proposed_locations = selected_stops.merge(stops[["stop_id", "stop_lat", "stop_lon"]], on="stop_id", how="left")

    # Drop any potential duplicates
    proposed_locations = proposed_locations.drop_duplicates().reset_index(drop=True)

    # Calculate the center of the map
    center_lat = shapes['shape_pt_lat'].mean()
    center_lon = shapes['shape_pt_lon'].mean() 

    # # Create the map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles=None)

    # # Add a tile layer with opacity control
    folium.TileLayer(
        tiles='https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
        attr='OpenStreetMap',
        name='Custom Background',
        opacity=0.6  # Set the desired opacity
    ).add_to(m)

    ax = shape_route.sort_values(by='modes')

    # Generate colors for routes
    num_routes = len(ax['route_id'].unique())  
    colors = plt.colormaps.get_cmap('Dark2')  
    
    # Convert colors to hex format
    color_list = [mcolors.to_hex(colors(i / num_routes)) for i in range(num_routes)]
    
    # Assign unique colors to each route_id
    route_colors = {route_id: color_list[i] for i, route_id in enumerate(ax['route_id'].unique())}
    
    # # Loop through each route_id and plot correctly
    for route_id in ax['route_id'].unique():
        color = route_colors[route_id]
        route_group = folium.FeatureGroup(name=f"Route: {route_id}")
        shape_ids = trips[trips['route_id'] == route_id]['shape_id'].unique()
        for shape_id in shape_ids:
            shape_data = shapes[shapes['shape_id'] == shape_id].sort_values(by='shape_pt_sequence')
            shape_coords = shape_data[['shape_pt_lat', 'shape_pt_lon']].values.tolist()
            folium.PolyLine(shape_coords, color=color, weight=2).add_to(m)
            #route_group.add_to(m)
    
    folium.LayerControl().add_to(m)
    
    for _, row in proposed_locations.iterrows():
        folium.Marker(location=[row["stop_lat"], row["stop_lon"]], icon=folium.Icon(color="blue")).add_to(m) 

    st.session_state["map"] = m
    st.session_state["routes_count"] = num_routes
    st.session_state["stops_count"] = Num_stop
    st.session_state["blocks_count"] = len(block_general)
    st.session_state["infeasible_blocks_count"] = len(infiseable_blocks)
    st.session_state["critical_blocks_count"] = len(blocks_below_critical)
    st.session_state["minimum_range_without_charger"] = min_range_without_charging
    st.session_state["num_locs"]=len(proposed_locations)

if "map" in st.session_state:
    st.write(f"Number of routes: {st.session_state['routes_count']}")
    st.write(f"Number of stops: {st.session_state['stops_count']}")
    st.write(f"Number of blocks: {st.session_state['blocks_count']}")
    st.write(f"Minimum bus range to cover all blocks without charging is {st.session_state['minimum_range_without_charger']} miles.")
    st.write(f"With your selected bus range and charging power:")
    st.write(f"&nbsp;&nbsp;&nbsp;&nbsp; -Number of infeasible blocks: {st.session_state['infeasible_blocks_count']}")
    st.write(f"&nbsp;&nbsp;&nbsp;&nbsp; -Number of blocks with range goes below {critical_range} miles: {st.session_state['critical_blocks_count']}")
    st.write(f"&nbsp;&nbsp;&nbsp;&nbsp; -Number of charger locations: {st.session_state['num_locs']}.")
    st.write(f"See below the map of the routes and the selected charging locations:")
    st_folium(st.session_state["map"], width=800, height=500)
