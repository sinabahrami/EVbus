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
from matplotlib.lines import Line2D
import textwrap

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Frame, PageTemplate, NextPageTemplate, KeepTogether, FrameBreak)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import Flowable
from textwrap import wrap
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import Table, TableStyle
from reportlab.pdfgen import canvas as pdfcanvas
import tempfile
import io
from io import BytesIO
import geopandas as gpd
import contextily as ctx
from shapely.geometry import LineString, Point
import shapely.affinity
from PIL import Image as PILImage


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

def create_bus_electrification_map(shapes_df, routes_df, trips_df, proposed_locations_df, wireless_track_shape_df, center_lat, center_lon, show_routes=True, show_chargers=True):
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
    
    # Routes
    if show_routes:
        all_routes = folium.FeatureGroup(name="All Routes", show=True)
        unique_routes = trips_df['route_id'].unique()
        num_routes = len(unique_routes)
        colormap = matplotlib.colormaps.get_cmap('tab20b')
        colors = [mcolors.to_hex(colormap(i / max(1, num_routes-1))) for i in range(num_routes)]
        route_colors = dict(zip(unique_routes, colors))

        for route_id in unique_routes:
            route_shapes = trips_df[trips_df['route_id']==route_id]['shape_id'].unique()
            for shape_id in route_shapes:
                shape_points = shapes_df[shapes_df['shape_id']==shape_id].sort_values('shape_pt_sequence')
                coords = shape_points[['shape_pt_lat','shape_pt_lon']].values.tolist()
                if len(coords)>1:
                    folium.PolyLine(coords, color=route_colors[route_id], weight=4).add_to(all_routes)
        all_routes.add_to(m)

    # Chargers
    if show_chargers and not proposed_locations_df.empty:
        charging_locations = folium.FeatureGroup(name="Stationary Chargers", show=True)
        for idx,row in proposed_locations_df.iterrows():
            folium.Marker(
                location=[row["stop_lat"], row["stop_lon"]],
                icon=folium.Icon(color="blue", icon="plug", prefix="fa"),
                popup=f"Charging Location ID: {row['stop_id']}"
            ).add_to(charging_locations)
        charging_locations.add_to(m)

    # Wireless tracks (optional)
    if show_chargers and not wireless_track_shape_df.empty:
        all_wireless_tracks = folium.FeatureGroup(name="Wireless Charging Tracks", show=True)
        for track in wireless_track_shape_df['counter'].unique():
            track_data = wireless_track_shape_df[wireless_track_shape_df['counter']==track].sort_values('target_shape_pt_sequence')
            coords = track_data[['shape_pt_lat','shape_pt_lon']].values.tolist()
            folium.PolyLine(coords, color="green", weight=4).add_to(all_wireless_tracks)
        all_wireless_tracks.add_to(m)

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
    
    if target_shape.empty:
        return None, [], 0  # No valid segment found
        
    if input_distance>target_shape["target_shape_dist_traveled"].max():
        input_distance=target_shape["target_shape_dist_traveled"].max()
    
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
            elif end_idx==len(target_shape)-1:
                segments.append(segment)
                segment_distances.append(total_distance)

    
    if not segments:
        return None, [], 0  # No valid segment found
    
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

def calculate_additional_buses(row, bus_range):
    if row['total_distance_miles'] > bus_range:
        min_val = min(row['range_tracking'])
        if min_val < 0:
            return int(np.ceil(abs(min_val) / bus_range))
    return 0

def reset_toggle():
    st.session_state.toggle_state = False  # Turn off toggle
    st.session_state.toggle_state_cost = False  # Turn off toggle

#####################################################
def generate_route_charger_maps(shapes_df, trips_df, proposed_locations_df, wireless_track_shape_df, center_lat, center_lon):
    # Convert routes to GeoDataFrame
    route_lines = []
    for route_id in trips_df['route_id'].unique():
        route_shapes = trips_df[trips_df['route_id'] == route_id]['shape_id'].unique()
        for shape_id in route_shapes:
            shape_points = shapes_df[shapes_df['shape_id'] == shape_id].sort_values('shape_pt_sequence')
            coords = list(zip(shape_points['shape_pt_lon'], shape_points['shape_pt_lat']))
            if len(coords) > 1:
                route_lines.append({'route_id': route_id, 'geometry': LineString(coords)})

    gdf_routes = gpd.GeoDataFrame(route_lines, crs="EPSG:4326").to_crs(epsg=3857)

    # Convert chargers to GeoDataFrame
    if not proposed_locations_df.empty:
        gdf_chargers = gpd.GeoDataFrame(
            proposed_locations_df.copy(),
            geometry=gpd.points_from_xy(proposed_locations_df['stop_lon'], proposed_locations_df['stop_lat']),
            crs="EPSG:4326"
        ).to_crs(epsg=3857)
        # Apply small jitter to separate overlapping points
        gdf_chargers['geometry'] = gdf_chargers['geometry'].apply(
            lambda p: shapely.affinity.translate(
                p,
                xoff=np.random.uniform(-300, 300),
                yoff=np.random.uniform(-300, 300)
            )
        )
    else:
        gdf_chargers = gpd.GeoDataFrame(columns=['geometry'], crs="EPSG:3857")

    # Convert wireless tracks to GeoDataFrame
    wireless_lines = []
    if not wireless_track_shape_df.empty:
        for track in wireless_track_shape_df['counter'].unique():
            track_data = wireless_track_shape_df[wireless_track_shape_df['counter']==track].sort_values('target_shape_pt_sequence')
            coords = list(zip(track_data['shape_pt_lon'], track_data['shape_pt_lat']))
            if len(coords) > 1:
                wireless_lines.append({'geometry': LineString(coords)})
    gdf_wireless = gpd.GeoDataFrame(wireless_lines, crs="EPSG:4326").to_crs(epsg=3857) if wireless_lines else gpd.GeoDataFrame(columns=['geometry'], crs="EPSG:3857")

    # Determine map extent
    all_points = gdf_routes.geometry.unary_union
    if not gdf_chargers.empty:
        all_points = all_points.union(gdf_chargers.geometry.unary_union)
    if not gdf_wireless.empty:
        all_points = all_points.union(gdf_wireless.geometry.unary_union)

    minx, miny, maxx, maxy = all_points.bounds
    buffer = 2000  # meters
    extent = [minx-buffer, maxx+buffer, miny-buffer, maxy+buffer]

    # Generate discrete colors for routes
    import matplotlib.patches as mpatches
    unique_routes = gdf_routes['route_id'].unique()
    colormap = matplotlib.cm.get_cmap('tab20b', len(unique_routes))
    route_colors = {rid: colormap(i) for i, rid in enumerate(unique_routes)}

    # Function to plot and return PNG bytes
    def plot_map(plot_routes=True, plot_chargers=True, plot_wireless=True):
        fig, ax = plt.subplots(figsize=(10,10))

        # Plot routes individually to create discrete legend
        legend_handles = []
        if plot_routes and not gdf_routes.empty:
            for route_id in unique_routes:
                gdf_routes[gdf_routes['route_id']==route_id].plot(ax=ax, color=route_colors[route_id], linewidth=3)
                legend_handles.append(mpatches.Patch(color=route_colors[route_id], label=f"Route {route_id}"))

        # Chargers
        if plot_chargers and not gdf_chargers.empty:
            gdf_chargers.plot(ax=ax, color='blue', markersize=50, marker='*', label='Stationary Charger')
            legend_handles.append(Line2D([0], [0], marker='*', color='w', label='Stationary Charger',markerfacecolor='blue', markersize=8))

        # Wireless tracks
        if plot_wireless and not gdf_wireless.empty:
            gdf_wireless.plot(ax=ax, color='green', linewidth=3, label='Wireless Track')
            legend_handles.append(mpatches.Patch(color='green', label='Wireless Track'))

        # Set extent
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

        # Add basemap
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

        # Remove axes
        ax.axis('off')

        # Add discrete legend
        if legend_handles:
            ax.legend(handles=legend_handles, loc='upper right', fontsize=9)

        # Tight layout
        plt.tight_layout()

        # Save to BytesIO
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return buf

    # Generate three maps
    routes_only = plot_map(plot_routes=True, plot_chargers=False, plot_wireless=False)
    chargers_only = plot_map(plot_routes=False, plot_chargers=True, plot_wireless=True)
    routes_plus_chargers = plot_map(plot_routes=True, plot_chargers=True, plot_wireless=True)

    return routes_only, chargers_only, routes_plus_chargers

class NumberedCanvas(pdfcanvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        # Save state of the just-completed page
        self._saved_page_states.append(dict(self.__dict__))
        # Start a fresh page state
        self._startPage()

    def save(self):
        # Save state for the last page as well
        self._saved_page_states.append(dict(self.__dict__))

        # Restore and write each saved page, adding page numbers as needed
        for i, state in enumerate(self._saved_page_states):
            self.__dict__.update(state)
            if i >= 1:  # skip cover
                logical_page = i
                self.saveState()
                try:
                    self.setStrokeColorRGB(0, 0, 0)  # black line
                    self.setLineWidth(1.0)  # thin line
                    # Draw line across bottom margin
                    self.line(1.0*inch, 0.75*inch, letter[0] - 1.0*inch, 0.75*inch)
                    self.setFont('Times-Italic', 9)
                    self.setFillColorRGB(0, 0, 0)
                    self.drawRightString(letter[0] - 1.0*inch, 0.5*inch, f"Page {logical_page}")
                finally:
                    self.restoreState()

            # Write the page
            if i < len(self._saved_page_states) - 1:  # not the last page
                pdfcanvas.Canvas.showPage(self)

        # finish
        pdfcanvas.Canvas.save(self)

class FullPageImage(Flowable):
    def __init__(
        self,
        image_path,
        width,
        height,
        title=None,
        subtitle=None,
        title_font_size=28,
        subtitle_font_size=18,
        max_title_width=40,
        max_subtitle_width=60,
        bottom_margin=1.5*inch,  # distance from bottom for first line
        top_margin=1.5*inch
    ):
        super().__init__()
        self.image_path = image_path
        self.width = width
        self.height = height
        self.title = title
        self.subtitle = subtitle
        self.title_font_size = title_font_size
        self.subtitle_font_size = subtitle_font_size
        self.max_title_width = max_title_width
        self.max_subtitle_width = max_subtitle_width
        self.bottom_margin = bottom_margin
        self.top_margin=top_margin

    def wrap(self, availWidth, availHeight):
        return self.width, self.height

    def draw(self):
        # Draw full-page image
        self.canv.drawImage(self.image_path, 0, 0, width=self.width, height=self.height)

        # Wrap title and subtitle
        title_lines = wrap(self.title, self.max_title_width) if self.title else []
        #subtitle_lines = wrap(self.subtitle, self.max_subtitle_width) if self.subtitle else []

        # Handle subtitle: first split by \n, then wrap each segment
        subtitle_lines = []
        if self.subtitle:
            for part in self.subtitle.split("\n"):
                wrapped = wrap(part, self.max_subtitle_width)
                subtitle_lines.extend(wrapped)
                
        # Starting Y position from bottom
        yt=self.top_margin+600
        yb = self.bottom_margin-20

        
        # Draw title above subtitle
        if title_lines:
            self.canv.setFont("Times-Bold", self.title_font_size)
            self.canv.setFillColorRGB(0, 0, 0)
            yt += 0.1*inch  # small gap above subtitle
            for line in title_lines:
                self.canv.drawCentredString(self.width / 2, yt, line)
                yt -= self.title_font_size + 4  # line spacing

        # Draw subtitle first (closest to bottom)
        if subtitle_lines:
            self.canv.setFont("Times-Bold", self.subtitle_font_size)
            self.canv.setFillColorRGB(0, 0, 0)  # white
            for line in subtitle_lines:  # draw from bottom up
                self.canv.drawCentredString(self.width / 2, yb, line)
                yb -= self.subtitle_font_size + 3  # spacing between lines

def generate_transit_report(
    inputs,
    outputs,
    map_image_path=None,
    econ_toggle=False,
    econ_figure_path=None,
    agency_name="Transit Agency",
    title_image_path=None,
    charger_image_path=None,
    routencharger_image_path=None
):
    buffer = BytesIO()
    # Figure counter
    figure_counter = 1
    table_counter =1 
    # Styles
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Heading1Center", parent=styles["Heading1"], alignment=1))
    styles.add(ParagraphStyle(name="BodyTextCustom", parent=styles["BodyText"], fontSize=11, leading=14))
    styles.add(ParagraphStyle(name='Caption',parent=styles["Normal"],alignment=TA_CENTER,fontSize=10,leading=12))
    
    page_width, page_height = letter

    # --- Define Frames & Templates ---
    # Full-page front page
    front_frame = Frame(
        x1=0, y1=0, width=page_width, height=page_height,
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, id='front'
    )
    front_template = PageTemplate(id='front', frames=[front_frame])

    # Main content frame
    main_frame = Frame(
        x1=1.0*inch, y1=1.0*inch,
        width=page_width - 2*inch,
        height=page_height - 2*inch,
        leftPadding=0.2, rightPadding=0.2, topPadding=0.2, bottomPadding=0.2,
        id='normal'
    )
    main_template = PageTemplate(id='normal', frames=[main_frame])#, onPageEnd=add_page_number)

    def add_figure(img_bytes, caption_text):
        nonlocal figure_counter
        try:
            # Open with PIL to validate
            pil_img = PILImage.open(io.BytesIO(img_bytes) if isinstance(img_bytes, bytes) else img_bytes)
            pil_img.verify()  # raises if invalid
    
            # Use BytesIO to feed ReportLab Image
            img_stream = io.BytesIO(img_bytes) if isinstance(img_bytes, bytes) else img_bytes
            reportlab_img = Image(img_stream, width=letter[0]-4*inch, height=4*inch)
    
        except Exception as e:
            reportlab_img = Paragraph(f"[Error loading image: {e}]", styles["Normal"])
    
        story.append(KeepTogether([
            reportlab_img,
            Spacer(1, 6),
            Paragraph(f"<b>Figure {figure_counter}:</b> {caption_text}", styles["Caption"]),
            Spacer(1, 1)
        ]))
        figure_counter += 1

    def add_table_from_dataframe(df, caption_text, styles, col_widths=None):
        nonlocal table_counter
        # Select the first three columns
        df_table = df.iloc[:, :3].copy()
        df_table.columns = ["Block ID", "Total Distance (miles)", "Routes Included"]  # nice headers

        # Round the second column to 1 decimal
        df_table["Total Distance (miles)"] = df_table["Total Distance (miles)"].astype(float).round(1)

        # Convert third column (list/array) to comma-separated string
        def format_routes(x):
            if isinstance(x, (list, tuple, pd.Series, np.ndarray)):
                return ", ".join(str(i) for i in x)
            else:
                return str(x)
        df_table["Routes Included"] = df_table["Routes Included"].apply(format_routes)

        # Convert DataFrame to list of lists
        data = [df_table.columns.tolist()] + df_table.values.tolist()
        
        # Determine column widths (optional: scale to text width)
        if col_widths is None:
            # Approximate: evenly divide available text width
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.units import inch
            page_width, page_height = letter
            text_width = page_width - 3*inch  # assuming main_frame has 1-inch margins
            col_widths = [text_width * 0.2, text_width * 0.3, text_width * 0.5]
        
        # Create the table
        tbl = Table(data, colWidths=col_widths, repeatRows=1)
        
        # Style the table
        tbl.setStyle(TableStyle([
            ('FONTNAME', (0,0), (-1,0), 'Times-Bold'),    # header bold
            ('ALIGN', (0,0), (-1,0), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('ALIGN', (0,1), (1,-1), 'CENTER'),          # first two columns values centered
            ('ALIGN', (2,1), (2,-1), 'LEFT'),            # third column values left-aligned
            ('GRID', (0,0), (-1,-1), 0.5, colors.black),
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('LEFTPADDING', (0,0), (-1,-1), 4),
            ('RIGHTPADDING', (0,0), (-1,-1), 4),
        ]))

        # Add table and caption to story
        caption_html = f"<b>Table {table_counter}:</b> {caption_text}"
        story.append(Paragraph(caption_html, styles["Caption"]))
        story.append(Spacer(1,6))
        story.append(tbl)
        story.append(Spacer(1,12))
        
        table_counter+=1

    # Document  
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    doc.addPageTemplates([front_template, main_template])

    story = []

    # --- Front Page ---
    if title_image_path:
        story.append(FullPageImage(
            title_image_path,
            width=letter[0],
            height=letter[1],
            title=f"Transit Electrification Report for {agency_name} Transit System",
            subtitle="Produced by \n https://transit-electrification.streamlit.app/" # \n developed by \n Yafeng Yin, Sina Bahrami, Manzi Li, Michele Mueller, and Caitlin Day"
        ))
        #story.append(PageBreak())
        
    story.append(NextPageTemplate('normal'))
    story.append(FrameBreak)
    story.append(NextPageTemplate('normal'))

    # --- Introduction ---
    story.append(Paragraph("Introduction", styles["Heading1"]))
    intro_text = (
        f"This report provides a detailed summary of the transit electrification study for the {agency_name} transit system. "
        f"The system includes {outputs.get('total_routes', 'N/A')} routes, and "
        f"{outputs.get('total_stops', 'N/A')} stops. " 
        f"Figure {figure_counter} shows the map of the service."
    )
    story.append(Paragraph(intro_text, styles["BodyTextCustom"]))
    story.append(Spacer(1, 12))

    if map_image_path:
        add_figure(map_image_path, "Map of the transit service.")

    
    intro_text2 = (
        f"The agency operates {outputs.get('total_blocks', 'N/A')} blocks, each defined as a sequence of consecutive routes assigned to a single vehicle for daily operations. "
        f"Table {table_counter} provides a detailed overview of these blocks, including their total driving distances and the routes serviced within each block."
    )    
    story.append(Paragraph(intro_text2, styles["BodyTextCustom"]))
    story.append(Spacer(1, 12))
    add_table_from_dataframe(
        outputs.get('block_general'), 
        caption_text="Summary of blocks.",
        styles=styles
    )

    # --- Inputs & Outputs ---
    story.append(Paragraph("Selected Electrification Characteristics", styles["Heading1"]))
    inputs_text = (
        f"The selected electric bus range is {inputs.get('bus_range', 'N/A')} miles, "
        f"and buses consume {inputs.get('energy_usage', 'N/A') / 60:.2f} kWh per mile of travel. "
    )

    if econ_toggle:
        inputs_text += f"Each bus costs ${inputs.get('bus_price', 'N/A'):,}. "

    if inputs.get('stationary_setup_time_hr', 0) >= 1:
        inputs_text += (
            f"The selected stationary charging system is plug-in, with a power output of {inputs.get('stationary_power', 'N/A')} kW. "
            f"Buses require {inputs.get('stationary_setup_time_hr', 0)} minutes to plug in and start charging after stopping. "
        )
    elif inputs.get('stationary_setup_time_hr', 0) > 0:
        inputs_text += (
            f"The selected stationary charging system is wireless, with a power output of {inputs.get('stationary_power', 'N/A')} kW. "
            f"Buses require {inputs.get('stationary_setup_time_hr', 0) * 60:.0f} seconds to start charging after stopping. "
        )
    else:
        inputs_text += (
            f"The selected stationary charging system is wireless, with a power output of {inputs.get('stationary_power', 'N/A')} kW, "
            f"and buses can begin charging immediately after stopping. "
        )

    if econ_toggle:
        inputs_text += (
            f"The average cost to build a stationary charging station is ${inputs.get('stationary_charger_cost', 'N/A'):,}. "
        )
        if inputs.get('stationary_setup_time_hr', 0) < 1:
            inputs_text += (
                f"Installing a receiver coil on each bus for wireless charging costs ${inputs.get('coil_install_cost', 'N/A'):,}. "
            )

    if inputs.get('dynamic_power', 0) > 0:
        inputs_text += (
            f"In addition, dynamic in-motion wireless charging is available, providing {inputs.get('dynamic_power', 0)} kW of charging power. "
        )
        if econ_toggle:
            inputs_text += (
                f"Constructing dynamic wireless charging track costs ${inputs.get('dynamic_track_cost', 'N/A'):,} per mile."
            )

    story.append(Paragraph(inputs_text, styles["BodyTextCustom"]))
    story.append(Spacer(1,12))


    # --- Results ---
    story.append(Paragraph("Results", styles["Heading1"]))

    # Start with electrifiable blocks
    outputs_text = f"With the selected bus and charger configuration, a total of {outputs.get('electrifiable_blocks','N/A')} blocks can be successfully electrified. "

    # Add infeasible blocks if any
    if outputs.get('non_electrifiable_blocks', 0) > 0:
        outputs_text += (f"However, {outputs.get('non_electrifiable_blocks','N/A')} blocks remain infeasible under the current configuration. ")

        report_infeasible_blocks = outputs.get('infeasible_block_ids', [])
        if report_infeasible_blocks and report_infeasible_blocks != 'N/A':
            if len(report_infeasible_blocks) == 1:
                blocks_str = report_infeasible_blocks[0]
            else:
                blocks_str = ", ".join(report_infeasible_blocks[:-1]) + ", and " + report_infeasible_blocks[-1]
            outputs_text += f"The specific block IDs corresponding to these infeasible cases are: {blocks_str}. "

    # Add stationary chargers if any
    if outputs.get('num_stationary_chargers', 0) > 0:
        outputs_text += f"To achieve the electrification of the feasible blocks, it will be necessary to install {outputs.get('num_stationary_chargers','N/A')} stationary charging stations. "

    # Add dynamic wireless charging if applicable
    if outputs.get('dynamic_lane_length', 0) > 0:
        outputs_text += f"In addition to the stationary chargers, {outputs.get('dynamic_lane_length','N/A')} miles of dynamic wireless charging track will be required. "

    # Reference figure if applicable
    if outputs.get('num_stationary_chargers', 0) > 0 or outputs.get('dynamic_lane_length', 0) > 0:
        outputs_text += f"The spatial distribution of these chargers is presented in Figure {figure_counter}. Furthermore, Figure {figure_counter+1} illustrates the arrangement of chargers along the transit routes."

    # Append paragraph
    story.append(Paragraph(outputs_text, styles["BodyTextCustom"]))
    story.append(Spacer(1,12))
    
    if charger_image_path:
        add_figure(charger_image_path, "Spatial distribution of chargers.")
        add_figure(routencharger_image_path, "Arrangement of chargers along transit routes.")
    
    
    if econ_toggle and econ_figure_path:
        if len(outputs.get('categories'))==2:
            econ_text = (
                f"Under the selected configuration, one alternative approach to electrifying all blocks is to expand the fleet size by purchasing "
                f"{outputs.get('additional_fleet_cost_no_ivc')/inputs.get('bus_price'):.0f} additional buses instead of installing chargers. "
                f"Figure {figure_counter} compares the cost of acquiring additional buses with the cost of charger installation."
            )
            story.append(Paragraph(econ_text, styles["BodyTextCustom"]))
            story.append(Spacer(1,12))
            add_figure(econ_figure_path, "Cost comparison.")
        elif len(outputs.get('categories'))==3:    
            econ_text = (
                f"Under the selected configuration, there are two possible approaches to electrifying all blocks: "
                f"(i) expanding the fleet size by purchasing {outputs.get('additional_fleet_cost_no_ivc')/inputs.get('bus_price'):.0f} additional buses "
                f"instead of installing chargers, or (ii) installing chargers while also purchasing "
                f"{outputs.get('additinal_fleet_cost')/inputs.get('bus_price'):.0f} additional buses. "
                f"Figure {figure_counter} compares the costs associated with these scenarios."
            )
            story.append(Paragraph(econ_text, styles["BodyTextCustom"]))
            story.append(Spacer(1,12))
            add_figure(econ_figure_path, "Cost comparison.")
        
    

    # --- Build PDF ---
    doc.build(story, canvasmaker=NumberedCanvas)
    buffer.seek(0)
    return buffer
###
def main():
    flag_done=0

    # Initialize session state if not already set
    if "output" not in st.session_state:
        st.session_state["output"] = None  # Or set a default DataFrame

    
    # Set page config for a cleaner interface
    st.set_page_config(
        page_title="Bus Electrification Analysis",
        page_icon="🚌",
        layout="wide"
    )
    
    # List of allowed agency zip files
    agencies = ["BATA (Traverse City)", "CATA (Lansing)", "DDOT (Detroit)", "MAX (Holland)", "METRO (Kalamazoo)", "Smart (Detroit)", "The Rapid (Grand Rapids)", "TheRide (Ann Arbor-Ypsilanti)", "UMich"]
    
    # Application UI
    st.title("🚌 Bus System Electrification Analysis")
    # st.markdown("*Internal Testing Release*")
    
    # Create sidebar for settings
    with st.sidebar:
        # Allow user to select agency
        selected_agency = st.selectbox("Transit Agency", agencies, on_change=reset_toggle)
        
        st.header("Configuration")

        # Initialize session state for the toggle
        if "toggle_state" not in st.session_state:
            st.session_state.toggle_state = False

        if "toggle_state_cost" not in st.session_state:
            st.session_state.toggle_state_cost = False
       
        
        # Energy and range parameters
        #st.subheader("Electric Bus Parameters")
        bus_range = st.number_input("Electric Bus Range [mile]", min_value=5, value=150, step=10, help="The distance a bus can travel with a fully charged battery.")
        charging_power = st.number_input("Stationary Charging Power [kW]", min_value=0, value=250, step=50, help="Stationary chargers can be either wireless or plug-in types.")
        min_stoppage_time = st.number_input(label="Stationary Charging Setup Time [min]", min_value=0.0, value=0.0, step=0.1,help="Wireless chargers can start charging immediately, whereas plug-in chargers require a few minutes to connect and begin charging.")
        dynamic_wireless_charging_power = st.number_input("In-motion Charging Power [kW]", min_value=0, value=0, step=10)

        # Advanced parameters with expander to keep interface clean
        with st.expander("Advanced Parameters"):
            energy_usage = st.number_input("Bus Energy Usage [kWh/mile]", min_value=0.05, value=2.5, step=0.05)*60
            #critical_range = st.number_input("Critical range threshold (miles)", min_value=5, value=20, step=5)
            

        if selected_agency=="BATA (Traverse City)":
            feasible_block_options=[424115, 424116, 424117, 424118, 424119, 424120, 424121, 424122, 424123, 424124, 424125, 424126, 424127, 424128, 424129, 424130, 424131, 424132, 424133, 424134, 424135, 424136, 424137, 424138, 424139, 424140, 424141, 424142]
            feasible_route_options=[5990, 5991, 5992, 5993, 5994, 5995, 5996, 6003, 6013, 6107, 6234, 6451]
        elif selected_agency=="CATA (Lansing)":
            feasible_block_options=['b_457160', 'b_457161', 'b_457162', 'b_457163', 'b_457164', 'b_457165', 'b_457166', 'b_457167', 'b_457168', 'b_457169', 'b_457170', 'b_457171', 'b_457172', 'b_457173', 'b_457174', 'b_457233', 'b_457234', 'b_457237', 'b_457238', 'b_457239', 'b_457266', 'b_457267', 'b_457268', 'b_457269', 'b_457279', 'b_457296', 'b_457297', 'b_457298', 'b_457299', 'b_457314', 'b_457315', 'b_457316', 'b_457340', 'b_457341', 'b_457342', 'b_457343', 'b_457362', 'b_457363', 'b_457364', 'b_457372', 'b_457381', 'b_457382', 'b_457393', 'b_457394', 'b_457403', 'b_457415', 'b_457416', 'b_457435', 'b_457436', 'b_457437', 'b_457446', 'b_457447', 'b_457462', 'b_457463', 'b_457464', 'b_457478', 'b_457479', 'b_457480', 'b_457490', 'b_457491', 'b_457502', 'b_457503', 'b_457518', 'b_457519', 'b_457520', 'b_457542', 'b_457543', 'b_457544', 'b_457545', 'b_457570', 'b_457571', 'b_457572', 'b_457573', 'b_457574', 'b_457575', 'b_457584', 'b_457585', 'b_457598', 'b_457599', 'b_457600', 'b_457612', 'b_457617', 'b_457622', 'b_457829', 'b_457835', 'b_457863', 'b_457864', 'b_457865']
            feasible_route_options=[11439, 11440, 11441, 11443, 11444, 11445, 11446, 11447, 11448, 11449, 11450, 11451, 11452, 11453, 11454, 11456, 11457, 11458, 11459, 11460, 11461, 11462, 11464, 11465, 11466, 11467, 11471, 11472, 11473, 11474, 11483, 11484]     
        elif selected_agency=="DDOT (Detroit)":
            feasible_block_options=['10-1', '10-2', '10-200', '10-201', '10-202', '10-203', '10-204', '10-205', '10-206', '10-207', '10-400', '10-401', '10-402', '10-500', '10-501', '10-502', '10-503', '11-200', '11-201', '11-202', '11-203', '12-1', '12-2', '15-200', '15-201', '15-202', '15-203', '15-204', '15-400', '15-401', '16-1', '16-200', '16-201', '16-202', '16-203', '16-204', '16-205', '16-206', '16-207', '16-208', '16-209', '16-210', '16-211', '16-212', '16-213', '16-214', '16-215', '16-216', '16-217', '16-400', '16-401', '16-402', '17-1', '17-2', '17-200', '17-201', '17-202', '17-203', '17-204', '17-205', '17-206', '17-207', '17-208', '17-209', '17-210', '17-211', '17-3', '17-400', '17-401', '17-402', '17-403', '17-500', '17-501', '18-200', '18-201', '18-202', '18-203', '18-204', '18-205', '18-500', '19-400', '19-500', '2-200', '2-201', '2-202', '2-203', '2-204', '2-500', '2-501', '23-400', '27-1', '27-200', '27-201', '27-202', '27-203', '27-400', '27-500', '29-1', '29-2', '3-1', '3-2', '3-200', '3-201', '3-202', '3-203', '3-204', '3-205', '3-206', '3-207', '3-208', '3-3', '3-400', '3-401', '3-402', '3-500', '3-501', '3-502', '3-503', '30-1', '30-200', '30-201', '30-400', '30-500', '31-1', '31-200', '31-201', '31-202', '31-203', '31-204', '31-205', '32-1', '32-200', '32-201', '32-202', '32-203', '32-204', '32-205', '32-206', '32-207', '32-208', '32-400', '32-401', '32-500', '32-501', '38-1', '38-200', '38-201', '38-202', '38-203', '38-204', '38-400', '38-401', '38-500', '38-501', '39-200', '39-201', '39-400', '39-500', '4-1', '4-200', '4-201', '4-202', '4-203', '4-204', '4-205', '4-206', '4-207', '4-208', '4-209', '4-210', '4-211', '4-212', '4-213', '4-214', '4-215', '4-216', '4-217', '4-218', '4-512', '40-1', '40-200', '40-201', '41-1', '41-200', '41-201', '41-400', '41-500', '42-200', '42-201', '43-200', '43-201', '43-202', '43-203', '43-204', '43-400', '43-500', '46-400', '47-200', '47-400', '47-500', '47-501', '52-200', '52-400', '54-200', '54-201', '54-202', '54-203', '54-400', '54-500', '6-1', '6-2', '6-200', '6-201', '6-202', '6-203', '6-204', '6-205', '6-206', '6-207', '6-208', '6-209', '6-210', '6-400', '6-401', '6-500', '60-1', '60-2', '60-200', '60-201', '60-400', '60-401', '60-402', '60-500', '60-501', '67-200', '67-201', '67-400', '68-400', '7-200', '7-201', '7-202', '7-203', '7-204', '7-205', '7-206', '7-207', '7-208', '7-209', '7-210', '7-211', '7-212', '7-213', '7-214', '7-215', '7-216', '7-217', '7-218', '7-219', '7-220', '7-221', '7-222', '7-400', '7-401', '7-402', '7-500', '7-501', '7-502', '8-1', '8-2', '8-200', '8-201', '8-202', '8-3', '8-4', '8-400', '8-5', '8-500', '8-501', '8-6', '8-7', '9-200', '9-201', '9-202', '9-203', '9-204', '9-205', '9-206', '9-207', '9-208', '9-209', '9-210', '9-211', '9-212', '9-213', '9-214', '9-215', '9-216', '9-500', '9-501', '9-502', '900-1', '900-2', '900-200', '900-201', '900-202', '900-203', '900-204', '900-205', '900-206', '900-207', '900-208', '900-209', '900-210', '900-211', '900-212', '900-213', '900-214', '900-215', '900-216', '900-217', '900-218', '900-3', '900-4', '900-400', '900-401', '900-500', '900-501', '900-502', '900-503']
            feasible_route_options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 23, 27, 29, 30, 31, 32, 38, 39, 40, 41, 42, 43, 46, 47, 52, 54, 60, 67, 68]
        elif selected_agency=="MAX (Holland)":
            feasible_block_options= [1, 2, 3, 4, 5, 6, 7, 8]           
            feasible_route_options= ['Rt1', 'Rt2', 'Rt3', 'Rt4', 'Rt5', 'Rt6', 'Rt7', 'Rt8']
        elif selected_agency=="METRO (Kalamazoo)":
            feasible_block_options= [101, 102, 201, 202, 301, 401, 501, 601, 701, 801, 901, 902, 1001, 1101, 1102, 1201, 1202, 1301, 1401, 1402, 1601, 1602, 1901, 2501, 2601, 2701]         
            feasible_route_options= [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 19, 25, 26, 27]
        elif selected_agency=="Smart (Detroit)":
            feasible_block_options= [11201, 11211, 11221, 11231, 11241, 11251, 11261, 11271, 11281, 11291, 11301, 11311, 11321, 11331, 11341, 11351, 11401, 11411, 11421, 11431, 11441, 11451, 11461, 11471, 11481, 11601, 11611, 11621, 11631, 11641, 11651, 12001, 12011, 12021, 12031, 12041, 12051, 12061, 12071, 12651, 12661, 12671, 12681, 12691, 12701, 12711, 12751, 12801, 12811, 12821, 12831, 12841, 12851, 12861, 12871, 13051, 13061, 13071, 13081, 18001, 18011, 18021, 18031, 18041, 18051, 23652, 23662, 23672, 23682, 23692, 23702, 24052, 24062, 24072, 24082, 24092, 24102, 24152, 24162, 24172, 24182, 24302, 24312, 24322, 24332, 24502, 24512, 24522, 24532, 24542, 24552, 24562, 24572, 24582, 24592, 24702, 24712, 24722, 24732, 24742, 24752, 24762, 24772, 24842, 24852, 24862, 24872, 24942, 24952, 24962, 24972, 24982, 24992, 25002, 25012, 27102, 27112, 27122, 27132, 27312, 27322, 27402, 27412, 27592, 27602, 27612, 27622, 27632, 27642, 27802, 27902, 27912, 27922, 27932, 27942, 27952, 27962, 27972, 27982, 27992, 28002, 28012, 85108, 85118, 85128, 85138, 85148, 85158, 85168, 85178, 85188, 85198, 85208, 85218, 85258, 85268, 85278, 85288, 85498, 85508, 85518, 85528, 85538, 85548, 85558, 85608, 85618, 85628, 85638, 85648, 85658, 85668, 85678, 85688, 85698, 85708, 85718, 85728, 85738, 85748, 85758, 85768, 86108, 86118, 86128, 86138, 87108, 87118, 87128, 87138, 87298, 87308, 87318, 87338, 87348, 87408, 87418, 87428, 87438, 87448, 87458, 87608, 87618, 87628, 87638, 87648, 87658, 87808, 87818, 87828, 87838, 87848, 87858, 88008, 88018, 88028, 88038, 88048, 88058, 912011, 912021, 912031, 912041, 912051, 912061, 912071, 912081, 912091, 912101, 912111, 912121, 912131, 912141, 912151, 912161, 912511, 912521, 912531, 912541, 912551, 912561, 912571, 912581, 912591, 912601, 912611, 912621, 912631, 912641, 912651, 912661, 922012, 922022, 922032, 922042, 922052, 922062, 922072, 922082, 922092, 922102, 922112, 922122, 922132, 922142, 922152, 922162, 922512, 922522, 922532, 922542, 922552, 922562, 922572, 922582, 922592, 922602, 922612, 922622, 922632, 922642, 922652, 922662, 981018, 982018, 982028, 982038, 982048, 982058, 982068, 982078, 982088, 982098, 982108, 982118, 982128, 982138, 982148, 982158, 11261261, 11262261, 11263261, 11264261, 11265261, 11266261, 11267261, 11268261, 11269261, 11270261, 11271261, 11272261, 22461461, 22462461, 22463461, 22464461, 22465461, 22466461, 22467461, 22468461, 22469461, 22470461, 22471461, 22472461, 22473461, 88561561, 88562561, 88563561, 88564561, 88565561, 88566561, 88567561, 88568561]
            feasible_route_options= [125, 140, 160, 200, 210, 250, 255, 261, 275, 280, 305, 375, 405, 415, 420, 430, 450, 460, 461, 462, 492, 494, 495, 510, 525, 530, 550, 560, 562, 563, 610, 615, 620, 635, 710, 730, 740, 759, 760, 780, 790, 796, 805, 830, 851, 901, 902, 903, 904, 905]
        elif selected_agency=="The Rapid (Grand Rapids)":
            feasible_block_options= [101, 102, 103, 104, 105, 106, 107, 201, 202, 203, 204, 205, 206, 207, 301, 306, 307, 308, 401, 402, 403, 404, 405, 406, 407, 408, 501, 502, 503, 505, 506, 601, 602, 603, 604, 605, 701, 702, 703, 801, 802, 803, 901, 902, 903, 904, 905, 907, 1001, 1002, 1003, 1004, 1005, 1101, 1102, 1103, 1104, 1201, 1202, 1301, 1302, 1402, 1501, 1502, 2401, 2402, 2701, 2801, 2802, 2803, 3301, 3302, 3303, 3701, 3702, 3703, 4401, 4402, 4403, 4501, 4502, 4503, 4504, 4505, 4506, 4507, 4508, 4509, 4801, 4802, 4803, 5101, 5102, 5103, 5201, 5202, 5203, 5501, 6001, 7102, 7202, 7301, 7402, 7502, 7602, 9001, 9002, 9003, 9004, 9005, 9006, 9007]
            feasible_route_options= [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 24, 27, 28, 33, 37, 44, 45, 48, 51, 52, 55, 60, 85, 90]
        elif selected_agency=="TheRide (Ann Arbor-Ypsilanti)":
            feasible_block_options= [4014, 4024, 4034, 4044, 4054, 4064, 4074, 4084, 4094, 4104, 4114, 4124, 4134, 4144, 4154, 4164, 4174, 5015, 5025, 5035, 5045, 5055, 5065, 5075, 5085, 5095, 6016, 6026, 6036, 6046, 6056, 6066, 6076, 6086, 6096, 260126, 260226, 260326, 260426, 260526, 260626, 260726, 260826, 260926, 270127, 270227, 270327, 270427, 270527, 270627, 270727, 270827, 270927, 271027, 271127, 271227, 271327, 271427, 300130, 300230, 300330, 300430, 300530, 300630, 300730, 340134, 340234, 420142, 420242, 420342, 420442, 420542, 420642, 420742, 420842, 420942, 421042, 421142, 430143, 430243, 430343, 610161, 610261, 610361, 610461, 620162, 620262, 620362, 620462, 630163, 630263, 630363, 630463, 630563, 630663, 1041104, 1042104, 1043104]
            feasible_route_options= [3, 4, 5, 6, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 42, 43, 44, 45, 46, 47, 61, 62, 63, 64, 65, 66, 67, 68, 104]
        elif selected_agency=="UMich":
            feasible_block_options =['BB200', 'BB201', 'BB202', 'BB203', 'BB204', 'BB205', 'BB206', 'BB207', 'BB208', 'BB209', 'BB210', 'BB211', 'BB212', 'BB213', 'BB214', 'BB215', 'BB216', 'BB217', 'BB218', 'CN100', 'CN101', 'CN102', 'CN103', 'CN104', 'CN105', 'CN106', 'CN107', 'CN108', 'CN109', 'CN110', 'CS100', 'CS101', 'CS102', 'CS103', 'CS104', 'CSX550', 'CSX551', 'CSX552', 'DD250', 'DD251', 'MX500', 'MX501', 'MX502', 'MX503', 'MX504', 'MX505', 'MX506', 'MX507', 'MX508', 'NES700', 'NES701', 'NES702', 'NES703', 'NW300', 'NW301', 'NW302', 'NW303', 'NW304', 'NW305', 'NW306', 'NW307', 'NW308', 'NW309', 'NW310', 'NW311', 'NX250', 'NX251', 'NX252', 'NX253', 'NX254', 'NX255', 'NX256', 'NX257', 'WS600', 'WS601', 'WS602', 'WS603', 'WX600', 'WX601']
            feasible_route_options=['BB', 'CN', 'CS', 'CSX', 'DD', 'MX', 'NES', 'NW', 'NX', 'OS', 'WS', 'WX']
        
        # Toggle button
        toggle_value = st.toggle("Specific route or block", value=st.session_state.toggle_state,help="Toggle to focus the analysis on specific routes or blocks. When analyzing specific routes, the model automatically includes other routes located on the same blocks as the chosen route(s).")
        # Update session state when toggle is changed
        st.session_state.toggle_state = toggle_value
       
        if st.session_state.toggle_state==True:
            option = st.radio("Choose an option:", ["Block", "Route"])
            add_flag=1
            if option =="Block":
                user_block_choice = st.multiselect("Choose desired block(s):", feasible_block_options)
                user_route_choice=feasible_route_options
            if option =="Route":
                user_route_choice = st.multiselect("Choose desired route(s):", feasible_route_options)
                user_block_choice=feasible_block_options
        else:
            add_flag=0
            user_block_choice=feasible_block_options
            user_route_choice=feasible_route_options
            
        
        critical_range=20

        toggle_value_cost = st.toggle(
            "Economic Analysis",
            value=st.session_state.get("toggle_state_cost", False),
            help="You can compare the costs of different scenarios."
        )
        st.session_state.toggle_state_cost = toggle_value_cost
        
        if st.session_state.toggle_state_cost:
            # Helper function for formatted input
            def number_input_with_commas(label, value, help_text=""):
                formatted_value = "{:,}".format(value)
                user_input = st.text_input(label, formatted_value, help=help_text)
                try:
                    # Remove commas before converting back
                    return int(user_input.replace(",", "").strip())
                except ValueError:
                    return value
        
            bus_cost = number_input_with_commas(
                "Electric Bus Price [$]",
                500_000,
                "Cost of purchasing an electric bus."
            )
        
            stationary_charger_cost = number_input_with_commas(
                "Cost of Building Stationary Charging [$]",
                200_000,
                "Average cost of building a stationary charging station."
            )
        
            if min_stoppage_time < 1:
                bus_coil_cost = number_input_with_commas(
                    "Cost of Installing Wireless Charging Receiver on Each Bus [$]",
                    50_000,
                    "Cost of installing a wireless charging receiver on each bus."
                )
            else:
                bus_coil_cost = 0
        
            dynamic_charger_cost = number_input_with_commas(
                "Cost of Constructing Dynamic Charging per Mile [$]",
                2_500_000,
                "Average cost of constructing a dynamic charging track per mile."
            )   
            
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

                num_blocks_total=len(block_general)
                # Filter to include only block_id that are in user choice
                block_general = block_general[block_general['block_id'].isin(user_block_choice)]
                block_general = block_general[block_general['routes_in_block_id'].apply(lambda x: any(sel_route in user_route_choice for sel_route in x))]
                
                
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
                
                #calculate cost with no IVC charger and additional buses
                block_general['additional_buses'] = block_general.apply(calculate_additional_buses, axis=1, args=(bus_range,))
                
                if toggle_value_cost==True:
                    existing_fleet_cost=len(block_general)*bus_cost
                    additional_fleet_cost_no_ivc=bus_cost*block_general['additional_buses'].sum()
                else:
                    additional_fleet_cost_no_ivc=0

                # Identify infeasible blocks
                infeasible_blocks = block_general[block_general["range_tracking"].apply(lambda rt: any(x < 0 for x in rt) if rt else False)]["block_id"].tolist()
                blocks_below_critical = block_general[block_general["range_tracking"].apply(lambda rt: any(x < critical_range for x in rt) if rt else False)]["block_id"].tolist()
            
                initial_num_infeasible_blocks=len(infeasible_blocks)

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
                    
                    block_general['additional_buses'] = block_general.apply(calculate_additional_buses, axis=1, args=(bus_range,))

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
                    counter_track=0
                    while len(infeasible_blocks)>0:
                        counter_track+=1
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
                
                        # Initialize an empty Counter
                        shape_counts = Counter()
                
                        # Iterate over the 'flattened_counts' column and update the Counter
                        for counts_dict in filtered_blocks['flattened_counts']:
                            shape_counts.update(counts_dict)
                        # Sort the shape IDs by count in descending order
                        sorted_shape_counts = sorted(shape_counts.items(), key=lambda x: x[1], reverse=True)
                        track_shape_ids = [shape for shape, _ in sorted_shape_counts]
                         
                        for selected_shape_id in track_shape_ids:
                            filtered_blocks['track_shape_count'] = filtered_blocks['flattened_counts'].apply(
                                lambda row: row[selected_shape_id] if selected_shape_id in row else 0
                            )
                            filtered_blocks['estimate_length-per_shape']=filtered_blocks["estimate_required_length"]/filtered_blocks["track_shape_count"]
                            new_track_shape, new_track_shapeids,new_distance=find_best_matching_segment(shapes, selected_shape_id, max(filtered_blocks.loc[filtered_blocks['track_shape_count']>0,'estimate_length-per_shape'])*1609, filtered_blocks,wireless_track_shape)
                            if new_distance>0:
                                break
                                           
                        wireless_track_length= wireless_track_length+new_distance/1609
                        if new_track_shape is not None:
                            if not wireless_track_shape.empty and wireless_track_shape["shape_id"].iloc[-1]==new_track_shape["shape_id"].iloc[0] and wireless_track_shape["target_shape_pt_sequence"].iloc[-1]==new_track_shape["target_shape_pt_sequence"].iloc[0]-1:
                                counter_track-=1
                            new_track_shape["counter"] = counter_track  # Assign the same value to all rows
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
                        lambda row: compute_range_tracking_lane(row["distances_list"], row["time_gaps"], row["end_id_list"], row["trips_by_route"], row["avg_speed_list"],bus_range, charging_power,dynamic_wireless_charging_power, energy_usage, min_stoppage_time, top_end_stop_ids, wireless_track_shapeids,wireless_track_shape,shapes) if row["total_distance_miles"] > bus_range and id in row["end_id_list"] else row["range_tracking"],
                        axis=1
                    )
                    
                    block_general['additional_buses'] = block_general.apply(calculate_additional_buses, axis=1, args=(bus_range,))

                    infeasible_blocks_copy = block_general[block_general["range_tracking"].apply(lambda rt: any(x < 0 for x in rt) if rt else False)]["block_id"].tolist()
                    
                    if len(infeasible_blocks_copy) > len(infeasible_blocks):
                        top_end_stop_ids.append(id)  # Add back if necessary

                # Apply function to blocks with total_distance_miles > Bus_range
                block_general["range_tracking"] = block_general.apply(
                    lambda row: compute_range_tracking_lane(row["distances_list"], row["time_gaps"], row["end_id_list"], row["trips_by_route"], row["avg_speed_list"],bus_range, charging_power,dynamic_wireless_charging_power, energy_usage, min_stoppage_time, top_end_stop_ids, wireless_track_shapeids,wireless_track_shape,shapes),
                    axis=1
                )

                block_general['additional_buses'] = block_general.apply(calculate_additional_buses, axis=1, args=(bus_range,))
                
                # Identify infeasible block_ids where any range_tracking value is negative
                infeasible_blocks = block_general[block_general["range_tracking"].apply(lambda rt: any(x < 0 for x in rt) if rt else False)]["block_id"].tolist()

                # Identify block_ids where any range_tracking value is below critical_range
                blocks_below_critical = block_general[block_general["range_tracking"].apply(lambda rt: any(x < critical_range for x in rt) if rt else False)]["block_id"].tolist() 

                                
                # Get proposed charging locations
                selected_stops = pd.DataFrame(top_end_stop_ids, columns=["stop_id"])
                proposed_locations = selected_stops.merge(stops[["stop_id", "stop_lat", "stop_lon"]], on="stop_id", how="left")
                proposed_locations = proposed_locations.drop_duplicates().reset_index(drop=True)
                
                final_number_infeasible_blocks=len(infeasible_blocks)
                if toggle_value_cost==True:
                    if min_stoppage_time<1:
                        bus_reciever_cost=(initial_num_infeasible_blocks-final_number_infeasible_blocks)*bus_coil_cost
                    else:
                        bus_reciever_cost=0

                    total_stationary_cost=len(proposed_locations)*stationary_charger_cost
                    total_dynamic_cost=wireless_track_length*dynamic_charger_cost

                    additinal_fleet_cost=bus_cost*block_general['additional_buses'].sum()
                else:
                    bus_reciever_cost=0
                    total_stationary_cost=0
                    total_dynamic_cost=0
                    additinal_fleet_cost=0


                # Calculate map center
                center_lat = shapes['shape_pt_lat'].mean()
                center_lon = shapes['shape_pt_lon'].mean()
                
                # Create map
                maptrips=trips[trips["route_id"].isin(block_general["routes_in_block_id"].explode())]
                bus_map = create_bus_electrification_map(
                    shapes,
                    routes,
                    maptrips,
                    proposed_locations,
                    wireless_track_shape,
                    center_lat,
                    center_lon
                )
                
                # Store results
                st.session_state["map"] = bus_map
                st.session_state["routes_count"] = len(trips['route_id'].unique())
                st.session_state["stops_count"] = stops['stop_id'].nunique()
                st.session_state["blocks_count"] = num_blocks_total
                st.session_state["add_flag"]=add_flag
                st.session_state["feasible_blocks_count"]=len(block_general)-len(infeasible_blocks)
                st.session_state["infeasible_blocks_count"] = len(infeasible_blocks)
                st.session_state["critical_blocks_count"] = len(blocks_below_critical)
                st.session_state["minimum_range_without_charger"] = min_range_without_charging
                st.session_state["num_locs"] = len(proposed_locations)
                st.session_state["wirelesslength"]= round(wireless_track_length,1)
                st.session_state["output"]=wireless_track_shape.to_csv(index=False).encode("utf-8")

                st.session_state["Agency_name"]=selected_agency
                st.session_state["block_info"]=block_general
                st.session_state["inf_block_info"]=infeasible_blocks
                
                st.session_state["initial_num_infeasible_blocks"]=initial_num_infeasible_blocks
                st.session_state["final_number_infeasible_blocks"]=final_number_infeasible_blocks
                st.session_state["additional_fleet_cost_no_ivc"]=additional_fleet_cost_no_ivc
                st.session_state["additinal_fleet_cost"]=additinal_fleet_cost
                st.session_state["total_stationary_cost"]=total_stationary_cost
                st.session_state["bus_reciever_cost"]=bus_reciever_cost
                st.session_state["total_dynamic_cost"]=total_dynamic_cost

                msg1.empty()
                msg2.empty()
                msg3.empty()
                flag_done=1

                routes_image_bytes, charger_image_bytes, full_image_bytes = generate_route_charger_maps(shapes, maptrips, proposed_locations, wireless_track_shape, center_lat, center_lon)
                               
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
            st.metric(label="Total Blocks", value=st.session_state['blocks_count'], help="A block is a sequence of consecutive routes assigned to a single vehicle")
        
        # Display additional information
        st.subheader("Analysis Results")
        if st.session_state['add_flag']==0:
            st.write(f"With the selected configurations:")
            if st.session_state['infeasible_blocks_count']==0:
                st.success("✅ All blocks can be electrified.")
            else:
                if st.session_state['infeasible_blocks_count'] ==1:
                    st.success(f"✅ {st.session_state['blocks_count']-st.session_state['infeasible_blocks_count']} blocks can be electrified.")
                    st.error(f"❌ {st.session_state['infeasible_blocks_count']} block cannot be electrified without adjusting the block schedule or assignment.")
                else:
                    st.success(f"✅ {st.session_state['blocks_count']-st.session_state['infeasible_blocks_count']} blocks can be electrified.")
                    st.error(f"❌ {st.session_state['infeasible_blocks_count']} blocks cannot be electrified without adjusting the block schedule or assignment.")
        else:
            st.write(f"The application of the filters yielded {st.session_state['feasible_blocks_count']+st.session_state['infeasible_blocks_count']} block(s). With the selected configurations:")
            if st.session_state['infeasible_blocks_count']==0:
                st.success("✅ All selected block(s) can be electrified.")
            else:
                if st.session_state['infeasible_blocks_count'] ==1:
                    st.success(f"✅ {st.session_state['feasible_blocks_count']} of selected block(s) can be electrified.")
                    st.error(f"❌ {st.session_state['infeasible_blocks_count']} seleted block cannot be electrified without adjusting the block schedule or assignment.")
                else:
                    st.success(f"✅ {st.session_state['feasible_blocks_count']} of selected block(s) can be electrified.")
                    st.error(f"❌ {st.session_state['infeasible_blocks_count']} of selected blocks cannot be electrified without adjusting the block schedule or assignment.")
                    
        if st.session_state['num_locs'] >1: 
            st.write(f"- {st.session_state['num_locs']} stationary charging locations are needed.")
        elif st.session_state['num_locs'] ==1: 
            st.write(f"- {st.session_state['num_locs']} stationary charging location is needed.")
        else:
            st.write(f"- No stationary charging location is needed.")
            
        st.write(f"- The total length of the dynamic wireless track is {st.session_state['wirelesslength']} miles.")

        st.write(f"{st.session_state['inf_block_info']}")
        # Display map
        st.subheader("Route Map with Proposed Charging Locations")
        st_folium(st.session_state["map"], width=1200, height=800, returned_objects=[])

       
        # if st.session_state["output"] is not None:
        #     st.download_button(
        #         label="Download CSV",
        #         data=st.session_state["output"],
        #         file_name="output.csv",
        #         mime="text/csv"
        #     )

    if flag_done==1 and toggle_value_cost==True:
        if st.session_state.get("initial_num_infeasible_blocks") == 0:
            categories = ["All Blocks Feasible with No Chargers"]
            Additional_fleet=np.array([st.session_state.get("additional_fleet_cost_no_ivc")])/1e6
            Stationary_charger=np.array([0])/1e6
            Bus_reciever=np.array([0])/1e6
            dynamic_charger=np.array([0])/1e6

        elif st.session_state['final_number_infeasible_blocks']==0:
            categories = [f"All Blocks Feasible with {st.session_state.get("additional_fleet_cost_no_ivc")/bus_cost} Additional Buses & No Chargers","All Blocks Feasible with Selected Chargers"]
            Additional_fleet=np.array([st.session_state.get('additional_fleet_cost_no_ivc'),st.session_state.get('additinal_fleet_cost')])/1e6
            Stationary_charger=np.array([0,st.session_state.get('total_stationary_cost')])/1e6
            Bus_reciever=np.array([0,st.session_state.get('bus_reciever_cost')])/1e6
            dynamic_charger=np.array([0,st.session_state.get('total_dynamic_cost')])/1e6

        else:
            categories = [f"All Blocks Feasible with {st.session_state.get('additional_fleet_cost_no_ivc')/bus_cost} Additional Buses & No Chargers",f"Existing Fleet & Selected Chargers with {st.session_state.get('final_number_infeasible_blocks')} Infeasible Blocks", f"All Blocks Feasible with {st.session_state.get('additinal_fleet_cost')/bus_cost} Buses & Selected Chargers"]
            Additional_fleet=np.array([st.session_state.get('additional_fleet_cost_no_ivc'),0,st.session_state.get('additinal_fleet_cost')])/1e6
            Stationary_charger=np.array([0,st.session_state.get('total_stationary_cost'),st.session_state.get('total_stationary_cost')])/1e6
            Bus_reciever=np.array([0,st.session_state.get('bus_reciever_cost'),st.session_state.get('bus_reciever_cost')])/1e6
            dynamic_charger=np.array([0,st.session_state.get('total_dynamic_cost'),st.session_state.get('total_dynamic_cost')])/1e6

        # X positions for bars
        x = np.arange(len(categories))
        bar_width = 0.5    

        # Create the stacked bars
        fig, ax = plt.subplots(figsize=(8,6))

        ax.bar(x, Additional_fleet, width=bar_width, label='Additional Fleet', color='green')
        ax.bar(x, Stationary_charger, width=bar_width, bottom=Additional_fleet, label='Stationary Charging Stations', color='orange')
        ax.bar(x, Bus_reciever, width=bar_width, bottom=np.array(Additional_fleet) + np.array(Stationary_charger), label='Bus Reciever Coil', color='blue')
        if dynamic_wireless_charging_power>0:
            ax.bar(x, dynamic_charger, width=bar_width, bottom=np.array(Additional_fleet) + np.array(Stationary_charger)+np.array(Bus_reciever), label='Dyanmic Charger Track', color='cyan')

        # Labels & legend
        ax.set_ylabel('Total Cost (million $)')
        ax.set_xticks(x)
        ax.set_xticklabels(['\n'.join(textwrap.wrap(label, 30)) for label in categories])
        ax.legend()
        st.pyplot(fig)
    if flag_done==1 and toggle_value_cost==True:    
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            fig.savefig(tmpfile.name, bbox_inches='tight')
            econ_figure_gen = tmpfile.name
    else:
        econ_figure_gen = None

    if flag_done==1:
        report_inputs = {"transit_agency_name": st.session_state["Agency_name"],
            "bus_range": bus_range,
            "stationary_power": charging_power,
            "stationary_setup_time_hr": min_stoppage_time,
            "dynamic_power": dynamic_wireless_charging_power,
            "energy_usage":energy_usage,
        }
        if toggle_value_cost==True:
            report_inputs.update({"bus_price": bus_cost,
                "stationary_charger_cost": stationary_charger_cost,
                "coil_install_cost": bus_coil_cost,
                "dynamic_track_cost": dynamic_charger_cost,
            })
        report_outputs = {
            "total_routes": st.session_state["routes_count"],
            "total_stops": st.session_state["stops_count"],
            "total_blocks": st.session_state["blocks_count"],
            "electrifiable_blocks": st.session_state["feasible_blocks_count"],
            "non_electrifiable_blocks": st.session_state["infeasible_blocks_count"],
            "infeasible_block_ids": st.session_state["inf_block_info"],
            "num_stationary_chargers": st.session_state["num_locs"],
            "dynamic_lane_length": st.session_state["wirelesslength"],
            "block_general":st.session_state["block_info"],
        }
        if toggle_value_cost==True:
            report_outputs.update({
                "categories":categories,
                "additional_fleet_cost_no_ivc":st.session_state["additional_fleet_cost_no_ivc"],
                "additinal_fleet_cost":st.session_state["additinal_fleet_cost"],
        })
    
        pdf_buffer = generate_transit_report(
            inputs=report_inputs,
            outputs=report_outputs,
            map_image_path=routes_image_bytes,
            econ_toggle=toggle_value_cost,
            econ_figure_path=econ_figure_gen,
            agency_name=st.session_state['Agency_name'],
            title_image_path="bus_title_image.png",
            charger_image_path=charger_image_bytes,
            routencharger_image_path=full_image_bytes
        )
        st.write(f"You can download a PDF report for your analysis by clicking on: ")
        st.download_button(
            label="📥 Report",
            data=pdf_buffer,
            file_name="transit_report.pdf",
            mime="application/pdf"
        )

if __name__ == "__main__":
    main()

















