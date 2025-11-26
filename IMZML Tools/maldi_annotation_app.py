"""
MALDI Tissue Annotation Tool
============================

A Streamlit-based interactive application for annotating tissue regions in MALDI-MSI data.
This tool allows users to load h5ad files, visualize the data, and manually draw polygons
to identify tissue regions, which can then be saved to a new file.

Requirements:
------------
Required Python packages are listed in requirements.txt:
- streamlit
- plotly
- numpy
- pandas
- anndata
- matplotlib
- streamlit-plotly-events

Running the Application:
-----------------------
1. Ensure you have activated the appropriate environment:

2. Run the Streamlit application:
   streamlit run maldi_annotation_app.py

3. The application should open in your default web browser automatically.
   If not, visit: http://localhost:8501

Usage:
------
1. Enter the path to your MALDI h5ad file or select it from a directory
2. Enable drawing mode and click on the plot to define a polygon around tissue regions
3. Click "Create Mask" to generate the tissue mask
4. Save the annotated data with the "Save Annotated Data" button

"""


import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import anndata as ad
from matplotlib.path import Path
import os
import glob
from streamlit_plotly_events import plotly_events

# Set page config with fixed layout
st.set_page_config(
    page_title="MALDI Tissue Annotation Tool", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure session state variables exist
if 'polygon_points' not in st.session_state:
    st.session_state.polygon_points = pd.DataFrame(columns=['x', 'y'])
if 'adata' not in st.session_state:
    st.session_state.adata = None
if 'in_tissue' not in st.session_state:
    st.session_state.in_tissue = None
if 'show_preview' not in st.session_state:
    st.session_state.show_preview = False
if 'last_file_path' not in st.session_state:
    st.session_state.last_file_path = ""
if 'coords' not in st.session_state:
    st.session_state.coords = None
if 'intensity_values' not in st.session_state:
    st.session_state.intensity_values = None
if 'drawing_mode' not in st.session_state:
    st.session_state.drawing_mode = False
if 'has_loaded' not in st.session_state:
    st.session_state.has_loaded = False

# Cache the data loading function
@st.cache_data
def load_anndata(file_path):
    """Load AnnData file with caching for better performance"""
    return ad.read_h5ad(file_path)

# Function to add a new point to the polygon
def add_point(x, y):
    """Add a new point to the polygon"""
    new_point = pd.DataFrame({'x': [x], 'y': [y]})
    st.session_state.polygon_points = pd.concat(
        [st.session_state.polygon_points, new_point], 
        ignore_index=True
    )

# Set up main page layout - title
st.title("MALDI Tissue Annotation Tool")

# Fixed layout with sidebar and main content
sidebar = st.sidebar
main_container = st.container()

# Sidebar content
with sidebar:
    st.header("Data Input")
    input_method = st.radio("Select input method:", ["Enter File Path", "Select From Directory"])
    
    if input_method == "Enter File Path":
        file_path = st.text_input("Enter path to h5ad file:", "")
        
        # Path examples help text
        st.markdown("""
        **Example paths:**
        - Windows: `C:\\Data\\maldi_data.h5ad`
        - Mac/Linux: `/home/user/data/maldi_data.h5ad`
        """)
    else:
        # Directory selection
        directory = st.text_input("Enter directory containing h5ad files:", "")
        
        if directory and os.path.exists(directory):
            h5ad_files = glob.glob(os.path.join(directory, "*.h5ad"))
            
            if h5ad_files:
                selected_file = st.selectbox("Select h5ad file:", h5ad_files)
                file_path = selected_file
            else:
                st.error(f"No h5ad files found in {directory}")
                file_path = ""
        else:
            if directory:
                st.error(f"Directory does not exist: {directory}")
            file_path = ""
    
    # Load file button - separate to prevent auto-reloading
    if file_path and os.path.exists(file_path):
        load_button = st.button("Load File", key="load_file_btn", use_container_width=True)
        if load_button:
            try:
                with st.spinner(f"Loading data from {file_path}..."):
                    st.session_state.adata = load_anndata(file_path)
                    st.session_state.last_file_path = file_path
                    st.session_state.polygon_points = pd.DataFrame(columns=['x', 'y'])
                    st.session_state.in_tissue = None
                    st.session_state.show_preview = False
                    
                    # Extract coordinates
                    if 'spatial' in st.session_state.adata.obsm:
                        st.session_state.coords = np.array(st.session_state.adata.obsm['spatial'])
                    elif 'x' in st.session_state.adata.obs and 'y' in st.session_state.adata.obs:
                        st.session_state.coords = np.array(st.session_state.adata.obs[['x', 'y']].values)
                    
                    # Extract visualization values
                    if 'TIC' in st.session_state.adata.obs.columns:
                        tic_values = st.session_state.adata.obs['TIC']
                        st.session_state.intensity_values = np.array(tic_values)
                    else:
                        if hasattr(st.session_state.adata.X, 'toarray'):
                            st.session_state.intensity_values = np.array(np.mean(st.session_state.adata.X.toarray(), axis=1))
                        else:
                            st.session_state.intensity_values = np.array(np.mean(st.session_state.adata.X, axis=1))
                    
                    st.session_state.has_loaded = True
                    st.success(f"Successfully loaded: {os.path.basename(file_path)}")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    # Visualization options - only if data is loaded
    if st.session_state.adata is not None:
        st.header("Data Information")
        st.text(f"Shape: {st.session_state.adata.shape}")
        
        # Display observations
        obs_columns = list(st.session_state.adata.obs.columns)
        if len(obs_columns) > 5:
            st.text(f"Observations: {obs_columns[:5] + ['...']}")
        else:
            st.text(f"Observations: {obs_columns}")
        
        # Visualization settings
        st.header("Visualization Options")
        
        # Intensity visualization method selection
        st.subheader("Intensity Display Method")
        intensity_type = st.radio(
            "Select intensity data to display:",
            ["TIC (Total Ion Current)", "Specific m/z Value"],
            index=0
        )
        
        # Based on selection, provide appropriate options
        if intensity_type == "TIC (Total Ion Current)":
            # Use TIC values if available
            if 'TIC' in st.session_state.adata.obs.columns:
                st.session_state.intensity_values = np.array(st.session_state.adata.obs['TIC'])
                st.info("Using TIC values for visualization")
            else:
                st.warning("TIC column not found. Using average intensity instead.")
                if hasattr(st.session_state.adata.X, 'toarray'):
                    st.session_state.intensity_values = np.array(np.mean(st.session_state.adata.X.toarray(), axis=1))
                else:
                    st.session_state.intensity_values = np.array(np.mean(st.session_state.adata.X, axis=1))
        
        elif intensity_type == "Specific m/z Value":
            # Get all m/z values
            mz_values = np.array([float(mz) for mz in st.session_state.adata.var_names])
            
            # Selection method
            mz_selection_method = st.radio(
                "How would you like to select m/z?",
                ["Range Selection", "Direct Input"],
                index=0
            )
            
            if mz_selection_method == "Range Selection":
                # Calculate m/z range
                min_mz = float(np.min(mz_values))
                max_mz = float(np.max(mz_values))
                
                # Create 10 ranges
                ranges = []
                step = (max_mz - min_mz) / 10
                
                for i in range(10):
                    start = min_mz + i * step
                    end = min_mz + (i+1) * step
                    # Count values in this range
                    count = np.sum((mz_values >= start) & (mz_values <= end))
                    ranges.append(f"{start:.2f} - {end:.2f} ({count} values)")
                
                # Select range
                selected_range = st.selectbox("Select m/z range:", ranges)
                
                # Extract range values
                range_parts = selected_range.split(" (")[0].split(" - ")
                start_val = float(range_parts[0])
                end_val = float(range_parts[1])
                
                # Get m/z values in this range
                filtered_mz = mz_values[(mz_values >= start_val) & (mz_values <= end_val)]
                
                # Convert to list of strings for selectbox
                filtered_mz_str = [f"{mz:.4f}" for mz in filtered_mz]
                
                # Select specific m/z from the range
                selected_mz_str = st.selectbox("Select specific m/z value:", filtered_mz_str)
                selected_mz = float(selected_mz_str)
                
            else:  # Direct Input
                # Get range for number input
                min_mz = float(np.min(mz_values))
                max_mz = float(np.max(mz_values))
                
                # Allow direct input
                mz_input = st.number_input(
                    "Enter m/z value:", 
                    min_value=min_mz,
                    max_value=max_mz,
                    value=(min_mz + max_mz)/2,  # Default to middle value
                    format="%.4f"
                )
                
                # Find closest m/z value
                closest_idx = np.abs(mz_values - mz_input).argmin()
                selected_mz = mz_values[closest_idx]
                
                if abs(selected_mz - mz_input) > 0.0001:
                    st.info(f"Using closest available m/z: {selected_mz:.4f}")
            
            # Get intensity for selected m/z value
            mz_idx = np.where(np.isclose(mz_values, selected_mz))[0][0]
            
            if hasattr(st.session_state.adata.X, 'toarray'):
                st.session_state.intensity_values = st.session_state.adata.X[:, mz_idx].toarray().flatten()
            else:
                st.session_state.intensity_values = st.session_state.adata.X[:, mz_idx].flatten()
            
            st.success(f"Using m/z {selected_mz:.4f} for visualization")
        
        # Point size and opacity
        point_size = st.slider("Point Size", 1, 20, 5, key="point_size")
        alpha = st.slider("Transparency", 0.1, 1.0, 1.0, key="alpha")
        
        # Colormap selection
        colormap = st.selectbox(
            "Colormap", 
            ["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Jet"], 
            index=0,
            key="colormap"
        )
        
        # Percentile for color scaling
        percentile = st.slider(
            "Clip Values at Percentile", 
            50, 100, 95, 
            key="percentile"
        )
        
        # Calculate color scaling value
        if st.session_state.intensity_values is not None:
            if percentile < 100:
                vmax = float(np.percentile(st.session_state.intensity_values, percentile))
            else:
                vmax = float(np.max(st.session_state.intensity_values))
        else:
            vmax = 100.0

# Main content area
with main_container:
    # Only render visualization area if data is loaded
    if st.session_state.has_loaded and st.session_state.coords is not None:
        # Split into 70% visualization / 30% tools
        col1, col2 = st.columns([7, 3])
        
        with col1:
            # Visualization area
            st.subheader("Data Visualization")
            
            # Draw mode toggle
            drawing_mode = st.checkbox("Enable Drawing Mode", value=st.session_state.drawing_mode)
            
            # Update session state if changed
            if drawing_mode != st.session_state.drawing_mode:
                st.session_state.drawing_mode = drawing_mode
            
            # Drawing instructions
            if st.session_state.drawing_mode:
                st.markdown("👆 **Click directly on the plot to add points to your polygon.**")
            else:
                st.markdown("*Enable drawing mode to add points to your polygon*")
            
            # Calculate data range and aspect ratio
            x_min, x_max = np.min(st.session_state.coords[:, 0]), np.max(st.session_state.coords[:, 0])
            y_min, y_max = np.min(st.session_state.coords[:, 1]), np.max(st.session_state.coords[:, 1])
            x_range = x_max - x_min
            y_range = y_max - y_min
            
            # Calculate dimensions with the correct aspect ratio based on data
            base_size = 800  # Base width
            width = base_size
            height = base_size * (y_range / x_range)  # Preserve aspect ratio
            # Apply reasonable limits to height
            height = min(max(height, 400), 1000)
            
            # Create Plotly figure
            fig = go.Figure()
            
            # Add data points
            fig.add_trace(go.Scatter(
                x=st.session_state.coords[:, 0], 
                y=st.session_state.coords[:, 1],
                mode='markers',
                marker=dict(
                    size=point_size,
                    color=st.session_state.intensity_values,
                    colorscale=colormap.lower(),
                    opacity=alpha,
                    cmin=0,
                    cmax=vmax,
                    colorbar=dict(
                        title="Intensity",
                        x=1.05,  # Position colorbar outside of plot
                        xpad=10  # Add padding 
                    )
                ),
                hoverinfo='none',
                name='MALDI Data'
            ))
            
            # Add existing polygon points if available
            if not st.session_state.polygon_points.empty:
                # Get the polygon points
                polygon_pts = st.session_state.polygon_points[['x', 'y']].values
                
                # Add line connecting points
                x_poly = polygon_pts[:, 0].tolist()
                y_poly = polygon_pts[:, 1].tolist()
                
                # Close the polygon if there are enough points
                if len(x_poly) > 2:
                    x_poly.append(x_poly[0])
                    y_poly.append(y_poly[0])
                
                # Add to figure
                fig.add_trace(go.Scatter(
                    x=x_poly,
                    y=y_poly,
                    mode='lines+markers',
                    marker=dict(size=10, color='red'),
                    line=dict(color='red', width=2),
                    name='Polygon'
                ))
            
            # Fixed layout parameters with aspect ratio and improved positioning
            fig.update_layout(
                title=dict(
                    text=f"MALDI Data - {os.path.basename(st.session_state.last_file_path)}",
                    y=0.96,  # Move title down slightly to avoid toolbar
                    x=0.5,
                    xanchor='center',
                    yanchor='top'
                ),
                width=int(width),
                height=int(height),
                autosize=False,
                hovermode='closest',
                margin=dict(l=50, r=80, t=60, b=50),
                legend=dict(
                    x=0.01,  # Position legend at top-left
                    y=0.99,
                    xanchor='left',
                    yanchor='top'
                ),
                plot_bgcolor='white',  # White background
                paper_bgcolor='white'  # White surrounding area
            )
            
            # Add padding to the coordinate ranges
            x_padding = x_range * 0.05
            y_padding = y_range * 0.05
            
            # Improved axis formatting
            fig.update_xaxes(
                range=[x_min - x_padding, x_max + x_padding],
                constrain="domain",
                title=dict(
                    text="X Coordinate",
                    standoff=20
                ),
                tickmode='auto',
                ticks='outside',  # Place ticks outside the axes
                ticklen=5,        # Length of tick marks
                tickwidth=1,      # Width of tick marks
                automargin=True,  # Automatically adjust margins for labels
                mirror=True,      # Show axis line on opposite side
                showline=True,    # Show axis line
                linewidth=1,      # Width of axis line
                linecolor='black' # Color of axis line
            )
            
            fig.update_yaxes(
                range=[y_min - y_padding, y_max + y_padding],
                scaleanchor="x",
                scaleratio=1,
                title="Y Coordinate",
                tickmode='auto',
                ticks='outside',
                ticklen=5, 
                tickwidth=1,
                automargin=True,
                mirror=True,
                showline=True,
                linewidth=1,
                linecolor='black'
            )
            
            # Display the plot with click event handling if in drawing mode
            if st.session_state.drawing_mode:
                # Use streamlit_plotly_events to capture clicks
                selected_points = plotly_events(
                    fig, 
                    click_event=True,
                    override_height=f"{int(height)}px",
                    override_width="100%"
                )
                
                # Process any clicks to add points
                if selected_points:
                    for point in selected_points:
                        x, y = point["x"], point["y"]
                        add_point(x, y)
                    st.rerun()  # Update the UI after adding points
            else:
                # Regular display without click events
                st.plotly_chart(fig, use_container_width=True)
            
            # Instructions for drawing with clicks
            with st.expander("How to draw a polygon"):
                st.markdown("""
                1. **Enable Drawing Mode** above
                2. Click directly on the plot to add points to your polygon
                3. Use the "Remove Last Point" button to correct mistakes
                4. When finished, click "Create Mask" to generate the tissue mask
                5. You can also add points manually using the "Manual Point Entry" section
                """)
        
        with col2:
            # Annotation tools column
            st.subheader("Annotation Tools")
            
            # Current polygon points with fixed height
            st.markdown("#### Current Polygon Points")
            st.dataframe(st.session_state.polygon_points, height=150)
            
            # Manual point entry (backup method)
            st.markdown("#### Manual Point Entry (Optional)")
            
            # Manual coordinate inputs
            col_x, col_y, col_add = st.columns([1, 1, 1])
            with col_x:
                # Default values that won't cause errors
                default_x = float(np.mean(st.session_state.coords[:, 0])) if st.session_state.coords is not None else 0.0
                manual_x = st.number_input("X", value=default_x, key="manual_x")
            with col_y:
                default_y = float(np.mean(st.session_state.coords[:, 1])) if st.session_state.coords is not None else 0.0
                manual_y = st.number_input("Y", value=default_y, key="manual_y")
            with col_add:
                add_btn = st.button("Add Point", key="add_point_btn")
                if add_btn:
                    add_point(manual_x, manual_y)
                    st.rerun()
            
            # Editing tools
            st.markdown("#### Editing Tools")
            col_clear, col_undo = st.columns(2)
            with col_clear:
                if st.button("Clear All Points", key="clear_btn", use_container_width=True):
                    st.session_state.polygon_points = pd.DataFrame(columns=['x', 'y'])
                    st.session_state.in_tissue = None
                    st.session_state.show_preview = False
                    st.rerun()
            
            with col_undo:
                if st.button("Remove Last Point", key="undo_btn", use_container_width=True) and not st.session_state.polygon_points.empty:
                    st.session_state.polygon_points = st.session_state.polygon_points.iloc[:-1]
                    st.rerun()
            
            # Create mask button
            st.markdown("#### Create Tissue Mask")
            create_mask_btn = st.button("Create Mask", key="create_mask_btn", use_container_width=True)
            if create_mask_btn:
                if len(st.session_state.polygon_points) > 2:
                    vertices = st.session_state.polygon_points[['x', 'y']].values
                    path = Path(vertices)
                    st.session_state.in_tissue = path.contains_points(st.session_state.coords)
                    st.session_state.show_preview = True
                    st.rerun()
                else:
                    st.warning("Need at least 3 points to create a polygon")
    
    # Tissue mask preview section - only shown when a mask has been created
    if st.session_state.show_preview and st.session_state.in_tissue is not None:
        # Use an expander to prevent layout shifts
        with st.expander("Tissue Mask Preview", expanded=True):
            preview_col1, preview_col2 = st.columns([1, 1])
            
            with preview_col1:
                # Calculate coordinate ranges for preview (same as main plot)
                x_min, x_max = np.min(st.session_state.coords[:, 0]), np.max(st.session_state.coords[:, 0])
                y_min, y_max = np.min(st.session_state.coords[:, 1]), np.max(st.session_state.coords[:, 1])
                x_range = x_max - x_min
                y_range = y_max - y_min
                
                # Calculate preview dimensions with correct aspect ratio
                preview_base = 500  # Base size for preview
                preview_width = preview_base
                preview_height = preview_base * (y_range / x_range)
                # Apply reasonable limits to preview height
                preview_height = min(max(preview_height, 300), 700)

                # Create preview figure
                mask_fig = go.Figure()
                
                # Background points
                mask_fig.add_trace(go.Scatter(
                    x=st.session_state.coords[~st.session_state.in_tissue, 0],
                    y=st.session_state.coords[~st.session_state.in_tissue, 1],
                    mode='markers',
                    marker=dict(size=point_size, color='gray', opacity=0.3),
                    name='Background'
                ))
                
                # Tissue points
                mask_fig.add_trace(go.Scatter(
                    x=st.session_state.coords[st.session_state.in_tissue, 0],
                    y=st.session_state.coords[st.session_state.in_tissue, 1],
                    mode='markers',
                    marker=dict(size=point_size, color='blue', opacity=0.7),
                    name='Tissue'
                ))
                
                # Polygon outline
                vertices = st.session_state.polygon_points[['x', 'y']].values
                x_poly = np.append(vertices[:, 0], vertices[0, 0])
                y_poly = np.append(vertices[:, 1], vertices[0, 1])
                
                mask_fig.add_trace(go.Scatter(
                    x=x_poly,
                    y=y_poly,
                    mode='lines',
                    line=dict(color='red', width=2),
                    name='Polygon'
                ))
                
                # Add padding to the coordinate ranges
                x_padding = x_range * 0.05
                y_padding = y_range * 0.05
                
                # Improved preview layout with correct aspect ratio
                mask_fig.update_layout(
                    title=dict(
                        text="Tissue Mask Preview",
                        y=0.95,
                        x=0.5,
                        xanchor='center',
                        yanchor='top'
                    ),
                    width=int(preview_width),
                    height=int(preview_height),
                    autosize=False,
                    margin=dict(l=50, r=50, t=60, b=50),
                    legend=dict(
                        x=0.01, 
                        y=0.99,
                        xanchor='left',
                        yanchor='top'
                    ),
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                )
                
                # Ensure the preview uses the same axis range and settings as the main plot
                mask_fig.update_xaxes(
                    range=[x_min - x_padding, x_max + x_padding],
                    constrain="domain",
                    title=dict(
                        text="X Coordinate",
                        standoff=20
                    ),
                    tickmode='auto',
                    ticks='outside',
                    ticklen=5,
                    automargin=True,
                    mirror=True,
                    showline=True,
                    linewidth=1,
                    linecolor='black'
                )
                
                mask_fig.update_yaxes(
                    range=[y_min - y_padding, y_max + y_padding],
                    scaleanchor="x",  # Key setting to maintain aspect ratio
                    scaleratio=1,     # 1:1 scaling between x and y
                    tickmode='auto',
                    ticks='outside',
                    ticklen=5,
                    automargin=True,
                    mirror=True,
                    showline=True,
                    linewidth=1,
                    linecolor='black',
                    title="Y Coordinate"
                )
                
                st.plotly_chart(mask_fig)
            
            with preview_col2:
                # Mask statistics
                st.markdown("#### Mask Statistics")
                total_points = len(st.session_state.in_tissue)
                tissue_points = int(np.sum(st.session_state.in_tissue))
                background_points = total_points - tissue_points
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Points", total_points)
                    st.metric("Tissue Points", tissue_points)
                with col2:
                    st.metric("Background Points", background_points)
                    st.metric("Tissue Percentage", f"{tissue_points/total_points*100:.2f}%")
                
                # Save annotated data
                st.markdown("#### Save Annotated Data")
                output_filename = st.text_input(
                    "Output Filename", 
                    value=os.path.join(
                        os.path.dirname(st.session_state.last_file_path),
                        os.path.basename(st.session_state.last_file_path).replace('.h5ad', '_annotated.h5ad')
                    ),
                    key="output_filename"
                )
                
                if st.button("Save Annotated Data", key="save_btn", use_container_width=True):
                    # Add tissue mask to the data
                    st.session_state.adata.obs['in_tissue'] = st.session_state.in_tissue
                    
                    try:
                        # Save the annotated file
                        st.session_state.adata.write(output_filename)
                        st.success(f"Data saved with tissue annotations to: {output_filename}")
                    except Exception as e:
                        st.error(f"Error saving file: {str(e)}")
    else:
        # Initial instructions
        if not st.session_state.has_loaded:
            st.info("Please enter the path to an h5ad file and click 'Load File' to begin")