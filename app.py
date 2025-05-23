# app.py
import os
import fastf1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Setup
os.makedirs('data', exist_ok=True)
fastf1.Cache.enable_cache('data')
st.set_page_config(layout="wide")
st.title("🏎️ F1 Lap Time Analysis & Telemetry Dashboard")

# Sidebar: Select Year
all_years = list(range(2018, 2026))
valid_years = [y for y in all_years if not fastf1.get_event_schedule(y).empty]
year = st.sidebar.selectbox("Select Year", valid_years)

# Get GP schedule
schedule = fastf1.get_event_schedule(year)
if schedule.empty:
    st.warning("No race data available for this year.")
    st.stop()

gp = st.sidebar.selectbox("Select Grand Prix", schedule['EventName'].dropna().unique())
session_type = st.sidebar.selectbox("Session Type", ['Race', 'Qualifying', 'Practice 1', 'Practice 2', 'Practice 3'])

# Map to FastF1 keywords
session_type_map = {
    'Race': 'R',
    'Qualifying': 'Q',
    'Practice 1': 'FP1',
    'Practice 2': 'FP2',
    'Practice 3': 'FP3'
}
session_key = session_type_map[session_type]

# Load session
with st.spinner("Loading session with telemetry..."):
    session = fastf1.get_session(year, gp, session_type)
    session.load(telemetry=True)
    session.laps['LapTimeSeconds'] = session.laps['LapTime'].dt.total_seconds()  # ✅ Add this line

if session.laps.empty:
    st.warning("No lap data available for this session.")
    st.stop()

telemetry_available = bool(session.pos_data)

st.markdown("### 📋 Race Overview")

# Get race schedule for the year
schedule = fastf1.get_event_schedule(year)

# Find the row for the selected Grand Prix
event_row = schedule[schedule['EventName'] == gp]

# Extract date and location safely
if not event_row.empty:
    try:
        event_date = pd.to_datetime(event_row.iloc[0]['Session1Date']).strftime('%B %d, %Y')
    except KeyError:
        event_date = "Unavailable"
    event_location = event_row.iloc[0].get('Location', "Unavailable")
else:
    event_date = "Unavailable"
    event_location = "Unavailable"

# Session name
session_name = session.name

# Fastest lap info
try:
    fastest_lap = session.laps.pick_fastest()
    fastest_driver = fastest_lap['Driver']
    lap_time = fastest_lap['LapTime']
    lap_time_str = f"{lap_time.seconds // 60}:{lap_time.seconds % 60:02}.{int(lap_time.microseconds / 1000):03}"
except Exception:
    fastest_driver = "Unavailable"
    lap_time_str = "Unavailable"

# Display Race Overview
st.markdown(f"**Event:** {gp} ({event_location})  \n"
            f"**Date:** {event_date}  \n"
            f"**Session:** {session_name}  \n"
            f"**Fastest Lap:** {fastest_driver} – {lap_time_str}")

# Driver selection
drivers = session.laps['Driver'].unique().tolist()
driver1 = st.sidebar.selectbox("Driver 1", sorted(drivers), key="driver1_select")
compare_mode = st.sidebar.checkbox("Compare with another driver?")
driver2 = st.sidebar.selectbox("Driver 2", sorted(drivers), index=1 if len(drivers) > 1 else 0, key="driver2_select") if compare_mode else None

# Lap data
def get_driver_data(driver):
    laps = session.laps.pick_driver(driver).reset_index(drop=True)
    laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
    laps = laps.dropna(subset=['LapTimeSeconds', 'Compound'])
    laps['CompoundEncoded'] = laps['Compound'].astype('category').cat.codes
    return laps

laps1 = get_driver_data(driver1)
laps2 = get_driver_data(driver2) if compare_mode else None

# ✅ Handle cases where laps are empty
if laps1.empty:
    st.error(f"No valid lap data available for {driver1}. Please select another driver or session.")
    st.stop()
if compare_mode and laps2.empty:
    st.error(f"No valid lap data available for {driver2}. Please select another driver or session.")
    st.stop()


# Model
def train_model(laps):
    X = laps[['LapNumber', 'CompoundEncoded']]
    y = laps['LapTimeSeconds']
    model = LinearRegression().fit(X, y)
    preds = model.predict(X)
    return model, preds, mean_absolute_error(y, preds), np.sqrt(mean_squared_error(y, preds))

model1, pred1, mae1, rmse1 = train_model(laps1)
if compare_mode:
    model2, pred2, mae2, rmse2 = train_model(laps2)

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Lap Time Prediction", "Sector Comparison", "Track Overlay",
    "Throttle & Gear", "Lap Time Heatmap", "Race Results"
])

# Tab 1: Lap Time Prediction
with tab1:
    st.subheader("📈 Lap Time Prediction")
    st.markdown(
        "This section uses linear regression to model and predict lap times based on lap number and tire compound. "
        "You can visually compare actual lap times versus predicted ones for each driver to identify consistency, strategy shifts, or performance drops over the race session."
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(laps1['LapNumber'], laps1['LapTimeSeconds'], label=f'{driver1} Actual', marker='o')
    ax.plot(laps1['LapNumber'], pred1, label=f'{driver1} Predicted', linestyle='--')
    if compare_mode:
        ax.plot(laps2['LapNumber'], laps2['LapTimeSeconds'], label=f'{driver2} Actual', marker='x')
        ax.plot(laps2['LapNumber'], pred2, label=f'{driver2} Predicted', linestyle='--')
    ax.set_xlabel("Lap Number")
    ax.set_ylabel("Lap Time (s)")
    ax.set_title("Lap Time Prediction")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    st.success(f"{driver1} MAE: {mae1:.2f}s | RMSE: {rmse1:.2f}s")
    if compare_mode:
        st.info(f"{driver2} MAE: {mae2:.2f}s | RMSE: {rmse2:.2f}s")
    st.markdown(
        "**📊 What do MAE and RMSE mean?**\n"
        "- **MAE (Mean Absolute Error):** Average difference between predicted and actual lap times. Lower is better.\n"
        "- **RMSE (Root Mean Squared Error):** Penalizes large errors more. Also lower is better.\n"
        "These metrics help evaluate prediction accuracy."
    )

# Tab 2
with tab2:
    st.subheader("🧩 Sector Time Comparison")
    st.markdown(
        "This section shows how the driver's performance is distributed across the three sectors of the track over the race or session. "
        "You can observe fluctuations in sector times, detect pace evolution, and spot pit stops or errors. Pit laps are highlighted with vertical markers."
    )

    def sector_plot(laps, label):
        fig, ax = plt.subplots(figsize=(10, 5))
        laps['LapNumber'] = laps['LapNumber'].astype(int)
        for sec, color in zip(['Sector1Time', 'Sector2Time', 'Sector3Time'], ['blue', 'orange', 'green']):
            if sec in laps.columns:
                ax.plot(laps['LapNumber'], laps[sec].dt.total_seconds(), label=f'{label} {sec}', color=color)
        if 'PitInTime' in laps.columns:
            pit_laps = laps[laps['PitInTime'].notna()]
            for lap_num in pit_laps['LapNumber']:
                ax.axvline(lap_num, linestyle='--', color='red', alpha=0.3)
                ax.text(lap_num, ax.get_ylim()[1] * 0.95, 'PIT', rotation=90, color='red', fontsize=8, ha='center')
        ax.set_xlabel("Lap Number (actual session lap)")
        ax.set_ylabel("Sector Time (s)")
        ax.set_title(f"Sector Times - {label}")
        ax.legend()
        ax.grid(True)
        return fig

    st.pyplot(sector_plot(laps1, driver1))
    if compare_mode:
        st.pyplot(sector_plot(laps2, driver2))


# Tab 3
with tab3:
    st.subheader("🗺️ Track Telemetry Overlay")
    st.markdown(
        "This section visualizes telemetry from the driver's fastest lap as a colored path on the circuit layout. "
        "The color represents the selected metric (Speed, Throttle, or Gear), helping you understand where on the track the car accelerated, slowed, or shifted gears."
    )

    if not telemetry_available:
        st.warning("⚠️ Telemetry data is not available for this session.")
    else:
        metric = st.selectbox("Color by", ["Speed", "Throttle", "nGear"])
        if compare_mode:
            col1, col2 = st.columns(2)
            for i, (driver, laps) in enumerate(zip([driver1, driver2], [laps1, laps2])):
                with (col1 if i == 0 else col2):
                    st.markdown(f"### {driver}")
                    try:
                        lap = laps.pick_fastest()
                        tel = lap.get_telemetry().add_distance()
                        fig, ax = plt.subplots(figsize=(6, 6))
                        cmap = 'viridis' if i == 0 else 'plasma'
                        scatter = ax.scatter(tel['X'], tel['Y'], c=tel[metric], cmap=cmap, s=5)
                        ax.axis('off')
                        ax.set_title(f"Fastest Lap - {metric}")
                        fig.colorbar(scatter, ax=ax).set_label(metric)
                        st.pyplot(fig)
                    except Exception as e:
                        st.warning(f"Could not load telemetry for {driver}: {e}")
        else:
            try:
                lap = laps1.pick_fastest()
                tel = lap.get_telemetry().add_distance()
                fig, ax = plt.subplots(figsize=(8, 8))
                scatter = ax.scatter(tel['X'], tel['Y'], c=tel[metric], cmap='viridis', s=5)
                ax.axis('off')
                ax.set_title(f"{driver1} - Fastest Lap ({metric})")
                fig.colorbar(scatter, ax=ax).set_label(metric)
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not load telemetry for {driver1}: {e}")

# Tab 4
with tab4:
    st.subheader("📊 Throttle & Gear vs Distance")
    st.markdown(
        "This section plots throttle input and gear selection over the distance of the driver's fastest lap. "
        "It provides insights into driving behavior, such as braking zones, acceleration points, and gear usage patterns along the circuit."
    )

    if not telemetry_available:
        st.warning("⚠️ Telemetry data is not available for this session.")
    else:
        if compare_mode:
            col1, col2 = st.columns(2)
            for i, (driver, laps) in enumerate(zip([driver1, driver2], [laps1, laps2])):
                with (col1 if i == 0 else col2):
                    st.markdown(f"### {driver}")
                    try:
                        lap = laps.pick_fastest()
                        tel = lap.get_telemetry().add_distance()
                        fig, ax = plt.subplots(figsize=(7, 4))
                        ax.plot(tel['Distance'], tel['Throttle'], color='green', label='Throttle')
                        ax.set_ylabel("Throttle (%)")
                        ax2 = ax.twinx()
                        ax2.plot(tel['Distance'], tel['nGear'], linestyle='--', color='orange' if i == 0 else 'red', label='Gear')
                        ax2.set_ylabel("Gear")
                        ax.set_xlabel("Distance (m)")
                        ax.grid(True)
                        fig.suptitle(f"{driver} - Throttle & Gear Profile")
                        st.pyplot(fig)
                    except Exception as e:
                        st.warning(f"Could not load telemetry for {driver}: {e}")
        else:
            try:
                lap = laps1.pick_fastest()
                tel = lap.get_telemetry().add_distance()
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(tel['Distance'], tel['Throttle'], color='green', label='Throttle')
                ax.set_ylabel("Throttle (%)")
                ax2 = ax.twinx()
                ax2.plot(tel['Distance'], tel['nGear'], linestyle='--', color='orange', label='Gear')
                ax2.set_ylabel("Gear")
                ax.set_xlabel("Distance (m)")
                ax.grid(True)
                fig.suptitle(f"{driver1} - Throttle & Gear Profile")
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not load telemetry for {driver1}: {e}")

                
# Tab 5: Lap Time Heatmap
with tab5:
    st.subheader("🗺️ Lap Time Heatmap")
    st.markdown(
        "This tab displays a color-coded heatmap of lap times for all drivers across the session. "
        "Cooler colors (blue) represent faster lap times, while warmer colors (red) indicate slower laps. "
        "It helps spot consistency, tire degradation, or performance drops visually over the race."
    )
    try:
        pivot = session.laps.pivot_table(index="LapNumber", columns="Driver", values="LapTimeSeconds")
        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(pivot.T, aspect="auto", cmap="coolwarm", interpolation="none")
        ax.set_xlabel("Lap Number")
        ax.set_ylabel("Driver")
        ax.set_title("Lap Time Heatmap (seconds)")
        ax.set_yticks(range(len(pivot.columns)))
        ax.set_yticklabels(pivot.columns)
        fig.colorbar(im, ax=ax).set_label("Lap Time (s)")
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not generate heatmap: {e}")

# Tab 6: Race Results - Clean Format
with tab6:
    st.subheader("🏁 Race Results")
    st.markdown(
        "This tab displays the final race results, showing finishing positions, driver names, team names, time gap to the winner, and highlights the driver with the fastest lap."
    )

    try:
        results = session.results.copy()
        laps = session.laps

        # Identify race winner
        winner_abbr = results.loc[results['Position'] == 1, 'Abbreviation'].values[0]
        winner_lap = laps.pick_driver(winner_abbr).pick_fastest()
        winner_time = winner_lap['LapTime']

        # Identify fastest lap driver
        fastest_lap = laps.pick_fastest()
        fastest_abbr = fastest_lap['Driver']

        # Replace unwanted DNF reasons
        dnf_terms = ['accident', 'collision', 'engine', 'gearbox', 'hydraulics', 'brakes', 'suspension',
                     'wheel', 'oil leak', 'overheating', 'retired', 'exhaust']
        results['Status'] = results['Status'].apply(
            lambda s: 'DNF' if any(term in s.lower() for term in dnf_terms) else s)

        def compute_gap(row):
            if row['Abbreviation'] == winner_abbr:
                return "Winner"
            if row['Status'] != 'Finished':
                return "DNF"
            try:
                driver_lap = laps.pick_driver(row['Abbreviation']).pick_fastest()
                gap = driver_lap['LapTime'] - winner_time
                return f"+{abs(gap.total_seconds()):.3f}s" # Removed abs() to avoid negative sign issues
            except:
                return "DNF"

        # Compute gap
        results['Gap'] = results.apply(compute_gap, axis=1)

        # Add "Fastest Lap" tag
        results['DriverStyled'] = results.apply(
            lambda row: f"{row['FullName']} (Fastest Lap)" if row['Abbreviation'] == fastest_abbr else row['FullName'],
            axis=1
        )

        # Prepare display table
        display_df = results[['Position', 'DriverStyled', 'TeamName', 'Gap']].sort_values(by='Position')
        display_df.columns = ['Position', 'Driver', 'Team', 'Gap']

        # --- Custom styling ---
        def style_row(row):
            if "(Fastest Lap)" in row['Driver']:
                return ['font-weight: bold; color: #00e6e6; font-size:18px; padding:10px;' for _ in row]
            elif row['Gap'] == "DNF":
                return ['color: gray; font-style: italic; font-size:17px; padding:10px;' for _ in row]
            else:
                return ['font-size:17px; padding:10px;' for _ in row]

        styled_df = (
            display_df.style
            .apply(style_row, axis=1)
            .set_table_styles([
                {"selector": "th", "props": [("color", "white"), ("background-color", "#1e1e1e"),
                                             ("text-align", "center"), ("font-size", "18px"), ("padding", "12px")]}
            ])
            .format({'Position': '{:.0f}'})
            .set_properties(**{
                'text-align': 'center',
                'font-size': '17px',
                'padding': '12px'
            })
        )

        # ✅ Display the styled table
        st.dataframe(styled_df, use_container_width=True, height=700)

    except Exception as e:
        st.warning(f"⚠️ Could not load race results: {e}")
