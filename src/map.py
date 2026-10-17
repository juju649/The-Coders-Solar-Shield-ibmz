
import folium
import pandas as pd

# Load Texas hospital data
hospitals = pd.read_csv("hospitals_tx.csv")

# Create map centered on West Texas
m = folium.Map(
    location=[31.0, -102.5],
    zoom_start=7,
    tiles=None
)

# Add Esri background map
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
    attr="Tiles © Esri",
    name="Esri World Street Map"
).add_to(m)

# Create separate layers for each type of data
hospital_layer = folium.FeatureGroup(name="Hospitals (Red)")
battery_layer = folium.FeatureGroup(name="Battery Storage (Green)")
risk_layer = folium.FeatureGroup(name="Medical Risk ZIP Codes (Orange)")
# Add hospitals as red markers
count = 0

for _, hospital in hospitals.iterrows():

    lat = hospital["latitude"]
    lon = hospital["longitude"]

    if pd.notna(lat) and pd.notna(lon):
        folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            color="red",
            fill=True,
            fill_color="red",
            fill_opacity=0.8,
            popup=str(hospital["name"])
        ).add_to(hospital_layer)

        count += 1


# Load Texas energy storage data
batteries = pd.read_csv("batteries_tx.csv")

# Keep battery technology records
batteries = batteries[
    batteries["technology"].astype(str).str.strip().eq("Batteries")
]

# Add batteries as green markers
battery_count = 0

for _, battery in batteries.iterrows():
    lat = battery["latitude"]
    lon = battery["longitude"]

    if pd.notna(lat) and pd.notna(lon):
        folium.CircleMarker(
            location=[lat, lon],
            radius=6,
            color="green",
            fill=True,
            fill_color="green",
            fill_opacity=0.9,
            popup=(
                f"{battery['name']}<br>"
                f"Capacity: {battery['capacity_mw']} MW<br>"
                f"Energy: {battery['energy_mwh']} MWh"
            )
        ).add_to(battery_layer)

        battery_count += 1

print("Battery markers added:", battery_count)


# Add Texas ZIP code medical-risk circles
zip_risk = pd.read_csv("zip_risk_coordinates_tx.csv")

risk_count = 0

for _, area in zip_risk.iterrows():
    if pd.isna(area["latitude"]) or pd.isna(area["longitude"]):
        continue

    patients = int(area["at_risk_count"])

    folium.CircleMarker(
        location=[area["latitude"], area["longitude"]],
        radius=max(3, min(15, patients / 30)),
        color="orange",
        fill=True,
        fill_color="orange",
        fill_opacity=0.35,
        popup=f"ZIP: {area['zip_code']}<br>Electricity-dependent Medicare beneficiaries: {patients}"
    ).add_to(risk_layer)

    risk_count += 1

print("Medical-risk ZIP circles added:", risk_count)

# Add all three layers to the map
hospital_layer.add_to(m)
battery_layer.add_to(m)
risk_layer.add_to(m)



# Load an actual AI-flagged solar-drop prediction
predictions = pd.read_csv("predictions.csv")

flagged = predictions[predictions["predicted_drop"] == 1]

event = flagged.nsmallest(
    1, "predicted_change_mw"
).iloc[0]

print("AI solar-drop alert loaded!")
print("Time:", event["time_utc"])
print("Predicted change:", event["predicted_change_mw"], "MW")

# Dallas is an illustrative impact location,
# not a location predicted by the statewide solar model
alert_lat = 32.78
alert_lon = -96.80


# Find the closest battery to the alert
candidate_batteries = batteries.copy()

from recommend import distance_km

candidate_batteries["alert_distance_km"] = candidate_batteries.apply(
    lambda row: distance_km(
        alert_lat, alert_lon,
        row["latitude"], row["longitude"]
    ),
    axis=1
)

candidate = candidate_batteries.nsmallest(
    1, "alert_distance_km"
).iloc[0]

# Create a separate layer for the demonstration
alert_layer = folium.FeatureGroup(
    name="Solar Drop Alert + Candidate Battery"
)

# Highlight the solar-drop alert location
folium.Marker(
    location=[alert_lat, alert_lon],
    
popup=folium.Popup(
    f"<b>AI SOLAR-DROP ALERT</b><br>"
    f"Prediction time: {event['time_utc']}<br>"
    f"Current Texas solar: {event['solar_mw']:,.1f} MW<br>"
    f"Predicted solar in 15 min: {event['predicted_solar_15']:,.1f} MW<br>"
    f"Predicted change: {event['predicted_change_mw']:,.1f} MW<br>"
    f"<b>AI alert triggered: YES</b><br><br>"
    f"<i>Texas-wide forecast. Dallas is an illustrative "
    f"impact location, not a predicted outage.</i>",
    max_width=350
),

    tooltip="Solar Drop Alert",
    icon=folium.Icon(color="purple", icon="flash")
).add_to(alert_layer)

# Highlight the candidate battery
folium.Marker(
    location=[candidate["latitude"], candidate["longitude"]],
    popup=(
        f"Candidate battery: {candidate['name']}<br>"
        f"Capacity: {candidate['capacity_mw']} MW<br>"
        f"Distance: {candidate['alert_distance_km']:.1f} km<br>"
        "Geographic recommendation only"
    ),
    tooltip="Candidate Battery",
    icon=folium.Icon(color="blue", icon="flash")
).add_to(alert_layer)

# Draw a line connecting the alert and battery
folium.PolyLine(
    locations=[
        [alert_lat, alert_lon],
        [candidate["latitude"], candidate["longitude"]]
    ],
    color="purple",
    weight=4,
    dash_array="8, 8",
    tooltip="Geographic distance only; not a power connection"
).add_to(alert_layer)

alert_layer.add_to(m)

# Add checkboxes to show or hide each layer
folium.LayerControl(collapsed=False).add_to(m)
# Save updated map
m.save("map.html")

print("Hospital map created successfully!")
print("Hospitals added:", count)
