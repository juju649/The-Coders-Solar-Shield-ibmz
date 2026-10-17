
import pandas as pd
from math import radians, sin, cos, sqrt, atan2

# Calculate distance between two locations in kilometers
def distance_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    a = (sin(dlat / 2) ** 2 +
         cos(radians(lat1)) * cos(radians(lat2)) *
         sin(dlon / 2) ** 2)

    return 2 * R * atan2(sqrt(a), sqrt(1 - a))


def recommend(lat, lon):
    # Load our cleaned Texas datasets
    hospitals = pd.read_csv("hospitals_tx.csv")
    batteries = pd.read_csv("batteries_tx.csv")
    risks = pd.read_csv(
        "zip_risk_coordinates_tx.csv",
        dtype={"zip_code": str}
    )

    # Calculate distance to hospitals
    hospitals["distance_km"] = hospitals.apply(
        lambda row: distance_km(
            lat, lon, row["latitude"], row["longitude"]
        ), axis=1
    )

    # Calculate distance to vulnerable ZIP codes
    risks["distance_km"] = risks.apply(
        lambda row: distance_km(
            lat, lon, row["latitude"], row["longitude"]
        ), axis=1
    )

    # Calculate distance to battery facilities
    batteries = batteries[
        batteries["technology"].astype(str).str.strip() == "Batteries"
    ].copy()

    batteries["distance_km"] = batteries.apply(
        lambda row: distance_km(
            lat, lon, row["latitude"], row["longitude"]
        ), axis=1
    )

    # Find nearby hospitals and vulnerable ZIP codes
    nearby_hospitals = hospitals.nsmallest(3, "distance_km")
    nearby_risks = risks[risks["distance_km"] <= 50].sort_values(
        "at_risk_count", ascending=False
    ).head(5)

    # Find the closest candidate battery
    closest_battery = batteries.nsmallest(1, "distance_km")

    print("\n=== GRID RESILIENCE RECOMMENDATION ===")
    print(f"Alert location: {lat}, {lon}")

    print("\nNearest hospitals:")
    print(nearby_hospitals[["name", "distance_km"]].to_string(index=False))

    print("\nHigh-risk ZIP codes within 50 km:")
    print(
        nearby_risks[
            ["zip_code", "at_risk_count", "distance_km"]
        ].to_string(index=False)
    )

    print("\nCandidate battery facility:")
    print(
        closest_battery[
            ["name", "capacity_mw", "energy_mwh", "distance_km"]
        ].to_string(index=False)
    )

    print("\nNOTE: This is a geographic planning recommendation.")
    print("It does not confirm grid connectivity or battery availability.")


# Example solar-drop alert near Dallas
if __name__ == "__main__":
    recommend(32.78, -96.80)
