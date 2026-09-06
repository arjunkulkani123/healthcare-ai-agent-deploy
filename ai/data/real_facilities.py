"""
Real Facility Data (Google Places API integration)
=======================================================

Replaces the synthetic Govt_Hospital_A/B/Private_Clinic_C dataset with
REAL hospitals and clinics near an actual location.

Honest scope boundary: Google Places gives us real facility names,
addresses, and coordinates -- genuinely real. It does NOT give us real
doctor rosters, working hours, or appointment availability, because no
public API for that exists (hospital scheduling systems are private,
proprietary, and vary by facility). So this module:

    REAL:      facility name, address, coordinates, rating
    SIMULATED: doctors, their working hours, service offerings,
               daily capacity (generated deterministically per
               facility so results are reproducible, and clearly
               documented as synthetic every time they're used)

This is the same honest split a production version of this system
would need until/unless it integrated directly with each hospital's
own (typically private, contractual) scheduling API.

Requires a Google Cloud API key with the "Places API" and "Geocoding
API" enabled. Set it as:
    - Local: environment variable GOOGLE_PLACES_API_KEY
    - Streamlit Cloud: add GOOGLE_PLACES_API_KEY in the app's Secrets
"""

import os
import hashlib
import random

import requests

GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"
PLACES_NEARBY_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

SERVICE_POOL = ["OPD", "Emergency", "Vaccination", "Diagnostic"]
FIRST_NAMES = ["Rao", "Iyer", "Sharma", "Mehta", "Khan", "Reddy", "Nair", "Gupta", "Verma", "Joshi"]


def get_api_key() -> str:
    key = os.environ.get("GOOGLE_PLACES_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets.get("GOOGLE_PLACES_API_KEY")
    except Exception:
        return None


def geocode_location(address: str, api_key: str) -> dict:
    """Converts a place name/address into {lat, lng}, or None on failure."""
    try:
        resp = requests.get(GEOCODE_URL, params={"address": address, "key": api_key}, timeout=10)
        data = resp.json()
        if data.get("status") != "OK" or not data.get("results"):
            return None
        loc = data["results"][0]["geometry"]["location"]
        return {"lat": loc["lat"], "lng": loc["lng"]}
    except Exception:
        return None


def search_nearby_hospitals(lat: float, lng: float, api_key: str, radius_m: int = 8000, max_results: int = 5) -> list:
    """
    Returns real hospitals/clinics near (lat, lng):
        [{"name": str, "address": str, "lat": float, "lng": float, "rating": float or None}, ...]
    Empty list on failure (caller should fall back to synthetic data).
    """
    try:
        resp = requests.get(
            PLACES_NEARBY_URL,
            params={
                "location": f"{lat},{lng}",
                "radius": radius_m,
                "type": "hospital",
                "key": api_key,
            },
            timeout=10,
        )
        data = resp.json()
        if data.get("status") not in ("OK", "ZERO_RESULTS"):
            return []

        results = []
        for place in data.get("results", [])[:max_results]:
            results.append({
                "name": place.get("name"),
                "address": place.get("vicinity"),
                "lat": place["geometry"]["location"]["lat"],
                "lng": place["geometry"]["location"]["lng"],
                "rating": place.get("rating"),
            })
        return results
    except Exception:
        return []


def _deterministic_rng(seed_text: str) -> random.Random:
    """A Random instance seeded from a hash of the facility name, so the
    same real facility always gets the same simulated roster instead of
    a different one every run (reproducibility matters for a system
    that explains its own reasoning)."""
    seed = int(hashlib.sha256(seed_text.encode()).hexdigest(), 16) % (2**31)
    return random.Random(seed)


def generate_synthetic_roster(facility_name: str) -> dict:
    """
    Generates a plausible-but-simulated doctor + operating profile for a
    REAL facility name. Returned shape matches ai/csp/domain.py's
    HOSPITALS/DOCTORS structure so it can be dropped in directly.
    """
    rng = _deterministic_rng(facility_name)

    is_public = "govt" in facility_name.lower() or "government" in facility_name.lower() or "district" in facility_name.lower()
    open_days = {"Mon", "Tue", "Wed", "Thu", "Fri"} | ({"Sat"} if rng.random() > 0.4 else set())

    n_doctors = rng.randint(1, 3)
    doctors = {}
    for i in range(n_doctors):
        doc_id = f"Dr_{rng.choice(FIRST_NAMES)}_{facility_name[:4]}{i}"
        n_services = rng.randint(1, 2)
        services = set(rng.sample(SERVICE_POOL, n_services))
        start_hour = rng.choice([8, 9, 10])
        end_hour = start_hour + rng.choice([4, 6, 8])
        work_days = rng.sample(sorted(open_days), k=max(1, len(open_days) - rng.randint(0, 1)))
        doctors[doc_id] = {
            "hospital": facility_name,
            "services": services,
            "work_days": set(work_days),
            "work_hours": (f"{start_hour:02d}:00", f"{min(end_hour, 21):02d}:00"),
        }

    hospital_entry = {
        "distance_km": None,  # filled in by the caller once real coordinates are known
        "type": "public" if is_public else "private",
        "open_days": open_days,
        "daily_capacity_per_doctor": rng.randint(6, 12),
    }

    return {"hospital": hospital_entry, "doctors": doctors}


def build_real_dataset(location_text: str, api_key: str = None) -> dict:
    """
    Main entry point: given a location (e.g. "Indore, Madhya Pradesh"),
    returns a dict compatible with ai/csp/domain.py's HOSPITALS/DOCTORS
    format, built from REAL nearby facilities with SIMULATED rosters.

    Returns None if geocoding/search fails (e.g. no API key, quota
    exceeded, invalid location) -- caller should fall back to the
    synthetic 3-hospital dataset in that case.
    """
    api_key = api_key or get_api_key()
    if not api_key:
        return None

    origin = geocode_location(location_text, api_key)
    if origin is None:
        return None

    places = search_nearby_hospitals(origin["lat"], origin["lng"], api_key)
    if not places:
        return None

    hospitals = {}
    doctors = {}
    for place in places:
        roster = generate_synthetic_roster(place["name"])
        distance_km = _haversine_km(origin["lat"], origin["lng"], place["lat"], place["lng"])
        roster["hospital"]["distance_km"] = round(distance_km, 1)
        roster["hospital"]["real_address"] = place["address"]
        roster["hospital"]["real_coordinates"] = (place["lat"], place["lng"])
        roster["hospital"]["rating"] = place["rating"]

        hospitals[place["name"]] = roster["hospital"]
        doctors.update(roster["doctors"])

    return {
        "origin": origin,
        "hospitals": hospitals,
        "doctors": doctors,
        "source": "Google Places API (real facilities) + simulated scheduling data",
    }


def _haversine_km(lat1, lng1, lat2, lng2) -> float:
    import math
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


if __name__ == "__main__":
    result = build_real_dataset("Indore, Madhya Pradesh")
    if result is None:
        print("No API key configured (or lookup failed) -- would fall back to synthetic data.")
    else:
        print(f"Origin: {result['origin']}")
        print(f"\nFound {len(result['hospitals'])} real facilities:\n")
        for name, info in result["hospitals"].items():
            print(f"  {name} -- {info['distance_km']} km, {info['type']}, rating={info.get('rating')}")
            print(f"    Address: {info['real_address']}")
        print(f"\nSimulated {len(result['doctors'])} doctors across these facilities.")
