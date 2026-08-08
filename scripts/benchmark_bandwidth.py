"""
Proof of Concept: Empirical Bandwidth & Data Volume Reduction Analysis
Compares raw 10Hz sensor streaming payload against 1Hz edge inference telemetry across 85 trucks.
"""

import json
import sys
import time
import numpy as np


def compute_bandwidth_metrics(fleet_size: int = 85, operating_hours: int = 10):
    # ---------------------------------------------------------
    # 1. Define Sample Payloads
    # ---------------------------------------------------------
    # Raw sensor frame (published at 10 Hz)
    raw_payload = {
        "truck_id": "TRUCK_085",
        "timestamp": time.time(),
        "temperature": 78.45,
        "vibration": 0.042,
        "voltage": 24.12,
        "current": 5.34,
        "pressure": 101.3,
        "rpm": 2100.0
    }

    # Edge inference result payload (published at 1 Hz on state change/summary)
    edge_payload = {
        "truck_id": "TRUCK_085",
        "timestamp": time.time(),
        "status_code": 0,          # 0: Normal, 1: Warning, 2: Critical
        "confidence": 0.985
    }

    # ---------------------------------------------------------
    # 2. Measure Payload Bytes (Serialized JSON)
    # ---------------------------------------------------------
    raw_bytes = len(json.dumps(raw_payload).encode('utf-8'))
    edge_bytes = len(json.dumps(edge_payload).encode('utf-8'))

    # Frequencies
    raw_freq_hz = 10    # 10 readings per second
    edge_freq_hz = 1    # 1 aggregated status per second

    # ---------------------------------------------------------
    # 3. Compute Rates per Vehicle & Fleet
    # ---------------------------------------------------------
    # Per truck bytes per second
    raw_rate_truck_bps = raw_bytes * raw_freq_hz
    edge_rate_truck_bps = edge_bytes * edge_freq_hz

    # Total fleet bytes per second (85 trucks)
    raw_rate_fleet_bps = raw_rate_truck_bps * fleet_size
    edge_rate_fleet_bps = edge_rate_truck_bps * fleet_size

    # Convert to Megabits per second (Mbps)
    raw_fleet_mbps = (raw_rate_fleet_bps * 8) / 1e6
    edge_fleet_mbps = (edge_rate_fleet_bps * 8) / 1e6

    # Daily data volume in Gigabytes (GB) per fleet
    seconds_per_day = operating_hours * 3600
    raw_daily_gb = (raw_rate_fleet_bps * seconds_per_day) / 1e9
    edge_daily_gb = (edge_rate_fleet_bps * seconds_per_day) / 1e9

    # Percentage Savings
    bandwidth_reduction_pct = ((raw_rate_fleet_bps - edge_rate_fleet_bps) / raw_rate_fleet_bps) * 100.0
    daily_data_saved_gb = raw_daily_gb - edge_daily_gb

    # ---------------------------------------------------------
    # 4. Display Results Table
    # ---------------------------------------------------------
    print("=" * 75)
    print("BANDWIDTH & DATA VOLUME SAVINGS PROOF REPORT")
    print("=" * 75)
    print(f"Fleet Size                    : {fleet_size} Trucks")
    print(f"Daily Operation Window        : {operating_hours} Hours/day\n")
    
    print(f"Raw Sensor Payload Size       : {raw_bytes} bytes @ {raw_freq_hz} Hz")
    print(f"Edge Telemetry Payload Size   : {edge_bytes} bytes @ {edge_freq_hz} Hz\n")
    
    print("-" * 75)
    print(f"{'Metric':<30} | {'Raw Streaming':<18} | {'Edge AI Telemetry':<18}")
    print("-" * 75)
    print(f"{'Per-Truck Bandwidth':<30} | {raw_rate_truck_bps/1024:.2f} KB/s           | {edge_rate_truck_bps/1024:.2f} KB/s")
    print(f"{'85-Truck Fleet Bandwidth':<30} | {raw_rate_fleet_bps/1024:.2f} KB/s ({raw_fleet_mbps:.3f} Mbps)| {edge_rate_fleet_bps/1024:.2f} KB/s ({edge_fleet_mbps:.3f} Mbps)")
    print(f"{'Daily Fleet Volume (' + str(operating_hours) + 'h)':<30} | {raw_daily_gb:.3f} GB            | {edge_daily_gb:.3f} GB")
    print(f"{'Monthly Volume (25 days)':<30} | {raw_daily_gb*25:.3f} GB           | {edge_daily_gb*25:.3f} GB")
    print("-" * 75)
    print(f"BANDWIDTH SAVINGS REDUCTION   : {bandwidth_reduction_pct:.2f}%")
    print(f"DAILY DATA SAVED PER FLEET    : {daily_data_saved_gb:.3f} GB/day")
    print(f"MONTHLY DATA SAVED (25 DAYS)  : {daily_data_saved_gb*25:.3f} GB/month")
    print("=" * 75)


if __name__ == "__main__":
    compute_bandwidth_metrics(fleet_size=85, operating_hours=10)