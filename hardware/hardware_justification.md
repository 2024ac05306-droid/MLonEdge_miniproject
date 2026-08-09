

## HW Selection Justification & Trade-Off Arguments

# Recommendation: Option 1 — Raspberry Pi 5 (8 GB) + AI HAT+ (13 TOPS Hailo-8L)
Option 1 provides the optimal balance across all three vertices of the Constraint Triangle:
Power Compliance (<10W): At 7.5W TDP, the Pi 5 + Hailo-8L runs comfortably within the 10W DC-DC power budget, avoiding thermal saturation in sealed truck cabins and preserving power during engine-off stops.
Latency Guarantee (<90s): The Hailo-8L NPU executes quantized INT8 model inference (model_ptq.tflite) in under 10 ms. Coupled with data buffering, total end-to-end processing is ~2.02 seconds—consuming just 2.24% of the 90-second latency budget.
Fleet Scalability: At ~₹15,000/truck, deployment across the 85-truck pilot costs ₹12.75 Lakhs, scaling to ₹39.75 Lakhs for the full 265-truck fleet. This provides Linux OS flexibility (native MQTT, Python, MLflow, edge storage) at a fraction of high-end AI accelerator costs.

# Argument against Option 2: Jetson Orin Nano Super Developer Kit
Power Budget Violation: At 15W under moderate load, Option 2 exceeds the 10W AI power constraint by 50%. In sealed automotive enclosures, this causes thermal throttling, reduced component lifespan, and power starvation over 12V DC-DC lines.
Severe Cost Inefficiency: At ~₹45,000/truck, the 85-truck pilot requires ₹38.25 Lakhs, ballooning to ₹1.19 Crores for 265 trucks.
Excessive Compute Overkill: Delivering 67 TOPS for low-frequency vibration and temperature telemetry creates an expensive compute surplus that provides no operational advantage for meeting the 90-second alert threshold.
# Argument against Option 3: STM32H7 Custom MCU Board

Memory & Processing Bottlenecks: While boasting an attractive 0.4W TDP and ~₹3,500 cost, the Cortex-M7 core lacks sufficient SRAM and vector compute to run 500 Hz 3-axis vibration FFT processing alongside continuous TFLite Micro inference smoothly.
Engineering Risk & Development Overhead: Custom board manufacturing, firmware development, lack of a full OS for local MQTT queueing during 90-minute connectivity dead zones, and OTA update management significantly increase hidden NRE (Non-Recurring Engineering) costs.

# Final recommendation: Option 1 — Raspberry Pi 5 + AI HAT+
It is the sweet spot:
- 7.5 W < 10 W 
- 13 TOPS provides adequate AI capability 
- Supports the <90-second alert target 
- Much cheaper than Jetson at fleet scale 
- More flexible than a custom STM32 solution 
- Better suited to your MLflow → pruning → quantization → OTA model deployment architecture
