import numpy as np
import pandas as pd
import random

np.random.seed(42)
random.seed(42)

NUM_BASE_SAMPLES = 500
NUM_EXTENDED_SAMPLES = 5000

OUTPUT_BASE = "iith_mess_dataset_500.csv"
OUTPUT_EXTENDED = "iith_mess_dataset_5000.csv"

def simulate_weather():
    temperature = np.random.normal(30, 3)
    humidity = np.random.normal(60, 10)
    wind_speed = np.random.normal(8, 2)
    rain = np.random.binomial(1, 0.2)
    air_quality = np.random.normal(80, 15)
    return temperature, humidity, wind_speed, rain, air_quality

def simulate_sleep_cycle():
    rising_time = int(np.random.normal(7*60, 40))
    sleeping_time = int(np.random.normal(23*60, 50))
    return rising_time, sleeping_time

def mess_duration_formula(meal_time, weekend, deadline, weather_temp):
    base = 20

    if meal_time == 0:
        base += 5
    elif meal_time == 1:
        base += 15
    else:
        base += 20

    if weekend == 1:
        base += 10

    if deadline == 0:
        base -= 5

    if weather_temp > 32:
        base -= 2

    noise = np.random.normal(0, 3)
    return max(5, base + noise)

data = []

for i in range(NUM_BASE_SAMPLES):

    student_id = random.randint(1, 200)
    day_of_week = random.randint(0, 6)
    meal_time = random.randint(0, 2)
    meal_type = random.randint(0, 1)
    is_weekend = 1 if day_of_week in [5,6] else 0
    class_day = random.randint(0,1)
    project_deadline = random.randint(0,1)

    temperature, humidity, wind_speed, rain, air_quality = simulate_weather()
    rising_time, sleeping_time = simulate_sleep_cycle()

    mess_duration = mess_duration_formula(meal_time, is_weekend, project_deadline, temperature)

    row = [
        student_id, day_of_week, meal_time, meal_type, is_weekend,
        class_day, project_deadline,
        temperature, humidity, wind_speed, rain, air_quality,
        rising_time, sleeping_time, mess_duration
    ]

    data.append(row)

columns = [
    "student_id","day_of_week","meal_time","meal_type","is_weekend",
    "class_day","project_deadline",
    "temperature","humidity","wind_speed","rain","air_quality",
    "rising_time","sleeping_time","mess_duration"
]

df = pd.DataFrame(data, columns=columns)
df.to_csv(OUTPUT_BASE, index=False)
print("Base dataset saved:", OUTPUT_BASE)

rows = []

for i in range(NUM_EXTENDED_SAMPLES):
    base_row = df.sample(1).iloc[0].copy()

    base_row["temperature"] += np.random.normal(0, 1.5)
    base_row["humidity"] += np.random.normal(0, 3)
    base_row["wind_speed"] += np.random.normal(0, 0.8)
    base_row["rising_time"] += int(np.random.normal(0, 10))
    base_row["sleeping_time"] += int(np.random.normal(0, 10))

    rows.append(base_row)

df_extended = pd.DataFrame(rows)
df_extended.to_csv(OUTPUT_EXTENDED, index=False)

print("Extended dataset saved:", OUTPUT_EXTENDED)
