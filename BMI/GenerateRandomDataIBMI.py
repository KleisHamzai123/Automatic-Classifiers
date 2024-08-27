import csv
import numpy as np

# Function to calculate BMI class
def calculate_bmi_class(bmi):
    if bmi < 17.5:
        return 0  # Underweight
    elif bmi < 25:
        return 1  # Normal weight
    elif bmi < 32:
        return 2  # Overweight
    else:
        return 3  # Obesity

# Define ranges for height (in centimeters) and weight (in kilograms)
height_range = (145, 210)
weight_range = (50, 130)

# Generate random heights and weights for each row, in total 400
heights = np.random.randint(height_range[0], height_range[1] + 1, size=400)
weights = np.random.randint(weight_range[0], weight_range[1] + 1, size=400)

# Calculate BMI for each row
bmis = weights / ((heights / 100) ** 2)

# Round BMI values to integers
rounded_bmis = np.round(bmis, 2)

# Determine BMI class for each row
bmi_classes = [calculate_bmi_class(bmi) for bmi in rounded_bmis]

# Write data to CSV file with 120 examples from 400 in total, taken from each class
csv_file = 'height_weight_bmi_samples3.csv'
class_limits = [10, 50, 40, 20]  # Number of examples to take from each class
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Height (cm)', 'Weight (kg)', 'BMI Class'])  # header
    class_count = [0, 0, 0, 0]  # Count of elements in each class
    for height, weight, bmi_class in zip(heights, weights, bmi_classes):
        if class_count[bmi_class] < class_limits[bmi_class]:  # Extract specified number of elements from each class
            writer.writerow([height, weight, bmi_class])  # Write row to CSV
            class_count[bmi_class] += 1  # Increment count for the current class

print(f"CSV file '{csv_file}' created!")