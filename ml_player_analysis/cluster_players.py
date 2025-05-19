print("🎉 Script is running!")  # Debug line to confirm script is executing

import os
import json
import pandas as pd

data_folder = 'player data'

# Confirm absolute path to folder
abs_path = os.path.abspath(data_folder)
print(f"📁 Looking for data in: {abs_path}")

# Check if folder exists
if not os.path.exists(data_folder):
    print("❌ Folder does NOT exist!")
    exit()

# Read and combine JSON files
all_data = []

files = os.listdir(data_folder)
print(f"📄 Found files: {files}")

for filename in files:
    if filename.endswith('.json'):
        full_path = os.path.join(data_folder, filename)
        print(f"➡️ Reading file: {full_path}")
        try:
            with open(full_path) as f:
                data = json.load(f)
                all_data.append(data)
        except Exception as e:
            print(f"⚠️ Error reading {filename}: {e}")

# Check and convert to DataFrame
if not all_data:
    print("❌ No data loaded. Check your .json files or folder structure.")
else:
    df = pd.DataFrame(all_data)
    df.to_csv('player_dataset.csv', index=False)
    print("✅ Data loaded into DataFrame:")
    print(df.head())
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Use only numerical columns for clustering
features = df[['flips', 'mismatches', 'durationInSeconds']]

# Apply K-Means with 2 or 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(features)

# Print cluster labels
print("\n🔢 Cluster assignments:")
print(df[['flips', 'mismatches', 'durationInSeconds', 'cluster']])

# Visualize clusters (2D using flips and mismatches)
plt.figure(figsize=(8, 5))
plt.scatter(df['flips'], df['mismatches'], c=df['cluster'], cmap='viridis', s=100)
plt.xlabel('Number of Flips')
plt.ylabel('Mismatches')
plt.title('🎯 Player Clusters Based on Behavior')
plt.grid(True)
plt.show(block=False)
# Create a simple label: did the player struggle? (1 = yes, 0 = no)
# Criteria: too many flips OR mismatches OR long time
df['struggled'] = df.apply(lambda row: 1 if row['flips'] > 42 or row['mismatches'] > 13 or row['durationInSeconds'] > 35 else 0, axis=1)
print("\n📌 Struggle labels added:")
print(df[['flips', 'mismatches', 'durationInSeconds', 'struggled']])
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns

# Define features and target
X = df[['flips', 'mismatches', 'durationInSeconds']]
y = df['struggled']

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train decision tree
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("\n📊 Decision Tree Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Plot predictions
plt.figure(figsize=(6, 4))
sns.scatterplot(x=df['flips'], y=df['mismatches'], hue=df['struggled'], palette='coolwarm', s=100)
plt.title("🧠 Difficulty Prediction (Red = Struggled)")
plt.xlabel("Flips")
plt.ylabel("Mismatches")
plt.grid(True)
plt.show()
print("\n🧠 Automated Design Insights:")

# Insight 1: Flips threshold
avg_flips = df['flips'].mean()
struggling_flips = df[df['struggled'] == 1]['flips'].mean()
if struggling_flips > avg_flips:
    print(f"🔍 Players who struggled flipped an average of {struggling_flips:.1f} cards vs {avg_flips:.1f} overall.")
    print("💡 Consider adding a hint system or limiting unnecessary flips.")

# Insight 2: Mismatches by cluster
avg_mismatches = df.groupby('cluster')['mismatches'].mean()
highest_cluster = avg_mismatches.idxmax()
print(f"\n🔍 Cluster {highest_cluster} has the highest average mismatches: {avg_mismatches[highest_cluster]:.1f}")
print("💡 Suggest reducing visual similarity or adding a match preview animation.")

# Insight 3: Duration warning
long_sessions = df[df['durationInSeconds'] > 35]
if not long_sessions.empty:
    print(f"\n🔍 {len(long_sessions)} players had sessions over 35 seconds.")
    print("💡 You could add a progress bar or feedback to maintain pacing.")

