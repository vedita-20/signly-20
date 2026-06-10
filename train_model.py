import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

DATA_FILE = "data.csv"
MODEL_FILE = "sign_model.pkl"

# 1. Verification Check
if not os.path.exists(DATA_FILE):
    print(f"❌ Error: Could not find '{DATA_FILE}' in this directory! Did it name something else?")
    exit()

print("📖 Loading your custom MediaPipe coordinate dataset...")
df = pd.read_csv(DATA_FILE)

# 2. Extract Data Split
# 'X' represents the 63 float coordinates, 'y' represents the letter labels (A, B, C...)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

print(f"📊 Dataset loaded successfully with {df.shape[0]} total samples across {len(y.unique())} classes.")

# 3. Split into Train (80%) and Test (20%) Sets
# 'stratify=y' ensures every single letter gets an equal representation in the testing pool
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Train the AI Engine (Random Forest)
print("\n🧠 Training the AI brain pattern recognizer... (This should take under a minute)")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 5. Evaluate Performance
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n🎯 --- AI FINAL EXAM REPORT ---")
print(f"Overall Accuracy: {accuracy * 100:.2f}%")
print("\n📝 Per-Letter Precision Breakdown:")
print(classification_report(y_test, y_pred))

# 6. Export the Brain File
with open(MODEL_FILE, "wb") as f:
    pickle.dump(model, f)

print(f"💾 Success! True-image model compiled and saved cleanly as '{MODEL_FILE}'")