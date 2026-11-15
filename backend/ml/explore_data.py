from pathlib import Path
import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data" / "creditcard.csv"
df = pd.read_csv(DATA)

print("Shape:", df.shape)
print("Missing values:", int(df.isnull().sum().sum()))

counts = df["Class"].value_counts()
print("\nClass counts:\n", counts)
print(f"\nFraud percentage: {counts[1] / len(df) * 100:.3f}%")

print("\nAmount statistics by class (0 = normal, 1 = fraud):")
print(df.groupby("Class")["Amount"].describe())
