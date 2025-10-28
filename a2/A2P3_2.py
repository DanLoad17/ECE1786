import pandas as pd
from sklearn.model_selection import train_test_split

# -----------------------
# Load the full dataset
# -----------------------
df = pd.read_csv("data\data.tsv", sep="\t")

# 1️⃣ First split: 80% temp, 20% test
temp_df, test_df = train_test_split(
    df,
    test_size=0.20,
    stratify=df['label'],
    random_state=42
)

# 2️⃣ Second split: from temp into 64% train, 16% validation
train_df, val_df = train_test_split(
    temp_df,
    test_size=0.20,         # 0.20 of 80% = 16% total
    stratify=temp_df['label'],
    random_state=42
)

# 3️⃣ Overfit set: 25 examples per class
overfit_df = (
    df.groupby('label', group_keys=False)
      .apply(lambda x: x.sample(n=25, random_state=42))
      .sample(frac=1, random_state=42)  # shuffle
      .reset_index(drop=True)
)

# -----------------------
# Save to TSV
# -----------------------
train_df.to_csv("train.tsv", sep="\t", index=False)
val_df.to_csv("validation.tsv", sep="\t", index=False)
test_df.to_csv("test.tsv", sep="\t", index=False)
overfit_df.to_csv("overfit.tsv", sep="\t", index=False)

# -----------------------
# Verification checks
# -----------------------
def class_balance(name, subset):
    counts = subset['label'].value_counts()
    print(f"{name}: {dict(counts)} (total {len(subset)})")

def check_overlap(a, b):
    overlap = len(pd.merge(a, b, on=['text', 'label']))
    return overlap

print("\nClass Balance:")
class_balance("Train", train_df)
class_balance("Validation", val_df)
class_balance("Test", test_df)
class_balance("Overfit", overfit_df)

print("\nOverlap Check (0 if correct):")
print("Train x Validation:", check_overlap(train_df, val_df))
print("Train x Test:", check_overlap(train_df, test_df))
print("Validation x Test:", check_overlap(val_df, test_df))
