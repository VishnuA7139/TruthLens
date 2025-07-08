import pandas as pd

# Define file paths
train_file = "train.tsv"
valid_file = "valid.tsv"
test_file = "test.tsv"
recent_file = "recent_news_dataset.csv"

def load_liar_data(file_path):
    columns = [
        "id", "label", "statement", "subject", "speaker", "speaker_job",
        "state_info", "party_affiliation", "barely_true_counts", "false_counts",
        "half_true_counts", "mostly_true_counts", "pants_on_fire_counts", "context"
    ]
    return pd.read_csv(file_path, sep="\t", names=columns)

print("Loading LIAR dataset...")
train_df = load_liar_data(train_file)
valid_df = load_liar_data(valid_file)
test_df = load_liar_data(test_file)

liar_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
print(f"LIAR dataset loaded with {len(liar_df)} rows.")

# Keep only relevant columns
liar_df = liar_df[["statement", "label"]]

# Normalize labels
def relabel(label_text):
    label_text = label_text.strip().lower()
    if label_text in ["true", "mostly-true", "half-true"]:
        return "real"
    else:
        return "fake"

liar_df["label"] = liar_df["label"].apply(relabel)
liar_df = liar_df.dropna(subset=["statement"])

print(f"After filtering: {len(liar_df)} rows.")

print("Loading recent news dataset...")
recent_df = pd.read_csv(recent_file)

if "statement" not in recent_df.columns or "label" not in recent_df.columns:
    raise ValueError("recent_news_dataset.csv must contain 'statement' and 'label' columns.")

recent_df["label"] = recent_df["label"].astype(str).str.strip().str.lower()
recent_df["label"] = recent_df["label"].replace({"1": "fake", "0": "real"})
recent_df = recent_df.dropna(subset=["statement"])

print(f"Recent dataset loaded with {len(recent_df)} rows.")

# Combine and shuffle
merged_df = pd.concat([liar_df, recent_df], ignore_index=True)
merged_df = merged_df.drop_duplicates(subset="statement")
merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Combined dataset size: {len(merged_df)} rows.")

merged_df.to_csv("cleaned_dataset.csv", index=False)
print("Dataset saved to cleaned_dataset.csv")

