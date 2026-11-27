import pandas as pd

LEAKAGE_COLS = [
    "view_count",
    "like_count",
    "comment_count",
    "channel_median_views",
    "viral_ratio",
    "viral_score_log",
    "engagement_rate",
    "is_viral",
]
TARGET = "is_viral"

train = pd.read_csv("data/raw/youtube_train.csv")
test = pd.read_csv("data/raw/youtube_test.csv")

# Tách nhãn y trước, rồi mới bỏ 8 cột khỏi X
y_train = train[TARGET]
y_test = test[TARGET]
X_train = train.drop(columns=LEAKAGE_COLS)
X_test = test.drop(columns=LEAKAGE_COLS)

print("X_train:", X_train.shape, "| y_train:", y_train.shape)
print("X_test:", X_test.shape, "| y_test:", y_test.shape)

con_sot = [c for c in LEAKAGE_COLS if c in X_train.columns or c in X_test.columns]
print("Cột leakage còn sót:", con_sot)  # phải là []
print(X_train.columns.tolist())