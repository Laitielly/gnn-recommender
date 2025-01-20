import os
import polars as pl
from tqdm import tqdm

def write_info(train_data, test_data, savepath):
    """Saves dataset statistics to a text file."""
    print("Start write statistics to info.txt")
    stats = {
        "users": len(train_data["user_id"].unique()),
        "items": len(train_data["track_id"].unique()),
        "train_size": train_data.shape[0],
        "train_label_0": train_data.filter(pl.col("play_fraction") < 0.5).shape[0],
        "train_label_1": train_data.filter(pl.col("play_fraction") >= 0.5).shape[0],
        "test_size": test_data.shape[0],
        "test_label_0": test_data.filter(pl.col("play_fraction") < 0.5).shape[0],
        "test_label_1": test_data.filter(pl.col("play_fraction") >= 0.5).shape[0],
    }

    info = (f"users: {stats['users']}\n"
            f"items: {stats['items']}\n"
            f"train: {stats['train_size']}\n"
            f"  0.00: {stats['train_label_0']}\n"
            f"  1.00: {stats['train_label_1']}\n"
            f"test: {stats['test_size']}\n"
            f"  0.00: {stats['test_label_0']}\n"
            f"  1.00: {stats['test_label_1']}\n")

    os.makedirs(savepath, exist_ok=True)
    with open(os.path.join(savepath, "info.txt"), "w") as f:
        f.write(info.strip())

    print("Statistics written to info.txt")

def write_samples(data, savepath, dataset_type):
    """Saves data to a text file in the specified directory."""
    print(f"Start write statistics to {dataset_type}.txt")
    sorted_data = data.sort("datetime")
    lines = (sorted_data
             .select(["user_id", "track_id", "play_fraction"])
             .to_numpy().tolist())

    with open(os.path.join(savepath, f"{dataset_type}.txt"), "w") as f:
        for user_id, track_id, play_fraction in lines:
            f.write(f"{int(user_id)} {int(track_id)} {play_fraction:.2f}\n")

    print(f"File written: {dataset_type}.txt")

def split_data_by_user(interactions, split_ratio, desc):
    """Splits user data into two parts based on the specified ratio."""
    part1, part2 = [], []
    grouped_interactions = interactions.group_by("user_id")
    count_user = len(set(interactions['user_id'].unique()))
    for _, group in tqdm(grouped_interactions, desc=desc, total=count_user):
        n_split = int(group.shape[0] * split_ratio)
        part1.append(group[:n_split])
        part2.append(group[n_split:])
    return pl.concat(part1), pl.concat(part2)

def preprocess_data(interactions, savepath):
    """Processes and splits data into training, validation, and test sets."""
    print("Removing rows with missing values...")
    interactions = interactions.drop_nulls()
    print(f"Remaining records after removing missing values: {interactions.shape[0]}.")

    print("Filtering by playback duration...")
    interactions = interactions.filter(pl.col("play_duration") <= 600)

    print("Calculating track lengths and play_fraction...")
    track_stats = interactions.group_by("track_id").agg(
        pl.col("play_duration").max().alias("estimated_track_length")
    )
    interactions = interactions.join(track_stats, on="track_id")
    interactions = interactions.with_columns(
        (pl.col("play_duration") / pl.col("estimated_track_length")).alias("play_fraction")
    )

    print("Filtering users with fewer than 5 interactions...")
    user_interaction_counts = interactions.group_by("user_id").agg(
        pl.len().alias("interaction_count")
    )
    interactions = interactions.join(user_interaction_counts, on="user_id")
    interactions = interactions.filter(pl.col("interaction_count") >= 5).drop("interaction_count")

    print("Indexing users and tracks...")
    user_mapping = {uid: idx for idx, uid in enumerate(tqdm(interactions['user_id'].unique(), desc="Indexing users"))}
    item_mapping = {iid: idx for idx, iid in enumerate(tqdm(interactions['track_id'].unique(), desc="Indexing tracks"))}

    interactions = interactions.with_columns([
        pl.col("user_id").cast(str).replace(user_mapping).cast(pl.Int64).alias("user_id"),
        pl.col("track_id").cast(str).replace(item_mapping).cast(pl.Int64).alias("track_id")
    ])

    print("Splitting data into train and test...")
    interactions_sorted = interactions.sort(["user_id", "datetime"])
    train_data, test_data = split_data_by_user(interactions_sorted, 0.8, "Processing users")

    write_info(train_data, test_data, savepath)

    print("Splitting train data into train and validation...")
    train_data, val_data = split_data_by_user(train_data.sort(["user_id", "datetime"]), 0.85, "Processing train users")

    write_samples(train_data, savepath, "train")
    write_samples(val_data, savepath, "valid")
    write_samples(test_data, savepath, "test")

def get_1k_popular_users(interactions, constant=10):
    user_activity = (
        interactions
        .group_by("user_id")
        .agg(pl.count("session_id").alias("interaction_count"))
        .sort("interaction_count", descending=True)
    )
    active_users = user_activity.head(1000)

    track_play_count = (
        interactions
        .group_by("track_id")
        .agg(pl.count("play_duration").alias("play_count"))
        .filter(pl.col("play_count") > constant)
    )

    filtered_interactions = (
        interactions
        .join(active_users, on="user_id", how="inner")
        .join(track_play_count, on="track_id", how="inner")
    )

    final_users = active_users
    final_items = track_play_count

    print("Final Users:")
    print(len(final_users))
    print("Final Items:")
    print(len(final_items))

    return filtered_interactions


if __name__ == "__main__":
    rootpath = "data/SberZvuk/files/zvuk-interactions.parquet"
    savepath = "data/SberZvuk/data/1kSberData"

    print("Loading logs...")
    log_data = pl.read_parquet(rootpath)
    log_data = get_1k_popular_users(log_data, constant=5000)

    print("Processing data...")
    preprocess_data(log_data, savepath)
