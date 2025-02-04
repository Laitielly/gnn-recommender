import polars as pl
import os
from tqdm import tqdm
from datetime import datetime


def reindex_items_across_splits(train_data, test_data):
    print("Reindex items in train data")

    unique_train_items = train_data["video_id"].unique().to_list()
    item_mapping_df = pl.DataFrame(
        {"video_id": unique_train_items, "new_id": range(len(unique_train_items))}
    )

    train_data = (
        train_data.join(item_mapping_df, on="video_id", how="inner")
        .with_columns(pl.col("new_id").alias("video_id"))
        .drop("new_id")
    )

    print("Reindex items in test data with train mapping")

    test_data = (
        test_data.join(item_mapping_df, on="video_id", how="inner")
        .with_columns(pl.col("new_id").alias("video_id"))
        .drop("new_id")
    )

    print("Reindex has benn finished")
    return train_data, test_data, item_mapping_df


def preprocess_data(log_data, savepath):
    print("Removing lines with gaps")
    log_data = log_data.drop_nulls()
    print(f"After deleting the gaps, there are {log_data.shape[0]} records.")

    print("Indexing users")
    unique_users = log_data["user_id"].unique()
    user_mapping = {uid: idx for idx, uid in enumerate(tqdm(unique_users, desc="Indexing users"))}

    user_index_df = pl.DataFrame(
        {
            "user_id_original": list(user_mapping.keys()),
            "user_id_indexed": list(user_mapping.values()),
        }
    )

    log_data = (
        log_data.join(user_index_df, left_on="user_id", right_on="user_id_original", how="inner")
        .with_columns(pl.col("user_id_indexed").alias("user_id"))
        .select(
            [
                "user_id",
                "video_id",
                "date",
                *(col for col in log_data.columns if col not in ["user_id", "video_id", "date"]),
            ]
        )
    )

    print(f"After indexing there are {log_data.shape[0]} records")

    print("Adding timestamp...")
    timestamps = [
        int(datetime.strptime(str(date), "%Y%m%d").timestamp())
        for date in log_data["date"].to_list()
    ]

    log_data = log_data.with_columns(pl.Series("timestamp", timestamps)).drop("date")

    print("Adding feedback")
    log_data = log_data.with_columns(
        pl.when(
            (pl.col("is_hate") == 1)
            | (
                (pl.col("is_like") == 0)
                & (pl.col("is_follow") == 0)
                & (pl.col("is_comment") == 0)
                & (pl.col("is_forward") == 0)
                & (pl.col("long_view") == 0)
                & (pl.col("play_time_ms") < 0.5 * pl.col("duration_ms"))
            )
        )
        .then(0)
        .otherwise(1)
        .alias("feedback")
    ).select(["user_id", "video_id", "timestamp", "feedback"])

    print("Split data")
    train_data, test_data = [], []

    for user_id, group in log_data.group_by("user_id"):
        group = group.sort("timestamp")
        n = group.shape[0]
        train_end = int(n * 0.8)
        train_data.append(group[:train_end])
        test_data.append(group[train_end:])

    train_data = pl.concat(train_data)
    test_data = pl.concat(test_data)

    print(f"Train: {train_data.shape[0]}, Test: {test_data.shape[0]}.")

    print("Filtering data on test")
    train_users = set(train_data["user_id"].unique())
    train_items = set(train_data["video_id"].unique())

    test_data = test_data.filter(
        pl.col("user_id").is_in(train_users) & pl.col("video_id").is_in(train_items)
    )

    print(f"After filtering: Test: {test_data.shape[0]}.")

    print("Reindex items")
    train_data, test_data, item_mapping_df = reindex_items_across_splits(train_data, test_data)

    os.makedirs(savepath, exist_ok=True)

    with open(os.path.join(savepath, "train.dat"), "w") as f:
        for user_id, group in train_data.group_by("user_id"):
            for row in group.iter_rows(named=True):
                f.write(f"{row['user_id']} {row['feedback']} {row['video_id']}\n")
    print("train.dat saved")

    with open(os.path.join(savepath, "test.dat"), "w") as f:
        for user_id, group in test_data.group_by("user_id"):
            for row in group.iter_rows(named=True):
                f.write(f"{row['user_id']} {row['feedback']} {row['video_id']}\n")
    print("test.dat  saved")

    with open(os.path.join(savepath, "user_list.dat"), "w") as f:
        for original_id, indexed_id in user_mapping.items():
            f.write(f"{original_id}\t{indexed_id}\n")
    print("user_list.dat saved")

    with open(os.path.join(savepath, "item_list.dat"), "w") as f:
        for row in item_mapping_df.iter_rows(named=True):
            f.write(f"{row['video_id']}\t{row['new_id']}\n")
    print("item_list.dat saved")

    print("KG generation")
    #    user_mapping_df = pl.DataFrame(
    #        {"original_id": list(user_mapping.keys()), "indexed_id": list(user_mapping.values())}
    #    )

    with open(os.path.join(savepath, "kg_final.txt"), "w") as f:
        for dataset in [train_data, test_data]:
            for row in dataset.iter_rows(named=True):
                f.write(f"{row['user_id']} {row['feedback']} {row['video_id']}\n")
    print("kg_final.txt saved")


rootpath = "~/research_gnn/data/shorts/KuaiRand-1K/data/"
savepath = "~/research_gnn/models/KGPolicy/data/data/"

print("loading logs")
big_log_matrix = pl.read_csv(rootpath + "log_random_4_22_to_5_08_1k.csv")
small_log_matrix1 = pl.read_csv(rootpath + "log_standard_4_08_to_4_21_1k.csv")
small_log_matrix2 = pl.read_csv(rootpath + "log_standard_4_22_to_5_08_1k.csv")

print("concat logs")
original_log_data = pl.concat([big_log_matrix, small_log_matrix1, small_log_matrix2])


popular_video_ids = (
    original_log_data.group_by("video_id")
    .agg(pl.count("video_id").alias("count"))
    .sort("count", descending=True)
    .limit(20000)["video_id"]
)

log_data = original_log_data.filter(pl.col("video_id").is_in(popular_video_ids))

preprocess_data(log_data, savepath)
