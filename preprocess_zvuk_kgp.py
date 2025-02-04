import polars as pl
import os


def preprocess_zvuk_kgp(interactions_path, save_path, max_users=1000, max_tracks=20000):
    print("Загрузка данных...")
    interactions = pl.read_parquet(interactions_path)

    print("Удаление строк с пропусками...")
    interactions = interactions.drop_nulls()
    print(f"После удаления пропусков осталось записей: {interactions.shape[0]}.")

    print("Конвертация даты...")
    interactions = interactions.with_columns(
        pl.col("datetime").cast(pl.Datetime).alias("timestamp")
    )

    print("Фильтрация данных (по дате, популярности)...")
    interactions = interactions.filter(
        (pl.col("timestamp") >= pl.datetime(2023, 5, 1))
        & (pl.col("timestamp") < pl.datetime(2023, 5, 15))
    )

    print(f"Отбор топ-{max_tracks} треков по популярности...")
    popular_tracks = (
        interactions.group_by("track_id")
        .agg(pl.count("track_id").alias("count"))
        .sort("count", descending=True)
        .limit(max_tracks)["track_id"]
    )
    interactions = interactions.filter(pl.col("track_id").is_in(popular_tracks))

    print(f"Отбор топ-{max_users} пользователей по активности...")
    active_users = (
        interactions.group_by("user_id")
        .agg(pl.count("user_id").alias("count"))
        .sort("count", descending=True)
        .limit(max_users)["user_id"]
    )
    interactions = interactions.filter(pl.col("user_id").is_in(active_users))

    print("Удаление дубликатов...")
    interactions = interactions.unique(subset=["user_id", "track_id"])

    print("Добавление фидбека...")
    interactions = interactions.with_columns(
        pl.when(pl.col("play_duration") > 15.0).then(1).otherwise(0).alias("feedback")
    )

    print("Разделение данных на train и test...")
    train_data, test_data = [], []
    for user_id, group in interactions.group_by("user_id"):
        group = group.sort("timestamp")
        n = group.shape[0]
        train_end = int(0.9 * n)
        train_data.append(group[:train_end])
        test_data.append(group[train_end:])

    train_data = pl.concat(train_data).sort("user_id")
    test_data = pl.concat(test_data).sort("user_id")

    print("Фильтрация test_data: удаляем user_id и track_id, которых нет в train...")
    test_users = train_data["user_id"].unique()
    test_tracks = train_data["track_id"].unique()
    test_data = test_data.filter(pl.col("user_id").is_in(test_users))
    test_data = test_data.filter(pl.col("track_id").is_in(test_tracks))

    print("Фильтрация train_data: удаляем user_id и track_id, которых нет в test...")
    train_users = test_data["user_id"].unique()
    train_tracks = test_data["track_id"].unique()
    train_data = train_data.filter(pl.col("user_id").is_in(train_users))
    train_data = train_data.filter(pl.col("track_id").is_in(train_tracks))

    print("Индексация пользователей и треков ПОСЛЕ фильтрации...")
    unique_users = train_data["user_id"].unique()
    unique_tracks = train_data["track_id"].unique()

    user_mapping = {uid: idx for idx, uid in enumerate(unique_users)}
    last_user_id = max(user_mapping.values()) + 1
    track_mapping = {tid: idx + last_user_id for idx, tid in enumerate(unique_tracks)}

    train_data = train_data.with_columns(
        pl.col("user_id").replace(user_mapping).alias("user_id"),
        pl.col("track_id").replace(track_mapping).alias("track_id"),
    )

    test_data = test_data.with_columns(
        pl.col("user_id").replace(user_mapping).alias("user_id"),
        pl.col("track_id").replace(track_mapping).alias("track_id"),
    )

    print("Сохранение файлов...")
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, "train.dat"), "w") as f:
        for row in train_data.iter_rows(named=True):
            f.write(f"{row['user_id']} {row['feedback']} {row['track_id']}\n")

    with open(os.path.join(save_path, "test.dat"), "w") as f:
        for row in test_data.iter_rows(named=True):
            f.write(f"{row['user_id']} {row['feedback']} {row['track_id']}\n")

    with open(os.path.join(save_path, "user_list.dat"), "w") as f:
        for uid, new_id in user_mapping.items():
            f.write(f"{uid}\t{new_id}\n")

    with open(os.path.join(save_path, "item_list.dat"), "w") as f:
        for tid, new_id in track_mapping.items():
            f.write(f"{tid}\t{new_id}\n")

    print("Генерация графа знаний...")
    kg_data = [
        f"{row['user_id']} {row['feedback']} {row['track_id']}\n"
        for row in train_data.iter_rows(named=True)
    ]

    with open(os.path.join(save_path, "kg_final.txt"), "w") as f:
        f.writelines(kg_data)

    print("Предобработка завершена!")


# Пути к файлам
interactions_path = "/home/roman/research_gnn/data/music/zvuk-interactions.parquet"
save_path = "/home/roman/research_gnn/models/KGPolicy_cuda/Data/data/zvuk"

# Запуск предобработки
preprocess_zvuk_kgp(interactions_path, save_path)
