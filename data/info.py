import polars as pl

file_path = "Zvuk/Zvuk_train.txt"
data = pl.read_csv(
    file_path,
    separator=" ",
    has_header=False,
    new_columns=["user_id", "item_id", "feedback"],
)


num_unique_users = data["user_id"].n_unique()
num_unique_items = data["item_id"].n_unique()
max_index = max(data["item_id"])
print(f"Количество уникальных пользователей: {num_unique_users}")
print(f"Количество уникальных элементов: {num_unique_items}")
print(f"max_index: {max_index}")
