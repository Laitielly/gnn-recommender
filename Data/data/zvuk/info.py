import pandas as pd


# Загрузка данных из файлов

user_list_path = "user_list.dat"

item_list_path = "item_list.dat"

train_path = "train.dat"

test_path = "test.dat"

kg_final_path = "kg_final.txt"


# Загрузка и анализ user_list

with open(user_list_path, "r") as f:
    user_list = [line.strip().split() for line in f.readlines()]


# Преобразование в DataFrame

df_users = pd.DataFrame(user_list, columns=["original_id", "mapped_id"])

df_users["mapped_id"] = df_users["mapped_id"].astype(int)


# Загрузка и анализ item_list

with open(item_list_path, "r") as f:
    item_list = [line.strip().split() for line in f.readlines()]


df_items = pd.DataFrame(item_list, columns=["original_id", "mapped_id"])

df_items["mapped_id"] = df_items["mapped_id"].astype(int)


# Проверим диапазон user_id и item_id

user_min, user_max = df_users["mapped_id"].min(), df_users["mapped_id"].max()

item_min, item_max = df_items["mapped_id"].min(), df_items["mapped_id"].max()


# Загрузка train и test для проверки уникальных item_id

train_items = set()

with open(train_path, "r") as f:
    for line in f:
        user, feedback, item = map(int, line.strip().split())

        train_items.add(item)


test_items = set()

with open(test_path, "r") as f:
    for line in f:
        user, feedback, item = map(int, line.strip().split())

        test_items.add(item)


# Проверим диапазон item_id в train и test

train_min, train_max = min(train_items), max(train_items)

test_min, test_max = min(test_items), max(test_items)


# Загрузка kg_final

kg_df = pd.read_csv(kg_final_path, sep=" ", header=None, names=["head", "relation", "tail"])

kg_min, kg_max = kg_df["tail"].min(), kg_df["tail"].max()


# Выведем результаты
print(
    "user_min, user_max, item_min, item_max, train_min, train_max, test_min, test_max, kg_min, kg_max"
)
print(
    user_min,
    user_max,
    item_min,
    item_max,
    train_min,
    train_max,
    test_min,
    test_max,
    kg_min,
    kg_max,
)
