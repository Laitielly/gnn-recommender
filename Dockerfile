# Используем базовый образ с Python 3.6
FROM python:3.6-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    libssl-dev \
    libffi-dev \
    zlib1g-dev \
    && apt-get clean

# Копируем все файлы проекта в контейнер
COPY . /app
