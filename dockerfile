# ベースとなるイメージを指定
FROM python:3.10.12

# 作業ディレクトリを設定
WORKDIR /app

# 必要なパッケージをインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ソースコードをコピー
COPY ./app /app

# Uvicornでアプリを実行
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
