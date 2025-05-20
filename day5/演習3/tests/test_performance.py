import time
import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


def test_model_inference_speed():
    """モデルの推論速度をテストする"""
    # テストデータの生成
    X, _ = make_classification(n_samples=1000, n_features=4, random_state=42)

    # モデルの作成と学習
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, np.random.randint(0, 2, size=1000))  # ダミーのターゲット

    # 推論速度の測定
    start_time = time.time()
    model.predict(X)
    inference_time = time.time() - start_time

    # 推論は0.5秒以内に完了すべき
    assert (
        inference_time < 0.5
    ), f"推論に{inference_time:.4f}秒かかりました。0.5秒以内にすべきです。"

    print(f"推論時間 : {inference_time:.4f}秒")


def test_model_memory_usage():
    """モデルのメモリ使用量をテストする"""
    import os
    import psutil
    import pickle

    # テストデータの生成
    X, y = make_classification(n_samples=1000, n_features=4, random_state=42)

    # モデルの作成と学習
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # メモリ使用量の測定（モデルをシリアライズして）
    model_bytes = pickle.dumps(model)
    model_size = len(model_bytes) / (1024 * 1024)  # MBに変換

    # モデルサイズは10MB以下であるべき
    assert (
        model_size < 10
    ), f"モデルサイズが{model_size:.2f}MBです。10MB以下にすべきです。"

    print(f"モデルサイズ: {model_size:.2f}MB")
