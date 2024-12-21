import os
import math
import cv2
import numpy as np
import onnxruntime as rt
import pandas as pd
from PIL import Image
import huggingface_hub
from collections import Counter

# グローバル変数としてモデルをキャッシュ
GLOBAL_MODEL = None
GLOBAL_TAGS_DF = None

MODEL_REPO = "SmilingWolf/wd-eva02-large-tagger-v3"
MODEL_FILENAME = "model.onnx"
LABEL_FILENAME = "selected_tags.csv"


def load_model(model_repo=MODEL_REPO):
    """
    モデルを読み込み、キャッシュする関数

    Args:
        model_repo (str): モデルのリポジトリパス

    Returns:
        tuple: モデルセッション、タグデータフレーム
    """
    global GLOBAL_MODEL, GLOBAL_TAGS_DF

    # モデルがまだキャッシュされていない場合のみ読み込む
    if GLOBAL_MODEL is None or GLOBAL_TAGS_DF is None:
        # モデルとラベルをHugging Faceからダウンロード
        csv_path = huggingface_hub.hf_hub_download(model_repo, LABEL_FILENAME)
        model_path = huggingface_hub.hf_hub_download(model_repo, MODEL_FILENAME)

        # タグ情報を読み込む
        GLOBAL_TAGS_DF = pd.read_csv(csv_path)
        tag_names, rating_indexes, general_indexes, character_indexes = load_labels(
            GLOBAL_TAGS_DF
        )

        # モデルを読み込む
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        GLOBAL_MODEL = rt.InferenceSession(model_path, providers=providers)

    return GLOBAL_MODEL, GLOBAL_TAGS_DF


def load_labels(dataframe):
    """
    データフレームからラベル情報を読み込む関数

    Args:
        dataframe (pandas.DataFrame): タグ情報が含まれるデータフレーム

    Returns:
        tuple: タグ名のリストと各カテゴリのインデックスリスト
    """
    # タグ名から'_'を削除して読みやすくする
    name_series = dataframe["name"].map(lambda x: x.replace("_", " "))
    tag_names = name_series.tolist()

    # カテゴリごとのインデックスを取得
    # カテゴリ9: 評価タグ
    # カテゴリ0: 一般タグ
    # カテゴリ4: キャラクタータグ
    rating_indexes = list(np.where(dataframe["category"] == 9)[0])
    general_indexes = list(np.where(dataframe["category"] == 0)[0])
    character_indexes = list(np.where(dataframe["category"] == 4)[0])

    return tag_names, rating_indexes, general_indexes, character_indexes


def predict_image_tags(
    image_path, model_repo=MODEL_REPO, general_thresh=0.55, character_thresh=0.85
):
    """
    画像からタグを予測する関数

    Args:
        image_path (str): 画像ファイルのパス
        model_repo (str): 使用するモデルのリポジトリ
        general_thresh (float): 一般タグの閾値 (デフォルト: 0.55)
        character_thresh (float): キャラクタータグの閾値 (デフォルト: 0.85)

    Returns:
        dict: タグ情報を含む辞書
    """
    # モデルとタグ情報を読み込む
    model, tags_df = load_model(model_repo)

    # タグ情報を読み込む
    tag_names, rating_indexes, general_indexes, character_indexes = load_labels(tags_df)

    # モデルの入力サイズを取得
    _, height, width, _ = model.get_inputs()[0].shape

    # 画像を準備
    image = Image.open(image_path)

    # RGBモードでない場合は変換
    if image.mode != "RGB":
        image = image.convert("RGB")

    # 画像を正方形にパディングとリサイズ
    image_shape = image.size
    max_dim = max(image_shape)
    pad_left = (max_dim - image_shape[0]) // 2
    pad_top = (max_dim - image_shape[1]) // 2

    padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    padded_image.paste(image, (pad_left, pad_top))
    padded_image = padded_image.resize((height, width), Image.BICUBIC)

    # NumPy配列に変換
    image_array = np.asarray(padded_image, dtype=np.float32)
    image_array = image_array[:, :, ::-1]  # RGBからBGRに変換
    image_array = np.expand_dims(image_array, axis=0)

    # タグ予測
    input_name = model.get_inputs()[0].name
    label_name = model.get_outputs()[0].name
    preds = model.run([label_name], {input_name: image_array})[0]

    # タグと予測確率を紐付け
    labels = list(zip(tag_names, preds[0].astype(float)))

    # 評価タグの処理
    ratings_names = [labels[i] for i in rating_indexes]
    rating = dict(ratings_names)

    # 一般タグの処理
    # 指定された閾値以上の確率を持つタグのみを選択
    general_names = [labels[i] for i in general_indexes]
    general_res = [x for x in general_names if x[1] > general_thresh]
    general_res = dict(general_res)

    # キャラクタータグの処理
    # 指定された閾値以上の確率を持つタグのみを選択
    character_names = [labels[i] for i in character_indexes]
    character_res = [x for x in character_names if x[1] > character_thresh]
    character_res = dict(character_res)

    # 一般タグを確率の高い順にソートして文字列化
    sorted_general_strings = sorted(
        general_res.items(),
        key=lambda x: x[1],
        reverse=True,
    )
    sorted_general_strings = [x[0] for x in sorted_general_strings]
    sorted_general_strings = ", ".join(sorted_general_strings)

    # 結果を辞書形式で返す
    return {
        "tags": sorted_general_strings,  # 確率の高い一般タグをカンマ区切りの文字列で
        "rating": rating,  # 評価タグ（例：セーフ、R-18など）
        "characters": character_res,  # キャラクタータグと確率
        "general_tags": general_res,  # 一般タグと確率の辞書
    }


def find_and_filter_tags(tag_counts, total_lines, output_folder):
    """
    タグのフィルタリングと選択を行い、selected_tags.txt に書き込む関数

    Args:
        tag_counts (Counter): タグの出現回数
        total_lines (int): 総動画数
        output_folder (str): 出力フォルダのパス
    """
    # 各タグの割合を計算
    tag_ratios = {tag: count / total_lines for tag, count in tag_counts.items()}

    # 70%以上出現するタグを選択
    selected_tags = [tag for tag, ratio in tag_ratios.items() if ratio >= 0.7]

    # 選択されたタグをselected_tags.txtに書き込む
    selected_tags_file = os.path.join(output_folder, "selected_tags.txt")
    with open(selected_tags_file, "w", encoding="utf-8") as f:
        f.write(", ".join(selected_tags))

    print(f"Selected tags: {', '.join(selected_tags)}\n")


def generate_video_captions(
    video_path: str,
    video_paths_file,
    selected_tags_file,
    frame_interval_percentage: float = 0.05,
):
    """
    動画からフレームを抽出し、タグを生成して書き込む関数
    captions.txt の使用を廃止し、selected_tags.txt と video_paths.txt に直接書き込む

    Args:
        video_path (str): 動画ファイルのパス
        video_paths_file (file object): video_paths.txt のファイルオブジェクト
        selected_tags_file (file object): selected_tags.txt のファイルオブジェクト
        frame_interval_percentage (float): フレーム間隔の割合 (デフォルト: 0.05)
    """
    # ビデオキャプチャーオブジェクトの作成
    cap = cv2.VideoCapture(video_path)

    # フレームカウンター
    frame_count = 0
    saved_frame_count = 0

    # 動画の詳細情報
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps != 0 else 0

    # フレーム間隔を動画の総フレーム数に基づいて計算
    frame_interval = math.ceil(total_frames * frame_interval_percentage)
    frame_interval = max(frame_interval, 1)  # フレーム間隔が0にならないように

    print(f"動画情報: {total_frames}フレーム, {fps:.2f} FPS, 長さ: {duration:.2f}秒")
    print(f"フレーム間隔: {frame_interval}フレーム")

    tag_counter = Counter()

    try:
        while True:
            # フレームの読み込み
            ret, frame = cap.read()

            # フレームがない場合は終了
            if not ret:
                break

            # 指定されたインターバルでフレーム抽出
            if frame_count % frame_interval == 0:
                # 一時的な一意のフレーム画像を保存
                temp_frame_path = os.path.join(
                    os.path.dirname(selected_tags_file.name),
                    f"temp_frame_{saved_frame_count}.jpg",
                )
                cv2.imwrite(temp_frame_path, frame)

                # タグ予測を実行
                results = predict_image_tags(temp_frame_path)

                # タグをカウント
                tags = results["tags"].split(", ")
                tag_counter.update(tags)

                # 一時ファイルを削除
                os.remove(temp_frame_path)

                print(f"フレーム {saved_frame_count} のタグ: {results['tags']}")

                saved_frame_count += 1

            frame_count += 1

        # 動画パスをvideo_paths.txtに書き込む
        video_paths_file.write(f"{os.path.abspath(video_path)}\n")

        # 動画ごとの選択タグをselected_tags.txtに書き込く
        # 70%以上出現するタグを選択
        total_tags = saved_frame_count
        selected_tags = [
            tag for tag, count in tag_counter.items() if count / total_tags >= 0.7
        ]
        selected_tags_file.write(f"{', '.join(selected_tags)}\n")

        print(
            f"{os.path.basename(video_path)} の処理が完了しました。選択タグ: {', '.join(selected_tags)}\n"
        )

    finally:
        # ビデオキャプチャーを必ず解放
        cap.release()


def process_all_videos_in_folder(
    folder_path: str, frame_interval_percentage: float = 0.05
):
    """
    指定されたフォルダ内のすべての動画を処理してタグを生成する関数

    Args:
        folder_path (str): 処理するフォルダのパス
        frame_interval_percentage (float): フレーム間隔の割合 (デフォルト: 0.05)
    """
    video_files = [
        f
        for f in os.listdir(folder_path)
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    ]

    if not video_files:
        print("指定されたフォルダに動画ファイルが見つかりませんでした。")
        return

    print(f"フォルダ内の動画ファイル: {video_files}")

    selected_tags_file_path = os.path.join(folder_path, "selected_tags.txt")
    video_paths_file_path = os.path.join(folder_path, "video_paths.txt")

    # selected_tags.txt を初期化（既存の場合は上書き）
    with open(
        selected_tags_file_path, "w", encoding="utf-8"
    ) as selected_tags_file, open(
        video_paths_file_path, "w", encoding="utf-8"
    ) as video_paths_file:

        for video_file in video_files:
            video_path = os.path.join(folder_path, video_file)
            print(f"処理中: {video_file}")
            try:
                generate_video_captions(
                    video_path=video_path,
                    video_paths_file=video_paths_file,
                    selected_tags_file=selected_tags_file,
                    frame_interval_percentage=frame_interval_percentage,
                )
            except Exception as e:
                print(f"{video_file} の処理中にエラーが発生しました: {e}")

    print("全ての動画の処理が完了しました。")


# 使用例
if __name__ == "__main__":
    folder_path = r"C:\Users\user\Downloads\outputs\480x720\videos"

    process_all_videos_in_folder(
        folder_path=folder_path,
        frame_interval_percentage=0.05,
    )
