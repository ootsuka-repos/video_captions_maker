import os
import cv2
import subprocess

# 入出力ディレクトリを指定
input_dir = r"C:\Users\user\Downloads\data"
output_dir = r"C:\Users\user\Downloads\outputs"

# 出力ディレクトリを作成
os.makedirs(output_dir, exist_ok=True)

# 解像度を設定
target_long = 720
target_short = 480

# 最大FPSを設定
max_fps = 60


# フレーム数を調整する
def process_video(input_path, output_path, aspect_ratio):
    try:
        # 動画を読み込む
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise Exception(f"Failed to open video file: {input_path}")

        # 動画のプロパティを取得
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        # FPSが60以上の場合は60に制限
        if fps > max_fps:
            fps = max_fps

        # 縦長か横長かを判定して解像度を調整
        if aspect_ratio > 1:  # 横長
            resized_width = target_long
            resized_height = target_short
            folder_name = f"{resized_width}x{resized_height}"
        else:  # 縦長
            resized_height = target_long
            resized_width = target_short
            folder_name = f"{resized_width}x{resized_height}"

        # 出力フォルダを作成
        resolution_folder_path = os.path.join(output_dir, folder_name)
        videos_folder_path = os.path.join(resolution_folder_path, "videos")

        os.makedirs(videos_folder_path, exist_ok=True)

        # 出力パスを更新
        output_path = os.path.join(videos_folder_path, os.path.basename(output_path))

        # フレーム数を確認して調整
        new_frame_count = total_frames  # 初期化
        if total_frames % 4 != 0 and (total_frames + 1) % 4 != 0:
            new_frame_count = (total_frames // 4) * 4
            if new_frame_count < total_frames:
                new_frame_count += 4
        new_duration = new_frame_count / fps

        # ffmpegを使って動画を処理
        command = [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-vf",
            f"scale={resized_width}:{resized_height}",
            "-t",
            str(new_duration),
            "-r",
            str(fps),
            "-c:v",
            "libx264",
            output_path,
        ]

        subprocess.run(command, check=True)

        # 処理後の解像度とフレーム数を表示
        print(f"Processed video resolution: {resized_width}x{resized_height}")
        print(f"Processed video frame count: {new_frame_count}")
        print(f"Processed video FPS: {fps}")

        cap.release()
        return output_path
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return None


# GIFをMP4に変換する関数
def convert_gif_to_mp4(input_path, output_path, aspect_ratio):
    try:
        # 動画のアスペクト比を取得
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise Exception(f"Failed to open GIF file: {input_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        aspect_ratio = width / height
        cap.release()

        # 縦長か横長かを判定して解像度を調整
        if aspect_ratio > 1:  # 横長
            resized_width = target_long
            resized_height = target_short
            folder_name = f"{resized_width}x{resized_height}"
        else:  # 縦長
            resized_height = target_long
            resized_width = target_short
            folder_name = f"{resized_width}x{resized_height}"

        # 出力フォルダを作成
        resolution_folder_path = os.path.join(output_dir, folder_name)
        videos_folder_path = os.path.join(resolution_folder_path, "videos")

        os.makedirs(videos_folder_path, exist_ok=True)

        # 出力パスを更新
        output_path = os.path.join(videos_folder_path, os.path.basename(output_path))

        # 解像度を偶数に調整
        command = [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-vf",
            f"scale={resized_width}:{resized_height}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            output_path,
        ]
        subprocess.run(command, check=True)

        print(f"Converted GIF to MP4 with resolution: {resized_width}x{resized_height}")

        return output_path
    except Exception as e:
        print(f"Error converting GIF to MP4: {e}")
        return None


# repeat_video_if_shortを修正
def repeat_video_if_short(output_path):
    try:
        # 出力パスからフォルダ名を取得
        resolution_folder = os.path.basename(
            os.path.dirname(os.path.dirname(output_path))
        )
        videos_folder = os.path.dirname(output_path)

        # 一時ファイルのパスを作成
        temp_output_path = os.path.join(
            videos_folder, f"temp_{os.path.basename(output_path)}"
        )

        cap = cv2.VideoCapture(output_path)
        if not cap.isOpened():
            raise Exception(f"Failed to open video file: {output_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # 50フレームを超えるまで繰り返す
        if total_frames <= 50:
            # 新しい動画ファイルを作成
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

            # フレーム数を数える変数
            current_total_frames = 0

            # 動画を繰り返し書き込む
            while current_total_frames < 50:
                cap = cv2.VideoCapture(output_path)
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    out.write(frame)
                    current_total_frames += 1

                    # 50フレームを超えたら停止
                    if current_total_frames >= 50:
                        break

                cap.release()

                if current_total_frames >= 50:
                    break

            out.release()

            # 一時ファイルを元のファイルに置き換える
            os.replace(temp_output_path, output_path)
            print(f"Repeated video {output_path} until frame count is over 50.")
    except Exception as e:
        print(f"Error repeating video {output_path}: {e}")


# メインスクリプト
for file_name in os.listdir(input_dir):
    input_path = os.path.join(input_dir, file_name)
    if file_name.lower().endswith((".mp4", ".avi", ".mov", ".webm")):
        output_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}.mp4")
        print(f"Processing {file_name}...")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Failed to open video file: {input_path}")
            continue

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        aspect_ratio = width / height
        cap.release()

        # Removed indexing related to prompt and videos

        processed_output_path = process_video(input_path, output_path, aspect_ratio)
        if processed_output_path:
            print(f"Saved processed video to {processed_output_path}")
            repeat_video_if_short(processed_output_path)

    elif file_name.lower().endswith(".gif"):
        output_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}.mp4")
        print(f"Converting {file_name} to MP4...")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Failed to open GIF file: {input_path}")
            continue

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        aspect_ratio = width / height
        cap.release()

        # Removed indexing related to prompt and videos

        converted_output_path = convert_gif_to_mp4(
            input_path, output_path, aspect_ratio
        )
        if converted_output_path:
            print(f"Saved converted GIF to {converted_output_path}")
            repeat_video_if_short(converted_output_path)
