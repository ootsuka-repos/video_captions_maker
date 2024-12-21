import os
import cv2
import subprocess

# 入出力ディレクトリを指定
input_dir = r"C:\Users\user\Downloads\data"
output_dir = r"C:\Users\user\Downloads\outputs2"

# 解像度とFPS設定
target_long = 768
target_short = 512
max_fps = 24
max_frames = 144

# 出力ディレクトリを作成
os.makedirs(output_dir, exist_ok=True)


def get_resolution(aspect_ratio):
    """アスペクト比に基づき解像度を返す"""
    if aspect_ratio > 1:  # 横長
        return target_long, target_short
    return target_short, target_long


def create_output_folders(resized_width, resized_height):
    """出力フォルダを作成"""
    folder_name = f"{resized_width}x{resized_height}"
    resolution_folder_path = os.path.join(output_dir, folder_name)
    videos_folder_path = os.path.join(resolution_folder_path, "videos")
    os.makedirs(videos_folder_path, exist_ok=True)
    return videos_folder_path


def build_ffmpeg_command(input_path, output_path, width, height, fps, duration=None):
    """ffmpegコマンドを構築"""
    command = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-vf",
        f"scale={width}:{height}",
        "-r",
        str(fps),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
    ]
    if duration:
        command.extend(["-t", str(duration)])
    command.append(output_path)
    return command


def process_video(input_path, output_path, aspect_ratio):
    """動画を処理"""
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise Exception(f"Failed to open video file: {input_path}")

        fps = min(cap.get(cv2.CAP_PROP_FPS), max_fps)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        resized_width, resized_height = get_resolution(aspect_ratio)
        videos_folder_path = create_output_folders(resized_width, resized_height)

        # 出力パスを更新
        output_path = os.path.join(videos_folder_path, os.path.basename(output_path))

        # フレーム数を調整
        new_frame_count = ((total_frames + 3) // 4) * 4  # 4の倍数に調整
        new_duration = new_frame_count / fps

        # ffmpegコマンド実行
        command = build_ffmpeg_command(
            input_path, output_path, resized_width, resized_height, fps, new_duration
        )
        subprocess.run(command, check=True)

        print(f"Processed video: {output_path} ({resized_width}x{resized_height})")
        return output_path
    except Exception as e:
        print(f"Error processing video: {e}")
        return None


def convert_gif_to_mp4(input_path, output_path, aspect_ratio):
    """GIFをMP4に変換"""
    try:
        resized_width, resized_height = get_resolution(aspect_ratio)
        videos_folder_path = create_output_folders(resized_width, resized_height)
        output_path = os.path.join(videos_folder_path, os.path.basename(output_path))

        # ffmpegコマンド実行
        command = build_ffmpeg_command(
            input_path, output_path, resized_width, resized_height, max_fps
        )
        subprocess.run(command, check=True)

        print(f"Converted GIF to MP4: {output_path} ({resized_width}x{resized_height})")
        return output_path
    except Exception as e:
        print(f"Error converting GIF: {e}")
        return None


def repeat_video_if_short(output_path):
    """短い動画を繰り返しフレーム数を増やす"""
    try:
        cap = cv2.VideoCapture(output_path)
        if not cap.isOpened():
            raise Exception(f"Failed to open video file: {output_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if total_frames >= max_frames:
            return

        temp_output_path = output_path.replace(".mp4", "_temp.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

        current_total_frames = 0
        while current_total_frames < max_frames:
            cap = cv2.VideoCapture(output_path)
            while True:
                ret, frame = cap.read()
                if not ret or current_total_frames >= max_frames:
                    break
                out.write(frame)
                current_total_frames += 1
            cap.release()
        out.release()

        os.replace(temp_output_path, output_path)
        print(f"Repeated video to match frame count: {output_path}")
    except Exception as e:
        print(f"Error repeating video: {e}")


def process_files():
    """入力ディレクトリ内のファイルを処理"""
    for file_name in os.listdir(input_dir):
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}.mp4")

        if file_name.lower().endswith((".mp4", ".avi", ".mov", ".webm")):
            print(f"Processing video: {file_name}")
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                print(f"Failed to open video: {input_path}")
                continue

            aspect_ratio = cap.get(cv2.CAP_PROP_FRAME_WIDTH) / cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
            cap.release()

            processed_path = process_video(input_path, output_path, aspect_ratio)
            if processed_path:
                repeat_video_if_short(processed_path)

        elif file_name.lower().endswith(".gif"):
            print(f"Converting GIF: {file_name}")
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                print(f"Failed to open GIF: {input_path}")
                continue

            aspect_ratio = cap.get(cv2.CAP_PROP_FRAME_WIDTH) / cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
            cap.release()

            converted_path = convert_gif_to_mp4(input_path, output_path, aspect_ratio)
            if converted_path:
                repeat_video_if_short(converted_path)


# メインスクリプト
if __name__ == "__main__":
    process_files()
