import os
import math
import cv2
import torch
from PIL import Image
from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    AutoTokenizer,
    AutoModelForCausalLM,
)


class VideoProcessor:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.blip2_processor = None
        self.blip2_model = None
        self.summarizer_tokenizer = None
        self.summarizer_model = None

    def load_models(self):
        print("Loading BLIP2 model...")
        self.blip2_processor = Blip2Processor.from_pretrained(
            self.config["blip2_model"]
        )
        self.blip2_model = Blip2ForConditionalGeneration.from_pretrained(
            self.config["blip2_model"]
        ).to(self.device)

        print("Loading Summarizer model...")
        self.summarizer_tokenizer = AutoTokenizer.from_pretrained(
            self.config["summarizer_model"]
        )
        self.summarizer_model = AutoModelForCausalLM.from_pretrained(
            self.config["summarizer_model"]
        ).to(self.device)

    def unload_models(self):
        del self.blip2_model
        del self.blip2_processor
        del self.summarizer_model
        del self.summarizer_tokenizer
        torch.cuda.empty_cache()

    def process_video(self, video_path, summary_s_file, video_paths_file):
        captions = []
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = math.ceil(total_frames * self.config["frame_interval"])

        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)

                    inputs = self.blip2_processor(
                        images=pil_image, return_tensors="pt"
                    ).to(self.device)
                    generated_ids = self.blip2_model.generate(**inputs)
                    caption = self.blip2_processor.batch_decode(
                        generated_ids, skip_special_tokens=True
                    )[0]
                    captions.append(caption)

                frame_count += 1

        finally:
            cap.release()

        # サマリーを生成
        summary = self.summarize_captions(captions)

        # 動画パスを書き込み
        video_paths_file.write(f"{os.path.abspath(video_path)}\n")

        # サマリーをタグとして書き込み
        summary_s_file.write(f"{summary}\n")

        return summary

    def summarize_captions(self, captions):
        text = " ".join(captions)
        messages = [
            {
                "role": "user",
                "content": f"Summarize the following text strictly: {text}",
            }
        ]

        tokenized = self.summarizer_tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)

        outputs = self.summarizer_model.generate(
            tokenized,
            max_new_tokens=100,
            stop_strings=["<extra_id_1>"],
            tokenizer=self.summarizer_tokenizer,
        )

        summary = self.summarizer_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary.split("<extra_id_1>")[2].strip().replace("Assistant", "").strip()

    def process_directory(self):
        video_files = [
            f
            for f in os.listdir(self.config["video_dir"])
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
        ]

        if not video_files:
            print("No video files found.")
            return

        self.load_models()

        # 出力ファイルのパスを入力フォルダ直下に設定
        summary_s_path = os.path.join(self.config["video_dir"], "summary_s.txt")
        video_paths_path = os.path.join(self.config["video_dir"], "video_paths.txt")

        try:
            with open(summary_s_path, "w", encoding="utf-8") as tags_file, open(
                video_paths_path, "w", encoding="utf-8"
            ) as paths_file:

                for video_file in video_files:
                    video_path = os.path.join(self.config["video_dir"], video_file)
                    print(f"Processing: {video_file}")

                    summary = self.process_video(video_path, tags_file, paths_file)
                    print(f"Summary for {video_file}: {summary}")

        finally:
            self.unload_models()


if __name__ == "__main__":
    config = {
        "video_dir": r"C:\Users\user\Downloads\outputs\480x720\videos",
        "blip2_model": "Salesforce/blip2-opt-2.7b-coco",
        "summarizer_model": "nvidia/Mistral-NeMo-Minitron-8B-Instruct",
        "frame_interval": 0.05,
    }

    processor = VideoProcessor(config)
    processor.process_directory()
