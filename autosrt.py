from faster_whisper import WhisperModel, utils
from pathlib import Path

def format_timestamp(start, end):
    # Helper function to format time in hours:minutes:seconds,milliseconds
    def format_time(t):
        hours, remainder = divmod(t, 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = int((seconds - int(seconds)) * 1000)
        return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{milliseconds:03}"

    start_formatted = format_time(start)
    end_formatted = format_time(end)
    return f"{start_formatted} --> {end_formatted}"

def generate_subtitles(segments, word_timestamps, keep_gaps):
    timestamps = []
    if word_timestamps:
        words = [word for segment in segments for word in segment.words]
        word_counter = 1
        previous_end_time = 0
        for word in words:
            split_words = word.word.strip().split()
            for split_word in split_words:
                start_time = previous_end_time if not keep_gaps else word.start
                end_time = word.end if split_word == split_words[-1] else (previous_end_time + 0.1 if not keep_gaps else word.end)
                timestamp = format_timestamp(start_time, end_time)
                timestamps.append(f"{word_counter}\n{timestamp}\n{split_word}\n")
                word_counter += 1
                previous_end_time = end_time
    else:
        for i, segment in enumerate(segments[:-1]):
            next_segment_start = segments[i + 1].start
            timestamps.append(f"{i+1}\n{format_timestamp(segment.start, next_segment_start)}\n{segment.text.strip()}\n")
        last_segment = segments[-1]
        timestamps.append(f"{len(segments)}\n{format_timestamp(last_segment.start, last_segment.end)}\n{last_segment.text.strip()}\n")

    return timestamps

def transcribe_video(video_path: str, model_id: str, split_on_word: bool = False, keep_gaps: bool = False):
    whisper_dir = Path(__file__).resolve().parent / "whisper"
    model_dir = Path(f"{whisper_dir}/model_{model_id}")
    model_file = Path(f"{model_dir}/model.bin")
    
    if not model_file.exists():
        print(f"Downloading Whisper model '{model_id}'")
        utils.download_model(size_or_id=model_id, output_dir=model_dir)
    
    model = WhisperModel(str(model_dir), device="auto")
    
    print("Transcribing...")
    segments, _ = model.transcribe(video_path, word_timestamps=split_on_word)
    
    timestamps = generate_subtitles(segments, split_on_word, keep_gaps)
    
    filename = Path(video_path).stem
    output = str(Path(__file__).resolve().parent / "transcribed") + f"/{filename}.srt"
    with open(output, mode="w", encoding='utf-8') as file:
        for timestamp in timestamps:
            file.write(timestamp + "\n")
    print("Transcription complete!")

def get_available_models():
    whisper_dir = Path(__file__).resolve().parent / "whisper"
    available_models = []
    for model_dir in whisper_dir.iterdir():
        if model_dir.is_dir() and (model_dir / "model.bin").exists():
            available_models.append(model_dir.name.replace("model_", ""))
    return available_models

def start():
    available_models = get_available_models()
    
    print("Available models:")
    for i, model in enumerate(available_models):
        print(f"{i+1}. {model}")
    print("0. Download a new model")

    choice = int(input("Select a model by entering the corresponding number:\n> "))
    
    if choice == 0:
        # Models available for download
        all_models = [
            "tiny.en", "tiny", "base.en", "base", "small.en", "small", 
            "medium.en", "medium", "large-v1", "large-v2", "large-v3", 
            "large", "distil-large-v2", "distil-medium.en", "distil-small.en", 
            "distil-large-v3"
        ]
        undownloaded_models = [model for model in all_models if model not in available_models]
        
        print("Models available:")
        for i, model in enumerate(undownloaded_models):
            print(f"{i+1}. {model}")
        
        new_model_choice = int(input("Select a model to download.\nLarger is more precise but requires higher specs:\n> "))
        selected_model = undownloaded_models[new_model_choice - 1]
    else:
        selected_model = available_models[choice - 1]

    path = input("Provide the path to the video/audio:\n> ")
    split = input("Split per-word? Y/N:\n> ")
    keep_gaps = input("Keep gaps between subtitles? Y/N:\n> ")
    
    transcribe_video(
        video_path=path, 
        model_id=selected_model, 
        split_on_word=(split.casefold() == "y"), 
        keep_gaps=(keep_gaps.casefold() == "y")
    )