import subprocess
import sys

def install_and_import(package):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    finally:
        globals()[package] = __import__(package)

install_and_import('faster_whisper')
install_and_import('pathlib')


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

def generate_subtitles(segments, word_timestamps):
    timestamps = []
    if word_timestamps:
        words = [word for segment in segments for word in segment.words]
        word_counter = 1
        for word in words:
            split_words = word.word.strip().split()
            for split_word in split_words:
                start_time = word.start if split_word == split_words[0] else None
                end_time = word.end if split_word == split_words[-1] else None
                start_formatted = format_timestamp(start_time, word.end) if start_time else format_timestamp(word.start, word.end)
                end_formatted = format_timestamp(word.start, end_time) if end_time else format_timestamp(word.start, word.end)
                timestamps.append(f"{word_counter}\n{start_formatted} --> {end_formatted}\n{split_word}\n")
                word_counter += 1
    else:
        for i, segment in enumerate(segments[:-1]):
            next_segment_start = segments[i + 1].start
            timestamps.append(f"{i+1}\n{format_timestamp(segment.start, next_segment_start)}\n{segment.text.strip()}\n")
        last_segment = segments[-1]
        timestamps.append(f"{len(segments)}\n{format_timestamp(last_segment.start, last_segment.end)}\n{last_segment.text.strip()}\n")

    return timestamps

def transcribe_video(video_path: str, split_on_word: bool = False):
    # Automatically selects between CUDA and CPU.
    # Change model_id based on precision. Requires higher specs:

    #model_id = "tiny.en"
    #model_id = "tiny"
    #model_id = "base.en"
    #model_id = "base"
    #model_id = "small.en"
    #model_id = "small"
    #model_id = "medium.en"
    #model_id = "medium"
    #model_id = "large-v1"
    #model_id = "large-v2"
    #model_id = "large-v3"
    #model_id = "large"
    #model_id = "distil-large-v2"
    #model_id = "distil-medium.en"
    #model_id = "distil-small.en"
    model_id = "small"
    
    whisper_dir = Path(__file__).resolve().parent / "whisper"
    model_dir = Path(f"{whisper_dir}/model_{model_id}")
    model_file = Path(f"{model_dir}/model.bin")
    
    if not model_file.exists():
        print(f"Downloading Whisper model '{model_id}'")
        utils.download_model(size_or_id=model_id, output_dir=model_dir)
    
    model = WhisperModel(str(model_dir), device="auto")
    
    print("Transcribing...")
    segments, _ = model.transcribe(video_path, word_timestamps=split_on_word)
    
    timestamps = generate_subtitles(segments, split_on_word)
    
    filename = Path(video_path).stem
    output = str(Path(__file__).resolve().parent / "transcribed") + f"/{filename}.srt"
    with open(output, mode="w", encoding='utf-8') as file:
        for timestamp in timestamps:
            file.write(timestamp + "\n")
    print("Transcription complete!")

if __name__ == "__main__":
    path = input("Provide the path to the video/audio:\n> ")
    split = input("Do you want to split per-word? Y/N:\n> ")
    if split.casefold() == "y":
        transcribe_video(path, True)
    else:
        transcribe_video(path)