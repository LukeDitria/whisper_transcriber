import os
from transformers import logging

from whisper_recorder import AudioWhisper

# Disable the HuggingFace warnings
logging.set_verbosity_error()


def generate_filename(source_path, default_base='recording', ext='txt'):
    if source_path and os.path.exists(source_path):
        # Get the filename without extension
        base_name = os.path.splitext(os.path.basename(source_path))[0]
    else:
        base_name = default_base

    # Ensure unique filename by adding incremental number if needed
    counter = 1
    while True:
        output_filename = f"{base_name}_transcription_{counter}.{ext}"
        if not os.path.exists(output_filename):
            return output_filename
        counter += 1


def main():
    processor = AudioWhisper()
    temp_files = []

    try:
        # Prompt for input type
        print("\nChoose input type:")
        print("1. Record audio")
        print("2. Transcribe audio file")
        print("3. Transcribe video file")

        choice = input("Enter your choice (1/2/3): ").strip()

        source_path = None  # Track the original source file path

        if choice == '1':
            # Record audio until spacebar is pressed
            wav_file = processor.record()
            temp_files.append(wav_file)
        elif choice == '2':
            # Transcribe an existing audio file
            wav_file = input("Enter the path to the audio file: ").strip()
            source_path = wav_file
        elif choice == '3':
            # Transcribe a video file
            video_path = input("Enter the path to the video file: ").strip()
            source_path = video_path
            wav_file = processor.extract_audio_from_video(video_path)
            temp_files.append(wav_file)
        else:
            print("Invalid choice. Exiting.")
            return

        # Transcribe the recording/file if we have one
        if wav_file:
            transcription = processor.transcribe(wav_file)

            # Generate output filename based on source file
            output_txt = generate_filename(source_path)
            with open(output_txt, "w") as text_file:
                text_file.write(transcription)
            print(f"\nTranscription saved to {output_txt}")

    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")

    finally:
        # Clean up any temporary files
        processor.cleanup_temp_files(temp_files)

if __name__ == "__main__":
    main()