import pyaudio
import wave
import torch
import os
from moviepy import VideoFileClip
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pynput import keyboard


class AudioWhisper:
    def __init__(self):
        print("Loading Whisper model...")
        # Whisper model setup
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model_id = "openai/whisper-large-v3"
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(self.model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            chunk_length_s=30,
            batch_size=16,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

        # Recording control
        self.recording = False
        self.frames = []

    def on_press(self, key):
        if key == keyboard.Key.space:
            if not self.recording:
                self.frames = []  # Clear previous recording
                self.recording = True
                print("\n* Recording started... Press spacebar again to stop")
            else:
                self.recording = False
                print("\n* Recording stopped")

    def record(self, output_filename="output.wav"):
        """Record audio between spacebar presses"""
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 2
        RATE = 44100

        # Set up keyboard listener
        listener = keyboard.Listener(on_press=self.on_press)
        listener.start()

        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        print("Press spacebar to start recording...")

        while True:
            if self.recording:
                try:
                    data = stream.read(CHUNK)
                    self.frames.append(data)
                except Exception as e:
                    print(f"Error recording: {e}")
                    break
            elif len(self.frames) > 0:  # Recording stopped and we have frames
                break

        stream.stop_stream()
        stream.close()
        p.terminate()
        listener.stop()

        if len(self.frames) == 0:
            print("No audio recorded")
            return None

        # Save the recorded audio
        wf = wave.open(output_filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()

        return output_filename

    def extract_audio_from_video(self, video_path, output_audio_path='temp_audio.wav'):
        """
        Extract audio from a video file

        Args:
            video_path (str): Path to the input video file
            output_audio_path (str, optional): Path to save the extracted audio. Defaults to 'temp_audio.wav'

        Returns:
            str: Path to the extracted audio file
        """
        try:
            # Load the video file
            video = VideoFileClip(video_path)

            # Extract the audio
            audio = video.audio

            # Write the audio to a file
            audio.write_audiofile(output_audio_path)

            # Close the video to free up resources
            video.close()

            return output_audio_path
        except Exception as e:
            print(f"Error extracting audio from video: {e}")
            return None

    def transcribe(self, audio_file):
        """Transcribe audio file using Whisper"""
        if audio_file is None:
            return "No audio file to transcribe"

        print("\nTranscribing...")
        result = self.pipe(audio_file)
        return result["text"]

    def cleanup_temp_files(self, files_to_remove):
        """
        Remove temporary files after processing

        Args:
            files_to_remove (list): List of file paths to remove
        """
        for file_path in files_to_remove:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error removing temporary file {file_path}: {e}")

