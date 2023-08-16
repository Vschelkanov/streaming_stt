import librosa
import torch
from transformers import pipeline
from faster_whisper import WhisperModel
import time


def main():
    file_path = 'dataset/test/L11Text_11.wav'
    audio, sr = librosa.load(file_path, sr=None)


    print('Loading HuggingFace Whisper model...')
    model_checkpoint = './models/checkpoint-300'
    asr_pipe = pipeline(model=model_checkpoint, task='automatic-speech-recognition', device='cpu')
    s_time = time.time()
    text = asr_pipe(audio)["text"]
    i_time = time.time() - s_time
    print(f'Transcript: {text}')
    print(f'Inference Time: {i_time} seconds')
    print('-' * 50, '\n')


    # ct2-transformers-converter --model ./models/checkpoint-300 --output_dir ./models/ctranslate2-whisper

    print('Loading Faster Whisper model...')
    model_checkpoint = './models/ctranslate2-whisper'

    faster_whisper_model = WhisperModel(model_size_or_path=model_checkpoint, device='cpu', compute_type='int8')
    s_time = time.time()
    segments, info = faster_whisper_model.transcribe(audio, beam_size=5)
    i_time = time.time() - s_time
    for segment in segments:
        print(f'Transcript: {segment.text}')
    print(f'Inference Time: {i_time} seconds')


if __name__ == '__main__':
    main()