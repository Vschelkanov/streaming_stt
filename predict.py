import librosa
import torch
from transformers import pipeline


def transcribe(processor, model, audio):
    input_values = processor(
        audio,
        sampling_rate=16_000,
        return_tensors='pt'
    ).input_features

    with torch.no_grad():
        logits = model(input_values).logits

    pred_ids = torch.argmax(logits, dim=-1)[0]
    transcript = processor.decode(pred_ids)
    return transcript


def main():
    model_checkpoint = 'models/initial'

    asr_pipe = pipeline(model=model_checkpoint, task='automatic-speech-recognition')

    file_path = 'dataset/test/Aliboeva_augmentation_Aliboeva_telephony_audio_010.wav'
    file_path = 'dataset/test/L03Text_7.wav'
    audio, sr = librosa.load(file_path, sr=None)
    text = asr_pipe(audio)["text"]

    print(text)


if __name__ == '__main__':
    main()