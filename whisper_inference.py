import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers import pipeline


class WhisperInference:
    def __init__(self, model_name):
        self.device = "cpu"
        self.pipe = pipeline(model=model_name, task='automatic-speech-recognition')

        # self.processor = WhisperProcessor.from_pretrained(model_name)
        # self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        # self.model.to(self.device)

    def buffer_to_text(self, audio_buffer):
        return self.pipe(audio_buffer)["text"].strip(), None

    # def sh_buffer_to_text(self, audio_buffer):
    #     if len(audio_buffer) == 0:
    #         return ""
    #
    #     inputs = self.processor(
    #         torch.tensor(audio_buffer),
    #         sampling_rate=16_000,
    #         return_tensors="pt",
    #         padding=True
    #     ).input_features
    #
    #     with torch.no_grad():
    #         predicted_ids = self.model.generate(
    #             inputs.to(self.device),
    #         )
    #
    #     transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    #
    #     return transcription, None

    def file_to_text(self, filename):
        audio_input, samplerate = sf.read(filename)
        assert samplerate == 16000
        return self.buffer_to_text(audio_input)


if __name__ == "__main__":
    print("Model test")
    asr = WhisperInference('./models/checkpoint-300')
    text = asr.file_to_text('test.wav')
    print(text)
