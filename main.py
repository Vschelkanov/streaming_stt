from faster_whisper import WhisperModel


def main():

    model = WhisperModel(model_size_or_path='./models/ctranslate2-whisper', device="cpu", compute_type="int8")

    segments, info = model.transcribe('dataset/test/Audio_1_010.wav', beam_size=5)

    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))


if __name__ == '__main__':
    main()