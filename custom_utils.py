import tensorflow as tf
import librosa


def get_input_dict(path, crop_size=64000):
    audio, sr = librosa.load(path, duration=10)
    print(audio.shape)
    wav = tf.Variable(audio)
    print(wav.get_shape())
    crop = tf.random_crop(wav, [crop_size], name="croppped_wav")
    print(crop.get_shape())
    return {
        "pitch": tf.Variable(0, name="pitch"),
        "key": tf.Variable(0, name="key"),
        "wav": crop
    }
