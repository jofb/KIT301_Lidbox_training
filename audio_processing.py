import miniaudio
import numpy as np
import scipy
import tensorflow as tf
from lidbox.features import cmvn
from lidbox.features.audio import spectrograms
from lidbox.features.audio import linear_to_mel
from lidbox.features.audio import framewise_rms_energy_vad_decisions

# loading in an mp3 file
def read_mp3(path, resample_rate=16000):
    if isinstance(path, bytes):
        # If path is a tf.string tensor, it will be in bytes
        path = path.decode("utf-8")
        
    f = miniaudio.mp3_read_file_f32(path)
    
    # Downsample to target rate, 16 kHz is commonly used for speech data
    new_len = round(len(f.samples) * float(resample_rate) / f.sample_rate)
    signal = scipy.signal.resample(f.samples, new_len)
    
    # Normalize to [-1, 1]
    signal /= np.abs(signal).max()
    
    return signal, resample_rate

# voice activity detection
def remove_silence(signal, rate):
    window_ms = tf.constant(10, tf.int32)
    window_frames = (window_ms * rate) // 1000
    
    # Get binary VAD decisions for each 10 ms window
    vad_1 = framewise_rms_energy_vad_decisions(
        signal=signal,
        sample_rate=rate,
        frame_step_ms=window_ms,
        # Do not return VAD = 0 decisions for sequences shorter than 300 ms
        min_non_speech_ms=300,
        strength=0.1)
    
    # Partition the signal into 10 ms windows to match the VAD decisions
    windows = tf.signal.frame(signal, window_frames, window_frames)
    # Filter signal with VAD decision == 1
    return tf.reshape(windows[vad_1], [-1])


# to log mel spectrogram given signals and rate
def logmelspectrograms(signals, rate):
    powspecs = spectrograms(signals, rate)
    melspecs = linear_to_mel(powspecs, rate, num_mel_bins=40)
    return tf.math.log(melspecs + 1e-6)

''' These wrapper functions are used for batch processing of data '''

# mp3 reader wrapper for tf
def read_mp3_wrapper(x):
    signal, sample_rate = tf.numpy_function(
        # Function
        read_mp3,
        # Argument list
        [x["path"]],
        # Return value types
        [tf.float32, tf.int32])
    return dict(x, signal=signal, sample_rate=tf.cast(sample_rate, tf.int32))


# remove silence wrapper
def remove_silence_wrapper(x):
    return dict(x, signal=remove_silence(x["signal"], x["sample_rate"]))


# extract features wrapper (logmel, mfccs)
def batch_extract_features(x):
    with tf.device("GPU"):
        signals, rates = x["signal"], x["sample_rate"]
        logmelspecs = logmelspectrograms(signals, rates[0])
        logmelspecs_smn = cmvn(logmelspecs, normalize_variance=False)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(logmelspecs)
        mfccs = mfccs[...,1:21]
        mfccs_cmvn = cmvn(mfccs)
    return dict(x, logmelspec=logmelspecs_smn, mfcc=mfccs_cmvn)


def signal_is_not_empty(x):
    return tf.size(x["signal"]) > 0
    