import numpy as np
import tensorflow as tf
import os
import pandas as pd
import miniaudio
import lidbox.models.xvector as xvector
TF_AUTOTUNE = tf.data.experimental.AUTOTUNE

import dataset_processing as dp
import audio_processing as ap

# consistent random seeds
np_rng = np.random.default_rng(1)
tf.random.set_seed(np_rng.integers(0, tf.int64.max))

# classification labels, for our case this is a list of languages
# estonian, mongolion, tamil, turkish
languages = """
    et
    mn
    ta
    tr
""".split()
languages = sorted(l.strip() for l in languages)

''' Set your output and input directories here '''
# input contains datasets, output contains model checkpoints and exports
input_dir = "../lidbox/data/cv-corpus"
output_dir = "./output/data/exp/cv4"

# assert that the directories exist
os.makedirs(output_dir, exist_ok=True)
assert os.path.isdir(input_dir), input_dir + " does not exist"

# validating the language code list against the data directory
dirs = sorted((f for f in os.scandir(input_dir) if f.is_dir()), key=lambda f: f.name)
missing_languages = set(languages) - set(d.name for d in dirs)
assert missing_languages == set(), "missing languages: {}".format(missing_languages)

# maps labels to ordered numbers
target2lang = tuple(sorted(languages))
lang2target = {lang: target for target, lang in enumerate(target2lang)}


''' Dataset Processing for batch training '''
# we need to split the dataset into three sections
split_names = ("train", "dev", "test")

# class used for dataset processing methods
df = dp.dataframe_processing(input_dir, lang2target)

# Concatenate metadata for all 4 languages into a single table for each split
splits = [pd.concat([df.tsv_to_lang_dataframe(lang, split) for lang in target2lang])
          for split in split_names]

# Concatenate split metadata into a single table, indexed by utterance ids
meta = (pd.concat(splits)
        .set_index("id", drop=True, verify_integrity=True)
        .sort_index())
del splits


df.assert_splits_disjoint_by_speaker(meta, split_names)

# checking that all audio files exist
for uttid, row in meta.iterrows():
    assert os.path.exists(row["path"]), row["path"] + " does not exist"
print("all audio files exist")

# adding duration to the metadata
meta["duration"] = np.array([
    miniaudio.mp3_get_file_info(path).duration for path in meta.path], np.float32)


# Augment training set metadata
meta = pd.concat([dp.random_oversampling(meta[meta["split"]=="train"], np_rng), meta]).sort_index()

assert not meta.isna().any(axis=None), "NaNs in metadata after augmentation"
df.assert_splits_disjoint_by_speaker(meta, split_names)

# at this point the dataset is sufficiently sorted



''' Model pipelines and methods '''

def pipeline_from_metadata(data, shuffle=False):
    """
	Full pipeline from audio metadata into input usable by the NN.
	Note the use of ap, the audio_processing file that contains all methods for audio processing
	"""
    if shuffle:
        # Shuffle metadata to get an even distribution of labels
        data = data.sample(frac=1, random_state=np_rng.bit_generator)
    ds = (
        # Initialize dataset from metadata
        tf.data.Dataset.from_tensor_slices(dp.metadata_to_dataset_input(data))
        # Read mp3 files from disk in parallel
        .map(ap.read_mp3_wrapper, num_parallel_calls=TF_AUTOTUNE)
        # Apply RMS VAD to drop silence from all signals
        .map(ap.remove_silence_wrapper, num_parallel_calls=TF_AUTOTUNE)
        # Drop signals that VAD removed completely
        .filter(ap.signal_is_not_empty)
        # Extract features in parallel
        .batch(1)
        .map(ap.batch_extract_features, num_parallel_calls=TF_AUTOTUNE)
        .unbatch()
    )
    return ds

# grabs the correct input type (logmelspec vs mfcc)
def as_model_input(x):
    return x[model_input_type], x["target"]


def create_model(num_freq_bins, num_labels):
    """
    Creates the model to train on from a tf Keras model.
    """
    model = xvector.create([None, num_freq_bins], num_labels, channel_dropout_rate=0.8)
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5))
    return model



# Mapping from dataset split names to tf.data.Dataset objects
split2ds = {
    split: pipeline_from_metadata(meta[meta["split"]==split], shuffle=split=="train")
    for split in split_names
}

# cache directory for tensorboard and iterator state
cachedir = os.path.join(output_dir, "cache")

split2ds["train"] = split2ds["train"].cache(os.path.join(cachedir, "data", "train"))

# only interested in logmelspec as input
model_input_type = "logmelspec"


''' Model creation and training '''

model = create_model(
    num_freq_bins=20 if model_input_type == "mfcc" else 40,
    num_labels=len(target2lang))

# these are callback functions that will occur during training, such as creating checkpoints or stopping early
callbacks = [
    # Write scalar metrics and network weights to TensorBoard
    tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(cachedir, "tensorboard", model.name),
        update_freq="epoch",
        write_images=True,
        profile_batch=0,
    ),
    # Stop training if validation loss has not improved from the global minimum in 10 epochs
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
    ),
    # Write model weights to cache everytime we get a new global minimum loss value
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(cachedir, "model", model.name),
        monitor='val_loss',
        save_weights_only=True,
        save_best_only=True,
        verbose=1,
    ),
]

# get our respective datasets
train_ds = split2ds["train"].map(as_model_input).shuffle(1000)
dev_ds = split2ds["dev"].cache(os.path.join(cachedir, "data", "dev")).map(as_model_input)

# train the model
history = model.fit(
    train_ds.batch(1),
    validation_data=dev_ds.batch(1),
    callbacks=callbacks,
    verbose=2,
    epochs=100)


# TODO here we should be doing batch testing with split2ds["test"] but instead we simply just train the model

# test_ds = split2ds["test"].map(lambda x: dict(x, input=x["logmelspec"])).batch(1)

# # print(test_ds)
# _ = model.load_weights(os.path.join(cachedir, "model", model.name))
# # # utt2pred = predict_with_keras_model(model, test_ds)

# # test_meta = meta[meta["split"]=="test"]
# # # assert not test_meta.join(utt2pred).isna().any(axis=None), "missing predictions"
# # # test_meta = test_meta.join(utt2pred)
# # test_meta
