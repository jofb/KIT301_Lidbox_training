import pandas as pd
import tensorflow as tf
import numpy as np
import os

class dataframe_processing:
    def __init__(self, input_dir, label_targets):
        self.input_dir = input_dir
        self.label_targets = label_targets
        
    def expand_metadata(self, row):
        """
        Update dataframe row by generating a unique utterance id,
        expanding the absolute path to the mp3 file,
        and adding an integer target for the label.
        """
        row.id = "{:s}_{:s}".format(
            row.path.split(".mp3", 1)[0].split("common_voice_", 1)[1],
            row.split)
        row.path = os.path.join(self.input_dir, row.lang, "clips", row.path)
        row.target = self.label_targets[row.lang]
        return row

    def tsv_to_lang_dataframe(self, lang, split):
        """
        Given a language and dataset split (train, dev, test),
        load the Common Voice metadata tsv-file from disk into a pandas.DataFrame.
        Preprocess all rows by dropping unneeded columns and adding new metadata.
        """
        df = pd.read_csv(
            os.path.join(self.input_dir, lang, split + ".tsv"),
            sep='\t',
            # We only need these columns from the metadata
            usecols=("client_id", "path", "sentence"))

        # Add language label as column
        df.insert(len(df.columns), "lang", lang)
        # Add split name to every row for easier filtering
        df.insert(len(df.columns), "split", split)
        # Add placeholders for integer targets and utterance ids generated row-wise
        df.insert(len(df.columns), "target", -1)
        df.insert(len(df.columns), "id", "")
        # Create new metadata columns
        df = df.transform(self.expand_metadata, axis=1)
        return df
    
    def assert_splits_disjoint_by_speaker(self, meta, split_names):
        split2spk = {split: set(meta[meta["split"]==split].client_id.to_numpy())
                    for split in split_names}

        for split, spk in split2spk.items():
            print("split {} has {} speakers".format(split, len(spk)))

        print()
        print("asserting all are disjoint")
        assert split2spk["train"] & split2spk["test"] == set(), "train and test, mutual speakers"
        assert split2spk["train"] & split2spk["dev"]  == set(), "train and dev, mutual speakers"
        assert split2spk["dev"]   & split2spk["test"] == set(), "dev and test, mutual speakers"
        print("ok")
    



# random oversampling to ensure data is better distributed
def random_oversampling(meta, np_rng):
    groupby_lang = meta[["lang", "duration"]].groupby("lang")
    
    total_dur = groupby_lang.sum()
    target_lang = total_dur.idxmax()[0]
    
    total_dur_delta = total_dur.loc[target_lang] - total_dur
    
    median_dur = groupby_lang.median()
    
    sample_sizes = (total_dur_delta / median_dur).astype(np.int32)
    
    samples = []
    
    for lang in groupby_lang.groups:
        sample_size = sample_sizes.loc[lang][0]
        sample = (meta[meta["lang"]==lang]
                .sample(n=sample_size, replace=True, random_state=np_rng.bit_generator)
                .reset_index()
                .transform(update_sample_id, axis=1))
        samples.append(sample)

    return pd.concat(samples).set_index("id", drop=True, verify_integrity=True)



def update_sample_id(row):
    row["id"] = "{}_copy_{}".format(row["id"], row.name)
    return row
    
    
    
# load into a tf dataset
def metadata_to_dataset_input(meta):   
    # Create a mapping from column names to all values under the column as tensors
    return {
        "id": tf.constant(meta.index, tf.string),
        "path": tf.constant(meta.path, tf.string),
        "lang": tf.constant(meta.lang, tf.string),
        "target": tf.constant(meta.target, tf.int32),
        "split": tf.constant(meta.split, tf.string),
    }
