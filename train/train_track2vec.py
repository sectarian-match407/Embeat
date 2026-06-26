# -*- coding: utf-8 -*-
# Written by GD Studio
# Date: 2026-05-28

import logging
import multiprocessing
import numpy as np
import os
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.word2vec import LineSentence

MAX_EPOCHS = 60
SHOW_LOSS = True
project_root = os.path.dirname(os.path.abspath(__file__))
TRAINING_DATA = f"{project_root}/spotify_playlists.txt"
OUTPUT_VECTOR = f"{project_root}/track2vec.wv"
OUTPUT_BINARY = f"{project_root}/track2vec.bin"

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO
)
if SHOW_LOSS:
    logging.getLogger("gensim").setLevel(logging.WARNING)


# Custom class for computing epoch loss
class EpochLossCallback(CallbackAny2Vec):
    def __init__(self, total_epochs: int):
        self.total_epochs = total_epochs
        self.epoch = 0
        self.prev_loss = np.float64(0)
        self.prev_vectors = None

    def on_epoch_begin(self, model=None):
        if model is None:
            return
        self.prev_vectors = model.wv.vectors.copy()

    def on_epoch_end(self, model=None):
        if model is None:
            return
        cumulative_loss = np.float64(model.get_latest_training_loss())
        epoch_loss = cumulative_loss - self.prev_loss
        self.prev_loss = cumulative_loss
        mean_delta = np.mean(np.linalg.norm(model.wv.vectors - self.prev_vectors, axis=1))
        logging.info(f"Epoch {self.epoch}/{self.total_epochs} | gensim_loss={epoch_loss:.4f} | mean_delta={mean_delta:.4f}")
        self.epoch = self.epoch + 1
        return


# Training entry
def main():
    print(f"Loading training data from: {TRAINING_DATA}")
    if not os.path.isfile(TRAINING_DATA):
        raise ValueError(f"training_data is missing and exit: {TRAINING_DATA}")
    sentences = LineSentence(TRAINING_DATA)
    print("Start training...")
    workers = multiprocessing.cpu_count()
    if SHOW_LOSS:
        compute_loss = True
        callbacks = [EpochLossCallback(MAX_EPOCHS)]
    else:
        compute_loss = False
        callbacks = None
    model = Word2Vec(
        sentences=sentences,
        vector_size=64,
        min_count=10,
        window=100,
        sg=1,
        negative=20,
        sample=1e-3,
        ns_exponent=-0.5,
        seed=616,
        epochs=MAX_EPOCHS,
        workers=workers,
        compute_loss=compute_loss,
        callbacks=callbacks
    )
    model.wv.save(OUTPUT_VECTOR)
    model.wv.save_word2vec_format(OUTPUT_BINARY, binary=True)
    print(f"DONE! Vocab length: {len(model.wv)}")


if __name__ == "__main__":
    main()
