import base64
import collections
import datetime as dt
import functools
import json
import logging
import pathlib
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple, Union

import dateutil.parser as dp
import gymnasium as gym
import h5py
import numpy as np
import pandas as pd
import polars as pl
import preprocessor as pp
import sentence_transformers as st
import torch
from rich.pretty import pprint
from torchmetrics.functional.text import bert, rouge
from torchmetrics.text import bert, rouge

BASE64_ALTCHARS = b"+-"


@dataclass
class Text:
    text: str
    embedding: torch.Tensor
    score: float


@dataclass
class Event:
    id: str
    date: str
    request_id: str


class CrisisEnv(gym.Env):
    """
    CrisisEnv is a reinforcement learning environment for the CrisisFACTS dataset.

    Attributes
    ----------
    facts_file : str
        Path to the ground truth file with the facts. Mainly used to get the summary requests.
    text_limit : int
        Maximum number of texts in the summary.
    fact_id : str
        ID of the fact to be summarized.
    confidence_threshold : float
        Confidence threshold for the answer of the model.
    data_folder : str
        Path to the folder with the IPC files with answers.
    render_mode : str
        Render mode for the environment.
    test_mode : bool
        Whether to run the environment in test mode.

    Observation Space
    -----------------
    Box(768, text_limit + 1)
        The observation space is a matrix with the embeddings of the texts in the summary and the
        embedding of the selected text.

    Action Space
    ------------
    Discrete(2)
        The action space is a discrete space with two actions: 0 for keeping the text and 1 for discarding it.

    Reward
    ------
    The reward is:
    - :math:`n - max_similarity` if the action is 0, where n is the number of queries answered by the text
    - :math:`- n + max_similarity` if the action is 1, where n is the number of queries answered by the text
    - :math:`- 5` if the action is 0 and the text is useless
    - :math:`1` if the action is 1 and the text is useless
    """

    metadata = {"render.modes": []}

    def __init__(
        self,
        requests_file: str,
        text_limit: int,
        facts_id: List[str],
        confidence_threshold: float = 0.8,
        data_folder: str = "data/crisisfacts",
        model_name: Literal["longformer", "ensemble", "electra"] = "ensemble",
        render_mode=None,
        mode: Literal["train", "eval", "predict"] = "train",
    ) -> None:
        super().__init__()

        if mode not in ["train", "eval", "predict"]:
            raise ValueError(
                f"Invalid mode {mode}. Valid modes are train, eval and predict"
            )

        # Data folders
        self.texts_folder = pathlib.Path(f"{data_folder}/texts/{model_name}")
        self.embeddings_folder = pathlib.Path(f"{data_folder}/embeddings")
        self._available_datasets = {
            p.stem for p in self.texts_folder.parent.rglob("*.ipc")
        } | {"ensemble"}

        # Gymnasium attributes
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            shape=(text_limit + 1, 768), low=-1, high=1
        )

        self.events = deque(
            pl.read_ipc(requests_file)
            .filter(pl.col("eventID").is_in([f"CrisisFACTS-{id}" for id in facts_id]))
            .filter(
                pl.concat_str(
                    [
                        pl.lit("crisisfacts"),
                        pl.col("eventID").str.split("-").list.last(),
                        "dateString",
                    ],
                    separator="_",
                ).is_in(self._available_datasets)
            )
            .sort("eventID", "dateString")
            .to_struct("event")
            .apply(
                lambda x: Event(
                    x["eventID"].split("-")[1], x["dateString"], x["requestID"]
                ),
                return_dtype=pl.Object(),
            )
        )  # type: deque[Event]

        # Observation and State
        self.summary = deque(maxlen=text_limit)  # type: deque[Text]
        self.observation = np.zeros((text_limit + 1, 768), dtype=np.float32)

        self.current_event = self.events[0]
        self.current_texts = iter([])
        self.selected_text = Text("", torch.empty(1), 0.0)

        # Store event metadata
        self.facts_id = facts_id
        self.data_folder = data_folder
        self.confidence_threshold = confidence_threshold

        self.render_mode = render_mode
        self.mode = mode

        # Dataset management
        self._used_datasets = set()

        self._number_of_queries = 0
        self._seed = None

    def seed(self, seed=None) -> None:
        self._seed = seed

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        logging.info(f"Resetting environment with seed {seed}")
        # Reset state
        self.current_event = self.events[0]
        self.events.rotate(-1)
        self.summary.clear()
        self.observation.fill(0)
        # Get new texts
        id_date = self.current_event.id + "_" + self.current_event.date
        self.current_texts = self._load_texts_by_day(id_date)
        return self._get_state()[0], {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if action == 0:
            reward = self._compute_reward(False)
            self.summary.append(self.selected_text)
        else:
            reward = self._compute_reward(True)

        # Verify if summary is full
        terminated = len(self.summary) == self.summary.maxlen
        # Get current state and check if it's valid
        state, ok = self._get_state()
        terminated = terminated or not ok

        return (
            state,
            float(reward),
            terminated,
            False,
            {
                "text": [s.text for s in self.summary],
                "score": [s.score for s in self.summary],
                "event": self.current_event,
            },
        )

    def render(self, mode: str = "ansi") -> Union[str, Dict[str, Any]]:
        raise NotImplementedError("Rendering is not supported")

    def close(self) -> None:
        pass

    def _load_texts_by_day(self, id_date: str) -> Iterator[Dict[str, Any]]:
        dataset_id = f"crisisfacts_{id_date}"
        logging.info(f"Loading dataset {dataset_id}")
        if dataset_id not in self._available_datasets:
            return iter([])
        # Datasets
        if "ensemble" in self.texts_folder.name:
            originals = [
                pl.scan_ipc(ipc, memory_map=False)
                for ipc in self.texts_folder.parent.rglob(f"*{dataset_id}*")
            ]
            original = originals[0]
            filtered = ensemble_scores(originals, self.confidence_threshold)
        else:
            original = pl.scan_ipc(
                self.texts_folder / f"{dataset_id}.ipc", memory_map=False
            )
            filtered = filter_dataset(original, self.confidence_threshold).select(
                "text", "source_type", "score", "unix_timestamp"
            )
        dataset = self._sample_dataset(filtered)

        self._number_of_queries = original.select("query").unique().collect().height

        scores = (
            dataset.select(pl.col("score") / self._number_of_queries * 100)
            .to_series()
            .sort(descending=True)
        )
        max_reward = scores.head(self.summary.maxlen).sum()
        max_reward += scores.filter(scores == 0).len()
        max_reward -= scores.tail(scores.len() - self.summary.maxlen).sum()
        logging.info(
            f"Taking {dataset.height} samples out of {filtered.height}. Maximum reward is {max_reward}"
        )

        return dataset.iter_rows(named=True)  # .sort("unix_timestamp")

    def _sample_dataset(self, dataset: pl.DataFrame) -> pl.DataFrame:
        if self.mode in ("eval", "predict"):
            return dataset
        return dataset.sample(fraction=1.0, shuffle=True)

    def _get_text(self) -> Tuple[str, float, bool]:
        text = next(self.current_texts, None)
        if text is None:
            return "", 0.0, False
        return (text["text"], text["score"], True)

    def _get_state(self) -> Tuple[np.ndarray, bool]:
        text, score, ok = self._get_text()
        if not ok:
            return self.observation, False

        base64encoder = lambda x: base64.b64encode(
            x.encode(), altchars=BASE64_ALTCHARS
        ).decode()

        encoding = np.empty((768,), dtype=np.float32)
        with h5py.File(
            self.embeddings_folder / f"embeddings_{self.current_event.id}.h5", "r"
        ) as f:
            f[base64encoder(text)].read_direct(encoding)
        self.selected_text = Text(text, torch.from_numpy(encoding).clone(), score)

        for i, summ in enumerate(self.summary):
            self.observation[i] = summ.embedding
        self.observation[-1] = self.selected_text.embedding

        return self.observation, True

    def _compute_reward(self, discarding: bool) -> float:
        score = self.selected_text.score
        reward = score / self._number_of_queries * 100
        # Reverse reward if discarding
        reward = -reward if discarding else reward
        if score != 0:
            max_sim = self._compute_max_similarity()
            similarity_reward = reward * max_sim
            # If discarding and the similarity is high, we want to increase the reward
            similarity_reward = similarity_reward if discarding else -similarity_reward
            reward += similarity_reward
        elif score == 0:
            reward = 1 if discarding else -5
        return reward

    def _compute_max_similarity(self) -> float:
        similarities = [
            torch.nn.functional.cosine_similarity(
                self.selected_text.embedding, s.embedding, dim=0
            )
            for s in self.summary
        ]
        return torch.tensor(similarities).max().item() if similarities else 0


class SimilarityCrisisEnv(CrisisEnv):
    """
    SimilarityCrisisEnv is a simplified version of CrisisEnv

    Observation space
    -----------------
    The observation space is a vector of size 768 + 1 + 1, where:
    - 768 is the size of the BERT embedding
    - 1 is the maximum similarity between the selected text and the summary
    - 1 is the remaining space in the summary
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(768 + 1 + 1,), dtype=np.float32
        )

    def _get_state(self) -> Tuple[np.ndarray, bool]:
        observation, ok = super()._get_state()
        max_similarity = np.array(self._compute_max_similarity())
        current_text = self.observation[-1]
        remaining_space = np.array(1 - len(self.summary) / self.summary.maxlen)

        max_similarity = np.expand_dims(max_similarity, axis=0)
        remaining_space = np.expand_dims(remaining_space, axis=0)
        # Concatenate the current text with the max similarity
        observation = np.concatenate(
            [current_text, max_similarity, remaining_space], dtype=np.float32
        )
        return observation, ok


def ensemble_scores(datasets: List[pl.LazyFrame], threshold: float):
    # Check if the answer is above the confidence threshold
    threshold_predicate = pl.col("score") > threshold
    # Check if the answer does not contain only punctuation
    valid_answer_predicate = (
        pl.col("answer").str.replace(r"[^\w\s]", "").str.lengths() > 0
    )
    # Count the number of query the text answers
    counter = (
        pl.col("answer")
        .filter(threshold_predicate & valid_answer_predicate)
        .unique_counts()
        .len()
    )

    filtered = [
        original.filter(pl.col("source_type") != "Facebook")
        .filter(pl.col("text").str.lengths() > 0)
        .filter(valid_answer_predicate & threshold_predicate)
        .groupby("text", "unix_timestamp", "source_type")
        .agg(counter.alias("score"))
        .collect()
        for original in datasets
    ]
    merged = functools.reduce(
        lambda x, y: x.join(
            y, on=["text", "unix_timestamp", "source_type"], how="inner"
        ).select(
            pl.exclude("score", "score_right"),
            pl.concat_list("score", "score_right").list.mean().floor().alias("score"),
        ),
        filtered,
    )
    all_samples = (
        datasets[0]
        .filter(pl.col("source_type") != "Facebook")
        .filter(pl.col("text").str.lengths() > 0)
        .unique(["text", "source_type", "unix_timestamp"])
        .collect()
    )
    all_samples = (
        merged.join(
            all_samples, on=["text", "source_type", "unix_timestamp"], how="outer"
        )
        .select("text", "source_type", "unix_timestamp", "score")
        .fill_null(0)
    )
    return all_samples


def filter_dataset(original: pl.LazyFrame, confidence_threshold: float):
    # Check if the answer is above the confidence threshold
    threshold_predicate = pl.col("score") > confidence_threshold
    # Check if the answer does not contain only punctuation
    valid_answer_predicate = (
        pl.col("answer").str.replace(r"[^\w\s]", "").str.lengths() > 0
    )
    # Count the number of query the text answers
    counter = (
        pl.col("score").filter(threshold_predicate & valid_answer_predicate).count()
    )
    return (
        original.filter(pl.col("source_type") != "Facebook")
        .filter(pl.col("text").str.lengths() > 0)
        .groupby("text", "unix_timestamp", "source_type")
        .agg(counter.alias("score"))
        .collect()
    )
