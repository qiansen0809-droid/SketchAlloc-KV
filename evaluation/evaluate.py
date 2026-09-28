# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0



import json
import logging
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import os

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import load_dataset
from evaluate_registry import DATASET_REGISTRY, PRESS_REGISTRY, SCORER_REGISTRY
from fire import Fire
from tqdm import tqdm
from transformers import Pipeline, pipeline

from kvpress import ComposedPress, DuoAttentionPress, FinchPress, ObservedAttentionPress, ThinKPress

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Dataclass to handle all the configuration for the evaluation."""

    # Core evaluation parameters
    dataset: str = "ruler"
    data_dir: Optional[str] = None
    model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    device: Optional[str] = None
    press_name: str = "knorm"
    compression_ratio: float = 1.0
    key_channel_compression_ratio: Optional[float] = None

    # Support loading a budget curve path from the command line
    budget_curve_path: Optional[str] = None

    # Dataset and generation parameters
    fraction: float = 1.0
    max_new_tokens: Optional[int] = None
    max_context_length: Optional[int] = None
    compress_questions: bool = False

    # Output and logging
    output_dir: str = "./results"
    log_level: str = "INFO"

    # Model-specific parameters
    model_kwargs: Optional[Dict[str, Any]] = None

    # Press information (will be set after press setup)
    press_init_command: Optional[str] = None

    # For reproducibility
    seed: int = 42

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate dataset
        assert self.dataset in DATASET_REGISTRY, f"No dataset found for {self.dataset}"
        assert self.dataset in SCORER_REGISTRY, f"No scorer found for {self.dataset}"

        # Validate press
        assert self.press_name in PRESS_REGISTRY, f"Press '{self.press_name}' not found in PRESS_REGISTRY"
        
        # Dynamically set output_dir based on compression_ratio (e.g. ./0.5_results, ./1.0_results)
        # Only applies when output_dir has not been explicitly set via CLI
        if self.output_dir == "./results":
            self.output_dir = f"./{self.compression_ratio}_results"
        if self.press_name == "no_press":
            # override compression_ratio to 0.0
            logger.info("Using 'no_press' configuration. Overriding compression_ratio to 0.0")
            self.compression_ratio = 0.0

        # Validate compression ratios
        assert (
            0.0 <= self.compression_ratio <= 1.0
        ), f"compression_ratio must be between 0.0 and 1.0, got {self.compression_ratio}"

        # Only validate key_channel_compression_ratio if it's not None
        if self.key_channel_compression_ratio is not None:
            assert (
                0.0 <= self.key_channel_compression_ratio <= 1.0
            ), f"key_channel_compression_ratio must be between 0.0 and 1.0, got {self.key_channel_compression_ratio}"

        # Validate fraction
        assert 0.0 < self.fraction <= 1.0, f"fraction must be between 0.0 and 1.0, got {self.fraction}"

        # Initialize model_kwargs if None
        if self.model_kwargs is None:
            self.model_kwargs = {}

    def get_results_dir(self, output_dir: Path) -> Path:
        """
        Generates the unique save directory and filenames based on configuration parameters.

        Parameters
        ----------
        output_dir : Path
            The output directory path

        Returns
        -------
        Path
            The path to the results directory
        """
        # Build directory name components
        components = [
            self.dataset,
            str(self.data_dir) if self.data_dir else "",
            self.model.replace("/", "--"),
            self.press_name,
            f"{self.compression_ratio:.2f}",
        ]

        # Include the curve name in the output directory path
        if self.budget_curve_path is not None:
            curve_name = os.path.basename(self.budget_curve_path).split('.')[0]
            components.append(f"curve_{curve_name}")
        # ============================================

        if self.fraction < 1.0:
            components.append(f"fraction{self.fraction:.3f}")
        if self.max_context_length is not None:
            components.append(f"max_context{self.max_context_length}")
        if self.compress_questions:
            components.append("compressed_questions")
        if self.key_channel_compression_ratio is not None:
            components.append(f"key_channel_cr{self.key_channel_compression_ratio:.2f}")

        dir_name = "__".join(filter(None, components))  # Filter None/empty strings
        config_dir = output_dir / dir_name

        # This is to avoid overwriting results
        if config_dir.exists():
            i = 1
            while (config_dir / f"{i}").exists():
                i += 1
            config_dir = config_dir / f"{i}"

        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir

    def save_config(self, config_filename: Path):
        """
        Saves the evaluation configuration to a YAML file.
        """
        with open(str(config_filename), "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, indent=2, sort_keys=False)


def _load_yaml_config(path: str | Path) -> dict:
    """Loads a YAML file. Returns an empty dict if it doesn't exist."""
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.warning(f"Config file not found at {path}. Using only command-line arguments and defaults.")
        return {}


class EvaluationRunner:
    """
    EvaluationRunner class that orchestrates the entire evaluation process.

    Parameters
    ----------
    config : EvaluationConfig
        The configuration for the evaluation run.
    """

    def __init__(self, config: EvaluationConfig):
        """
        Initializes the EvaluationRunner with a given configuration.
        """
        self.config = config
        self.pipeline: Optional[Pipeline] = None
        self.press: Any = None
        self.df: Optional[pd.DataFrame] = None
        self._setup_logging()
        self._setup_deterministic_seeds()
        logger.info(f"Initialized EvaluationRunner with config:\n{json.dumps(asdict(self.config), indent=2)}")

    def _setup_deterministic_seeds(self):
        """Set deterministic seeds for reproducible results."""
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        random.seed(self.config.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.info(f"Set deterministic seeds to {self.config.seed}")

    def _setup_logging(self):
        """Configures the logging level based on the config."""
        log_level = self.config.log_level.upper()
        logging.basicConfig(level=getattr(logging, log_level), format="%(asctime)s - %(levelname)s - %(message)s")

    def _setup_directories(self) -> Path:
        """
        Creates the output directory for saving results if it doesn't exist.
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory set to: {output_dir}")
        return output_dir

    def _setup_press(self):
        """
        Initializes the KVPress instance and applies compression ratios based on its type.
        """
        press_name = self.config.press_name
        compression_ratio = self.config.compression_ratio
        key_channel_compression_ratio = self.config.key_channel_compression_ratio

        press = PRESS_REGISTRY[press_name]

        if hasattr(press, "dataset"):  
            press.dataset = self.config.dataset  
        if hasattr(press, "data_dir"):  
            press.data_dir = self.config.data_dir 
        if hasattr(press, "model_name"):
            press.model_name = os.path.basename(self.config.model)

        # Pass the budget curve path from CLI to the press
        if hasattr(press, "budget_curve_path") and self.config.budget_curve_path is not None:
            press.budget_curve_path = self.config.budget_curve_path
        # Trigger post-setup initialization if defined
        if hasattr(press, "_post_setup_init"):  
            press._post_setup_init() 

        if isinstance(press, DuoAttentionPress):
            press.head_compression_ratio = compression_ratio
        elif isinstance(press, ComposedPress):
            for ps in press.presses:
                if isinstance(ps, ThinKPress):
                    assert key_channel_compression_ratio is not None
                    ps.key_channel_compression_ratio = key_channel_compression_ratio
                elif hasattr(ps, "compression_ratio"):
                    ps.compression_ratio = compression_ratio
        elif isinstance(press, ThinKPress):
            assert key_channel_compression_ratio is not None
            press.key_channel_compression_ratio = key_channel_compression_ratio
        elif hasattr(press, "compression_ratio"):
            press.compression_ratio = compression_ratio
        else:
            logger.warning(f"Press {press.__class__.__name__} has no 'compression_ratio' attribute.")

        self.press = press
        self.config.press_init_command = str(press)
        logger.info(f"KV Press '{press_name}' setup.")

    def _load_dataset(self):
        """
        Loads the dataset specified in the config and applies sampling.
        """
        dataset_name = self.config.dataset
        data_dir = str(self.config.data_dir) if self.config.data_dir else None
        fraction = self.config.fraction

        logger.info(f"Loading dataset: {DATASET_REGISTRY[dataset_name]} (data_dir: {data_dir})")
        df = load_dataset(DATASET_REGISTRY[dataset_name], data_dir=data_dir, split="test").to_pandas()

        if fraction < 1.0:
            df = df.sample(frac=fraction, random_state=self.config.seed)
            logger.info(f"Sampled {len(df)} samples ({fraction:.2f}) from original dataset.")

        self.df = df
        logger.info(f"Dataset loaded with {len(self.df)} entries.")

    def _prepare_data_for_inference(self):
        """
        Prepares the loaded dataframe for inference.
        """
        if self.df is None:
            raise ValueError("Dataset not loaded. Call _load_dataset() first.")

        if isinstance(self.press, FinchPress):
            if not self.config.compress_questions:
                raise ValueError("FinchPress requires compress_questions to be set to True")
            logger.info("FinchPress detected, updating model and tokenizer with delimiter token.")
            self.press.update_model_and_tokenizer(self.pipeline.model, self.pipeline.tokenizer)
            self.df["context"] = self.df["context"] + self.press.delimiter_token

        if self.config.compress_questions:
            logger.info("Compressing questions into context.")
            self.df["context"] = self.df["context"] + self.df["question"]
            self.df["question"] = ""

        logger.info(f"Dataset processed with {len(self.df)} entries.")

    def _setup_model_pipeline(self):
        """
        Sets up the model and tokenizer pipeline for text generation.
        """
        model_name = self.config.model
        device = self.config.device
        model_kwargs = self.config.model_kwargs or {}

        if device is None:
            device = "auto"
            logger.info(f"No device specified, using 'auto' device mapping.")

        if isinstance(self.press, (ObservedAttentionPress)):
            model_kwargs["attn_implementation"] = "eager"
            logger.info("ObservedAttentionPress detected, setting attn_implementation to 'eager'.")
        else:
            try:
                import flash_attn
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Flash Attention 2 detected, setting attn_implementation to 'flash_attention_2'.")
            except ImportError:
                logger.info("Flash Attention 2 not available, using default attn_implementation.")

        pipeline_kwargs = {
            "model": model_name,
            "model_kwargs": model_kwargs,
            "trust_remote_code": True,
        }
        if device == "auto":
            pipeline_kwargs["device_map"] = "auto"
        else:
            pipeline_kwargs["device"] = device

        self.pipeline = pipeline("kv-press-text-generation", **pipeline_kwargs)
        self.pipeline.model.eval()
        logger.info("Model pipeline loaded.")

    @torch.inference_mode()
    def _run_inference(self, df_to_process: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the inference process on a given dataframe.
        """
        df_to_process["predicted_answer"] = None
        df_context_grouped = df_to_process.groupby("context")
        assert all(df_context_grouped["answer_prefix"].nunique() == 1)

        for context, df_group in tqdm(df_context_grouped, total=len(df_context_grouped), desc="Running Inference"):
            output = self.pipeline(
                context,
                questions=df_group["question"].to_list(),
                answer_prefix=df_group["answer_prefix"].iloc[0],
                press=self.press,
                max_new_tokens=self.config.max_new_tokens or df_group["max_new_tokens"].iloc[0],
                max_context_length=self.config.max_context_length,
                dataset_name = self.config.dataset,
                data_dir_name = self.config.data_dir,
            )
            df_to_process.loc[df_group.index, "predicted_answer"] = output["answers"]
            if self.press is not None and hasattr(self.press, "compression_ratio"):
                df_to_process.loc[df_group.index, "compression_ratio"] = self.press.compression_ratio
            else:
                df_to_process.loc[df_group.index, "compression_ratio"] = 0.0
            torch.cuda.empty_cache()

        logger.info("Inference completed for the current task.")
        return df_to_process

    def _save_results(self, df_to_save: pd.DataFrame, save_filename: Path):
        """
        Saves prediction results to a CSV file in append mode.
        """
        write_header = not save_filename.exists()
        mode = 'w' if write_header else 'a'
        columns_to_save = [col for col in df_to_save.columns if col != "context"]
        df_to_save[columns_to_save].to_csv(str(save_filename), index=False, mode=mode, header=write_header)
        logger.info(f"Results for the task appended to {save_filename}")

    def _calculate_and_save_metrics(self, df_for_metrics: pd.DataFrame, save_filename: Path):
        """
        Calculates metrics for a task and updates the metrics JSON file.
        """
        scorer = SCORER_REGISTRY[self.config.dataset]
        all_metrics = {}
        if save_filename.exists():
            with open(str(save_filename), "r") as f:
                try:
                    all_metrics = json.load(f)
                except json.JSONDecodeError:
                    pass  # File is empty, start with an empty dict

        logger.info(f"Calculating metrics for dataset: {self.config.dataset}")
        new_metrics = scorer(df_for_metrics)
        all_metrics.update(new_metrics)

        with open(str(save_filename), "w") as f:
            json.dump(all_metrics, f, indent=4)

        logger.info(f"Metrics updated in {save_filename}")
        logger.info(f"Newly calculated metrics for this task:\n{json.dumps(new_metrics, indent=2)}")

        # Print a quick summary of the task result
        print("-" * 50)
        task_name = list(new_metrics.keys())[0] if new_metrics else "Unknown Task"
        task_score = list(new_metrics.values())[0] if new_metrics else "N/A"
        print(f"✅ Task Completed: {task_name} | Score: {task_score}")
        print("-" * 50)

    def run_evaluation(self):
        """
        Orchestrates the entire evaluation process, running task by task.
        """
        logger.info("Starting evaluation run...")
        output_dir = self._setup_directories()
        results_dir = self.config.get_results_dir(output_dir)
        predictions_filename = results_dir / "predictions.csv"
        metrics_filename = results_dir / "metrics.json"
        config_filename = results_dir / "config.yaml"

        if config_filename.exists():
            logger.info(
                f"Config file found at {config_filename}, indicating a prior run. "
                f"To force a re-run, delete the directory: {results_dir}\nSkipping..."
            )
            return

        # Setup is done once
        self._load_dataset()
        self._setup_press()
        self._setup_model_pipeline()
        self._prepare_data_for_inference()
        self.config.save_config(config_filename)

        # Clear any previous partial results before starting the loop
        if predictions_filename.exists(): predictions_filename.unlink()
        if metrics_filename.exists(): metrics_filename.unlink()

        # Loop through each task
        tasks = self.df["task"].unique()
        logger.info(f"Found {len(tasks)} tasks to evaluate: {list(tasks)}")

        for task in tasks:
            logger.info(f"--- Starting evaluation for task: {task} ---")
            task_df = self.df[self.df["task"] == task].copy()
            processed_task_df = self._run_inference(task_df)
            self._save_results(processed_task_df, predictions_filename)
            self._calculate_and_save_metrics(processed_task_df, metrics_filename)
            logger.info(f"--- Finished evaluation for task: {task} ---")

        logger.info("All tasks completed. Evaluation run finished successfully.")


class CliEntryPoint:
    """
    CLI entry point for building configuration and running the evaluation.
    """

    def __call__(self, config_file: Optional[str] = "./evaluate_config.yaml", **cli_overrides):
        """
        Builds the configuration and runs the evaluation.
        """
        # 1. Start with dataclass defaults.
        final_args = asdict(EvaluationConfig())
        # 2. Layer YAML values on top.
        if config_file:
            final_args.update(_load_yaml_config(config_file))
        # 3. Layer CLI arguments on top.
        final_args.update({k: v for k, v in cli_overrides.items() if v is not None})

        try:
            config = EvaluationConfig(**final_args)
        except (TypeError, AssertionError) as e:
            print(f"Error: Invalid configuration argument provided. {e}", file=sys.stderr)
            sys.exit(1)

        runner = EvaluationRunner(config)
        runner.run_evaluation()


if __name__ == "__main__":
    Fire(CliEntryPoint)