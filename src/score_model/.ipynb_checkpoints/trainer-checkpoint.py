import os
import torch
import torch.nn as nn
import numpy as np
import logging
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional, Tuple
from typing_extensions import TypedDict
from tqdm import tqdm
from difflib import SequenceMatcher
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import radon.complexity as radon_complexity
from sympy import simplify, SympifyError
from sympy.parsing.sympy_parser import parse_expr
import threading
from torch.utils.data import DataLoader
import itertools
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Union
import re
from src.score_model.human_eval.test import test
from src.score_model.code_tester import safe_execute_code
from src.score_model.code_tester import extract_all_solutions
from .config import Config
from .model import AdvancedModel
import gc
from src.score_model.config import Config

logger = logging.getLogger(__name__)

class RewardsDict(TypedDict):
    """
    TypedDict for rewards and related metrics.
    """
    rewards: torch.Tensor
    bleu: List[float]
    rouge: List[Dict[str, float]]
    cyclomatic: List[float]

class SCoReTrainer:
    """
    Trainer class for the SCoRe system.
    """

    def __init__(
        self,
        model: AdvancedModel,
        ref_model: AdvancedModel,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Config
    ):
        self.model = model
        self.ref_model = ref_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.kl_loss_fn = nn.KLDivLoss(reduction='batchmean')
        self.global_step = 0
        self.reward_history: List[float] = []
        self.edit_distance_ratios: List[float] = []
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.mixed_precision)
        if config.task == 'MATH':
            self.rouge = Rouge()
            self.smoothing = SmoothingFunction()

    def compute_kl_divergence(self, logits: torch.Tensor, ref_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between model logits and reference logits.
        """
        try:
            log_probs = nn.functional.log_softmax(logits, dim=-1)
            ref_probs = nn.functional.softmax(ref_logits, dim=-1)
            kl_div = self.kl_loss_fn(log_probs, ref_probs)
            return kl_div
        except Exception as e:
            logger.error(f"Error computing KL divergence: {e}")
            raise RuntimeError("KL divergence computation failed.") from e

    def prepare_batch(
            self,
            batch: Union[List[Dict[str, str]], Dict[str, List[str]]]
        ) -> Tuple[List[str], List[str], Optional[List[str]]]:
            """
            Prepare a batch of data for processing.
            Handles both list-of-dicts and dict-of-lists formats.
            """
            try:
                if isinstance(batch, dict):
                    # Handle dict-of-lists format (from DataLoader)
                    if self.config.task == 'CODE':
                        raw_inputs = batch.get('text', batch.get('prompt', []))
                        correct = batch.get('code', batch.get('canonical_solution', []))
                        tests = batch.get('test_list', batch.get('test', []))

                        # Format the prompts for each item in the batch
                        inputs = []
                        for i in range(len(raw_inputs)):
                            prompt = "You are an expert Python programmer, and here is your task:\n\n"
                            prompt += raw_inputs[i] + "\n\n"
                            prompt += "Your code should pass these tests (include imports from libraries if needed):\n"

                            # Handle the test cases - they might be a list for each item
                            if tests and i < len(tests):
                                if isinstance(tests[i], list):
                                    # Join test cases with newlines if they're a list
                                    prompt += "\n".join(tests[i])
                                else:
                                    # Otherwise just add the single test
                                    prompt += str(tests[i])

                            prompt += "\n\nPlease provide your solution:"
                            inputs.append(prompt)
                    else:
                        raise ValueError(f"Invalid task specified: {self.config.task}")
                else:
                    if self.config.task == 'CODE':
                        raw_inputs = [item.get('text', item.get('prompt', '')) for item in batch]
                        correct = [item.get('code', item.get('canonical_solution', '')) for item in batch]
                        tests = [item.get('test_list', item.get('test', '')) for item in batch]

                        # Format the prompts for each item
                        inputs = []
                        for i in range(len(raw_inputs)):
                            prompt = "You are an expert Python programmer, and here is your task:\n\n"
                            prompt += raw_inputs[i] + "\n\n"
                            prompt += "Your code should pass these tests:\n"
                            # Handle the test cases
                            if tests and i < len(tests):
                                if isinstance(tests[i], list):
                                    # Join test cases with newlines if they're a list
                                    prompt += "\n".join(tests[i])
                                else:
                                    # Otherwise just add the single test
                                    prompt += str(tests[i])

                            prompt += "\n\nPlease provide your solution:"
                            inputs.append(prompt)
                    else:
                        raise ValueError(f"Invalid task specified: {self.config.task}")

                logger.debug(f"Batch prepared with {len(inputs)} samples.")
                return inputs, correct, tests
            except KeyError as e:
                logger.error(f"Missing key in batch data: {e}")
                raise KeyError(f"Missing key in batch data: {e}") from e
            except Exception as e:
                logger.error(f"Error preparing batch: {e}")
                raise RuntimeError(f"Failed to prepare batch: {e}") from e

    def train(self) -> None:
        """
        Train the model through both training stages.
        """
        try:
            logger.info("Starting training process.")
            for epoch in range(self.config.num_epochs_stage_one):
                logger.info(f"Starting Stage I Training - Epoch {epoch + 1}")
                self.stage_one()
                torch.cuda.empty_cache()
            for epoch in range(self.config.num_epochs_stage_two):
                logger.info(f"Starting Stage II Training - Epoch {epoch + 1}")
                self.stage_two()
                torch.cuda.empty_cache()
            logger.info("Training completed successfully.")
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise

    from tqdm import tqdm

    def stage_one(self) -> None:
        """
        Stage I training: Train the model with initial rewards, tracking both first and second turn outputs.
        """
        self.model.train()
        total_loss, total_reward_t1, total_reward_t2 = 0.0, 0.0, 0.0

        # Create a progress bar with additional metrics
        progress_bar = tqdm(
            self.train_loader, 
            desc="Stage I Training",
            bar_format="{l_bar}{bar:30}{r_bar}",
            postfix={"loss": 0.0, "reward_t1": 0.0, "reward_t2": 0.0, "kl": 0.0}
        )

        for batch_idx, batch in enumerate(progress_bar):
            torch.cuda.empty_cache()
            gc.collect()

            self.global_step += 1
            try:
                # Prepare batch and initial encoding
                inputs, correct, tests = self.prepare_batch(batch)
                encodings = self.model.tokenizer(
                    inputs,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_seq_len1
                ).to(self.config.device)
            except Exception as e:
                logger.error(f"Error during batch encoding: {e}")
                continue

            try:
                with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                    # Forward pass through model
                    logits = self.model(encodings['input_ids'], encodings['attention_mask'])

                    # Get reference logits for KL divergence
                    with torch.no_grad():
                        ref_logits = self.ref_model(encodings['input_ids'], encodings['attention_mask'])

                    # Compute KL divergence loss
                    kl_loss = self.compute_kl_divergence(logits, ref_logits)

                    # First-turn generation
                    generated_ids = self.model.generate_text(encodings, max_length=self.config.max_seq_len1)
                    generated = self.model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

                    # Extract first-turn code solutions
                    first_extract_codes = []
                    if isinstance(generated, list):
                        for gen in generated:
                            try:
                                code = extract_all_solutions(gen)
                                first_extract_codes.append(code)
                            except Exception as e:
                                print(f"Extraction error for item in first turn: {e}")

                    # Calculate rewards for first turn (for tracking purposes)
                    with torch.no_grad():
                        first_turn_rewards = safe_execute_code(first_extract_codes, tests)['rewards']

            except Exception as e:
                logger.error(f"Error during first turn processing: {e}")
                continue
            torch.cuda.empty_cache()
            gc.collect()
            try:
                # Prepare prompts for second turn
                prompt2 = "There might be an error in the code below because of lack of understanding of the question. Please correct the error, if any, and rewrite the solution. Only output the final correct Python program! \n"
                inputs_two = []

                for i in range(len(first_extract_codes)):
                    if first_extract_codes[i]:  # Check if the list is not empty
                        try:
                            code = first_extract_codes[i][0][1]
                            inputs_two.append(prompt2 + code + "\n" + "Corrected Solution: ")
                        except (IndexError, TypeError) as e:
                            print(f"Skipping malformed extraction at index {i}: {e}")

                # Skip if no valid second-turn inputs
                if not inputs_two:
                    logger.warning("No valid inputs for second turn, skipping batch")
                    continue

                # Process second turn
                second_encodings = self.model.tokenizer(
                    inputs_two,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_seq_len2
                ).to(self.config.device)

                # Generate second turn outputs
                second_generated_ids = self.model.generate_text(second_encodings, max_length=self.config.max_seq_len2)
                generated_code2 = self.model.tokenizer.batch_decode(second_generated_ids, skip_special_tokens=True)

                # Extract code from second turn outputs
                second_extract_codes = []
                if isinstance(generated_code2, list):
                    for gen in generated_code2:
                        try:
                            code = extract_all_solutions(gen)
                            second_extract_codes.append(code)
                        except Exception as e:
                            print(f"Extraction error for item in second turn: {e}")

                second_turn_rewards = safe_execute_code(second_extract_codes, tests)['rewards']

            except Exception as e:
                logger.error(f"Error during second turn or reward computation: {e}")
                continue

            try:
                # Use second turn rewards for loss computation
                loss = -second_turn_rewards + self.config.beta_2 * kl_loss

                # Update progress bar with current metrics
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}", 
                    "reward_t1": f"{first_turn_rewards:.4f}",
                    "reward_t2": f"{second_turn_rewards:.4f}",
                    "kl": f"{kl_loss.item():.4f}"
                })

            except Exception as e:
                logger.error(f"Error during loss computation: {e}")
                continue

            try:
                # Optimization step
                self.optimizer.zero_grad()
                if self.config.mixed_precision:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.model.parameters(), self.config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()

                self.scheduler.step()
            except Exception as e:
                logger.error(f"Error during backward pass or optimization step: {e}")
                continue

            # Update tracking variables
            total_loss += loss.item()
            total_reward_t1 += first_turn_rewards
            total_reward_t2 += second_turn_rewards

            # Store for later analysis
            self.reward_history.append({
                'turn1': first_turn_rewards,
                'turn2': second_turn_rewards
            })

            # Periodic logging
            if self.global_step % self.config.logging_steps == 0:
                logger.info(
                    f"Step {self.global_step}, Loss: {loss.item():.4f}, "
                    f"Reward T1: {first_turn_rewards:.4f}, "
                    f"Reward T2: {second_turn_rewards:.4f}, "
                    f"KL: {kl_loss.item():.4f}"
                )

        # Calculate and log final metrics
        avg_loss = total_loss / len(self.train_loader)
        avg_reward_t1 = total_reward_t1 / len(self.train_loader)
        avg_reward_t2 = total_reward_t2 / len(self.train_loader)

        logger.info(f"Stage I Average Loss: {avg_loss:.4f}")
        logger.info(f"Stage I Average Reward T1: {avg_reward_t1:.4f}")
        logger.info(f"Stage I Average Reward T2: {avg_reward_t2:.4f}")
        logger.info(f"Stage I Reward Improvement: {avg_reward_t2 - avg_reward_t1:.4f}")

    def stage_two(self) -> None:
        """
        Stage II training: Train the model with rewards from both turns and improvement bonuses.
        """
        self.model.train()
        total_loss, total_reward_t1, total_reward_t2, total_bonuses, total_combined = 0.0, 0.0, 0.0, 0.0, 0.0

        # Create a progress bar with expanded metrics
        progress_bar = tqdm(
            self.train_loader, 
            desc="Stage II Training",
            bar_format="{l_bar}{bar:30}{r_bar}",
            postfix={"loss": 0.0, "r1": 0.0, "r2": 0.0, "bonus": 0.0}
        )

        for batch_idx, batch in enumerate(progress_bar):
            torch.cuda.empty_cache()
            gc.collect()
            self.global_step += 1

            try:
                # Prepare batch and initial encoding
                inputs, correct, tests = self.prepare_batch(batch)
                encodings = self.model.tokenizer(
                    inputs,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_seq_len1
                ).to(self.config.device)
            except Exception as e:
                logger.error(f"Error during batch encoding: {e}")
                continue

            try:
                with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                    # Forward pass
                    logits = self.model(encodings['input_ids'], encodings['attention_mask'])

                    # Get reference logits for KL divergence
                    with torch.no_grad():
                        ref_logits = self.ref_model(encodings['input_ids'], encodings['attention_mask'])

                    # Compute KL divergence loss
                    kl_loss = self.compute_kl_divergence(logits, ref_logits)

                    # First-turn generation
                    generated_ids = self.model.generate_text(encodings, max_length=self.config.max_seq_len1)
                    generated = self.model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

                    # Extract first-turn code solutions
                    first_extract_codes = []
                    if isinstance(generated, list):
                        for gen in generated:
                            try:
                                code = extract_all_solutions(gen)
                                first_extract_codes.append(code)
                            except Exception as e:
                                print(f"Extraction error for first-turn item: {e}")

                    # Calculate rewards for first turn
                    rewards_1 = safe_execute_code(first_extract_codes, tests)['rewards']

            except Exception as e:
                logger.error(f"Error during first-turn processing: {e}")
                continue
            torch.cuda.empty_cache()
            gc.collect()
            try:
                # Prepare prompts for second turn
                prompt2 = "There might be an error in the code below because of lack of understanding of the question. Please correct the error, if any, and rewrite the solution. Only output the final correct Python program! \n"
                inputs_two = []

                for i in range(len(first_extract_codes)):
                    if first_extract_codes[i]:  # Check if the list is not empty
                        try:
                            code = first_extract_codes[i][0][1]
                            inputs_two.append(prompt2 + code + "\n" + "Corrected Solution: ")
                        except (IndexError, TypeError) as e:
                            print(f"Skipping malformed extraction at index {i}: {e}")

                # Skip if no valid second-turn inputs
                if not inputs_two:
                    logger.warning("No valid inputs for second turn, skipping batch")
                    continue

                # Process second turn
                second_encodings = self.model.tokenizer(
                    inputs_two,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_seq_len2
                ).to(self.config.device)

                # Generate second turn outputs
                second_generated_ids = self.model.generate_text(second_encodings, max_length=self.config.max_seq_len2)
                generated_code2 = self.model.tokenizer.batch_decode(second_generated_ids, skip_special_tokens=True)

                # Extract code from second turn outputs
                second_extract_codes = []
                if isinstance(generated_code2, list):
                    for gen in generated_code2:
                        try:
                            code = extract_all_solutions(gen)
                            second_extract_codes.append(code)
                        except Exception as e:
                            print(f"Extraction error for second-turn item: {e}")
                torch.cuda.empty_cache()
                gc.collect()
                # Calculate rewards for second turn and bonuses
                rewards_2 = safe_execute_code(second_extract_codes, tests)['rewards']

                # Calculate improvement bonus
                bonuses = self.config.alpha * (rewards_2 - rewards_1)

                # Calculate total rewards (first + second + bonus)
                total_rewards = rewards_1 + rewards_2 + bonuses

            except Exception as e:
                logger.error(f"Error during second-turn processing: {e}")
                continue

            try:
                # Use combined rewards for loss computation
                loss = -total_rewards + self.config.beta_2 * kl_loss

                # Update progress bar with current metrics
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}", 
                    "r1": f"{rewards_1:.4f}",
                    "r2": f"{rewards_2:.4f}",
                    "bonus": f"{bonuses:.4f}"
                })

            except Exception as e:
                logger.error(f"Error during loss computation: {e}")
                continue

            try:
                # Optimization step
                self.optimizer.zero_grad()
                if self.config.mixed_precision:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.model.parameters(), self.config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()

                self.scheduler.step()
            except Exception as e:
                logger.error(f"Error during backward pass or optimization: {e}")
                continue

            # Update tracking variables
            total_loss += loss.item()
            total_reward_t1 += rewards_1
            total_reward_t2 += rewards_2
            total_bonuses += bonuses
            total_combined += total_rewards

            # Store detailed reward history
            self.reward_history.append({
                'turn1': rewards_1,
                'turn2': rewards_2,
                'bonus': bonuses,
                'total': total_rewards
            })

            # Periodic logging
            if self.global_step % self.config.logging_steps == 0:
                logger.info(
                    f"Step {self.global_step}, Loss: {loss.item():.4f}, "
                    f"Total Reward: {total_rewards:.4f}, "
                    f"First Reward: {rewards_1:.4f}, "
                    f"Second Reward: {rewards_2:.4f}, "
                    f"Bonus: {bonuses:.4f}, "
                    f"KL: {kl_loss.item():.4f}"
                )

        # Calculate and log final metrics
        batch_count = len(self.train_loader)
        avg_loss = total_loss / batch_count
        avg_reward_t1 = total_reward_t1 / batch_count
        avg_reward_t2 = total_reward_t2 / batch_count
        avg_bonuses = total_bonuses / batch_count
        avg_combined = total_combined / batch_count

        # Log comprehensive metrics
        logger.info(f"Stage II Average Loss: {avg_loss:.4f}")
        logger.info(f"Stage II Average First-Turn Reward: {avg_reward_t1:.4f}")
        logger.info(f"Stage II Average Second-Turn Reward: {avg_reward_t2:.4f}")
        logger.info(f"Stage II Average Bonus: {avg_bonuses:.4f}")
        logger.info(f"Stage II Average Total Reward: {avg_combined:.4f}")
        logger.info(f"Stage II Reward Improvement: {avg_reward_t2 - avg_reward_t1:.4f}")


    def evaluate(self) -> None:
        """
        Evaluate the model on the validation set.
        """
        self.model.eval()
        total_correct_t1, total_correct_t2, total_samples = 0.0, 0.0, 0
        rewards_first = []
        rewards_second = []

        try:
            with torch.no_grad():
                for batch in tqdm(self.val_loader, desc="Evaluation"):
                    torch.cuda.empty_cache()
                    gc.collect()
                    try:
                        # Prepare batch and initial encoding
                        inputs, correct, tests = self.prepare_batch(batch)
                        encodings = self.model.tokenizer(
                            inputs,
                            return_tensors='pt',
                            padding=True,
                            truncation=True,
                            max_length=self.config.max_seq_len
                        ).to(self.config.device)

                        # First-turn generation
                        generated_ids = self.model.generate_text(encodings, max_length=self.config.max_seq_len1)
                        generated = self.model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

                        # Extract first-turn codes
                        first_extract_codes = []
                        if isinstance(generated, list):
                            for gen in generated:
                                try:
                                    code = extract_all_solutions(gen)
                                    first_extract_codes.append(code)
                                except Exception as e:
                                    print(f"Extraction error for item: {e}")

                        # Calculate first-turn rewards
                        rewards_1 = safe_execute_code(first_extract_codes, tests)
                        torch.cuda.empty_cache()
                        gc.collect()

                        # Prepare second-turn inputs
                        prompt2 = "There might be an error in the code below because of lack of understanding of the question. Please correct the error, if any, and rewrite the solution. Only output the final correct Python program! \n"
                        inputs_two = []
                        for i in range(len(first_extract_codes)):
                            if first_extract_codes[i]:  # Check if the list is not empty
                                try:
                                    code = first_extract_codes[i][0][1]
                                    inputs_two.append(prompt2 + code + "\n" + "Corrected Solution: ")
                                except (IndexError, TypeError) as e:
                                    print(f"Skipping malformed extraction at index {i}: {e}")

                        # Second-turn generation
                        second_encodings = self.model.tokenizer(
                            inputs_two,
                            return_tensors='pt',
                            padding=True,
                            truncation=True,
                            max_length=self.config.max_seq_len2
                        ).to(self.config.device)
                        second_generated_ids = self.model.generate_text(second_encodings, max_length=self.config.max_seq_len2)
                        generated_code2 = self.model.tokenizer.batch_decode(second_generated_ids, skip_special_tokens=True)

                        # Extract second-turn codes
                        second_extract_codes = []
                        if isinstance(generated_code2, list):
                            for gen in generated_code2:
                                try:
                                    code = extract_all_solutions(gen)
                                    second_extract_codes.append(code)
                                except Exception as e:
                                    print(f"Extraction error for item: {e}")

                        # Calculate second-turn rewards
                        rewards_2 = safe_execute_code(second_extract_codes, tests)

                        # Process results for this batch
                        batch_rewards_1 = []
                        batch_rewards_2 = []

                        # Create dictionaries for easier lookup
                        rewards_1_dict = {i["function"]: i for i in rewards_1.get("results", [])}
                        rewards_2_dict = {i["function"]: i for i in rewards_2.get("results", [])}

                        # Collect all unique function names from both result sets
                        all_functions = set(rewards_1_dict.keys()) | set(rewards_2_dict.keys())

                        # Process all functions in a consistent order
                        for func_name in sorted(all_functions):
                            # Process rewards_1
                            if func_name in rewards_1_dict:
                                result = rewards_1_dict[func_name]
                                batch_rewards_1.append(1 if result["passed_tests"] == 3 else 0)
                            else:
                                # Function missing in rewards_1
                                batch_rewards_1.append(0)

                            # Process rewards_2
                            if func_name in rewards_2_dict:
                                result = rewards_2_dict[func_name]
                                batch_rewards_2.append(1 if result["passed_tests"] == 3 else 0)
                            else:
                                # Function missing in rewards_2
                                batch_rewards_2.append(0)

                        # Add batch results to overall results
                        rewards_first.extend(batch_rewards_1)
                        rewards_second.extend(batch_rewards_2)

                    except Exception as e:
                        logger.error(f"Error processing batch: {e}")
                        continue
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")

        # Calculate final metrics
        total_samples = len(rewards_first)
        correct_t1 = sum(rewards_first)
        correct_t2 = sum(rewards_second)
        accuracy_t1 = correct_t1 / total_samples if total_samples > 0 else 0
        accuracy_t2 = correct_t2 / total_samples if total_samples > 0 else 0

        # Calculate delta (improvement from t1 to t2)
        delta = accuracy_t2 - accuracy_t1

        # Count transitions
        incorrect_to_correct = sum(1 for r1, r2 in zip(rewards_first, rewards_second) if r1 == 0 and r2 == 1)
        correct_to_incorrect = sum(1 for r1, r2 in zip(rewards_first, rewards_second) if r1 == 1 and r2 == 0)

        # Calculate fractional metrics
        incorrect_t1 = total_samples - correct_t1  # Number of incorrect solutions in first attempt
        delta_i_to_c_frac = incorrect_to_correct / incorrect_t1 if incorrect_t1 > 0 else 0
        delta_c_to_i_frac = correct_to_incorrect / correct_t1 if correct_t1 > 0 else 0

        # Log the metrics
        logger.info(f"Accuracy@t1: {accuracy_t1:.4f}")
        logger.info(f"Accuracy@t2: {accuracy_t2:.4f}")
        logger.info(f"Δ(t1,t2): {delta:.4f}")
        logger.info(f"Δ_i→c(t1,t2): {delta_i_to_c_frac:.4f}")
        logger.info(f"Δ_c→i(t1,t2): {delta_c_to_i_frac:.4f}")