"""
EM-DPO Trainer: Using EM Algorithm for Reward Learning, Standard DPO for Training

This trainer uses EM algorithm to learn reward models and clusters,
but then uses standard DPO (not MaxMin-DPO) for training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
)
from torch.optim import AdamW
from typing import Optional, Union, Dict, List, Tuple, Any
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import logging

from trainers.dpo_trainer import DPOTrainer
from trainers.dpo_config import DPOConfig
from trainers.utils import logprobs_from_logits, set_seed, flatten_dict
from dataclasses import dataclass
import typing

logger = logging.getLogger(__name__)

# Define PreTrainedModelWrapper type (same as in dpo_trainer.py)
PreTrainedModelWrapper = typing.Union[nn.Module, nn.DataParallel]


@dataclass
class EMDPOConfig(DPOConfig):
    """Configuration for EM-DPO training."""
    
    num_clusters: int = 2
    """Number of user clusters/subpopulations"""
    
    # EM algorithm parameters
    em_convergence_threshold: float = 1e-4
    """Convergence threshold for EM algorithm"""
    
    em_max_iterations: int = 50
    """Maximum number of EM iterations"""
    
    reward_model_name: Optional[str] = None
    """Base model name for reward models (if None, will use model_name)"""
    
    reward_learning_rate: float = 1e-5
    """Learning rate for reward model training in M-step"""
    
    reward_num_epochs: int = 3
    """Number of epochs for reward model training in M-step"""
    
    reward_batch_size: int = 8
    """Batch size for reward model training"""


class RewardModel(nn.Module):
    """Simple reward model wrapper for sequence classification models."""
    
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        
    def forward(self, input_ids, attention_mask=None):
        """Forward pass that returns scalar rewards."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # For sequence classification with num_labels=1, logits is [batch_size, 1]
        if outputs.logits.dim() == 2 and outputs.logits.size(1) == 1:
            return outputs.logits.squeeze(-1)  # [batch_size]
        return outputs.logits  # Fallback
    
    def compute_reward(self, x: str, y: str) -> float:
        """Compute reward for a (prompt, response) pair."""
        text = f"{x}{y}"  # Simple concatenation, can be customized
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
        # Ensure input_ids is on the correct device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            reward = self.forward(**inputs)
        return reward.item() if reward.numel() == 1 else reward.mean().item()


def compute_reward_probability(reward_model: RewardModel, x: str, y1: str, y2: str, device: str) -> float:
    """
    Compute w(φ_u, x, y1, y2) = exp(r_φ_u(y1,x)) / (exp(r_φ_u(y1,x)) + exp(r_φ_u(y2,x)))
    """
    reward_model.eval()
    with torch.no_grad():
        r_y1 = reward_model.compute_reward(x, y1)
        r_y2 = reward_model.compute_reward(x, y2)
        
        # Use log-sum-exp trick for numerical stability
        exp_r_y1 = np.exp(r_y1)
        exp_r_y2 = np.exp(r_y2)
        w = exp_r_y1 / (exp_r_y1 + exp_r_y2)
    return w


class EMRewardLearner:
    """
    Algorithm 2: Learning Rewards with EM Algorithm
    
    Implements the EM algorithm to learn reward models for different user clusters.
    """
    
    def __init__(
        self,
        num_clusters: int,
        reward_model_name: str,
        tokenizer: PreTrainedTokenizerBase,
        device: str = "cuda",
        convergence_threshold: float = 1e-4,
        max_iterations: int = 50,
        reward_learning_rate: float = 1e-5,
        reward_num_epochs: int = 3,
    ):
        self.num_clusters = num_clusters
        self.reward_model_name = reward_model_name
        self.tokenizer = tokenizer
        self.device = device
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        self.reward_learning_rate = reward_learning_rate
        self.reward_num_epochs = reward_num_epochs
        
        # Initialize reward models for each cluster
        # IMPORTANT: Reward models need their own tokenizer, not the policy tokenizer!
        # Using policy tokenizer with reward model can cause vocab size mismatch
        try:
            reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
            if reward_tokenizer.pad_token is None:
                reward_tokenizer.pad_token = reward_tokenizer.eos_token or "[PAD]"
        except Exception as e:
            logger.warning(f"Failed to load tokenizer for reward model {reward_model_name}: {e}. Using policy tokenizer.")
            reward_tokenizer = tokenizer
        
        self.reward_models = []
        for u in range(num_clusters):
            config = AutoConfig.from_pretrained(reward_model_name)
            config.num_labels = 1
            model = AutoModelForSequenceClassification.from_pretrained(
                reward_model_name, config=config
            )
            model.to(device)
            reward_model = RewardModel(model, reward_tokenizer)  # Use reward_tokenizer, not policy tokenizer
            self.reward_models.append(reward_model)
        
        # Cluster assignments: user_id -> cluster_id
        self.user_assignments = {}
    
    def e_step(self, preference_data: List[Dict]) -> Dict[int, int]:
        """
        E-step: Hard cluster assignment for each user.
        
        For each user h, assign to cluster u that maximizes:
        Π_{(x,y1,y2,h) ∈ D} w(φ_u, x, y1, y2)
        
        Args:
            preference_data: List of dicts with keys: 'user_id', 'prompt', 'chosen', 'rejected'
            
        Returns:
            Dictionary mapping user_id to cluster_id
        """
        # Group data by user
        user_data = {}
        for item in preference_data:
            user_id = item.get('user_id', item.get('annotator', 0))  # Support multiple field names
            if user_id not in user_data:
                user_data[user_id] = []
            user_data[user_id].append(item)
        
        # Assign each user to the best cluster
        assignments = {}
        for user_id, items in tqdm(user_data.items(), desc="E-step: Assigning users to clusters"):
            best_cluster = 0
            best_score = -float('inf')
            
            for u in range(self.num_clusters):
                # Compute product of w(φ_u, x, y1, y2) for all preference pairs
                log_likelihood = 0.0
                for item in items:
                    x = item['prompt']
                    y1 = item['chosen']
                    y2 = item['rejected']
                    
                    try:
                        w = compute_reward_probability(self.reward_models[u], x, y1, y2, self.device)
                        # Use log to avoid numerical underflow
                        log_likelihood += np.log(w + 1e-10)
                    except Exception as e:
                        logger.warning(f"Error computing reward probability: {e}")
                        continue
                
                if log_likelihood > best_score:
                    best_score = log_likelihood
                    best_cluster = u
            
            assignments[user_id] = best_cluster
        
        self.user_assignments = assignments
        return assignments
    
    def m_step(self, preference_data: List[Dict], assignments: Dict[int, int]):
        """
        M-step: Update each reward model φ_u by minimizing negative log-likelihood
        on the assigned users' data.
        """
        # Group data by cluster
        cluster_data = {u: [] for u in range(self.num_clusters)}
        for item in preference_data:
            user_id = item.get('user_id', item.get('annotator', 0))
            if user_id in assignments:
                cluster_id = assignments[user_id]
                cluster_data[cluster_id].append(item)
        
        # Train each reward model on its assigned data
        for u in range(self.num_clusters):
            if len(cluster_data[u]) == 0:
                logger.warning(f"Cluster {u} has no assigned data, skipping update")
                continue
            
            logger.info(f"M-step: Training reward model for cluster {u} on {len(cluster_data[u])} samples")
            self._train_reward_model(self.reward_models[u], cluster_data[u])
    
    def _train_reward_model(self, reward_model: RewardModel, data: List[Dict], batch_size: int = 8):
        """Train a single reward model on preference data using ranking loss."""
        reward_model.train()
        optimizer = AdamW(reward_model.model.parameters(), lr=self.reward_learning_rate)
        
        # Simple training loop with batching
        for epoch in range(self.reward_num_epochs):
            total_loss = 0.0
            num_batches = 0
            
            # Process in batches
            for batch_start in range(0, len(data), batch_size):
                batch_end = min(batch_start + batch_size, len(data))
                batch_data = data[batch_start:batch_end]
                
                # Collect texts
                chosen_texts = []
                rejected_texts = []
                for item in batch_data:
                    x = item['prompt']
                    y_chosen = item['chosen']
                    y_rejected = item['rejected']
                    chosen_texts.append(f"{x}{y_chosen}")
                    rejected_texts.append(f"{x}{y_rejected}")
                
                # Tokenize batch with proper padding
                # Use max_length padding to ensure consistent tensor sizes
                chosen_inputs = reward_model.tokenizer(
                    chosen_texts, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=512, 
                    padding="max_length",
                    return_attention_mask=True
                )
                rejected_inputs = reward_model.tokenizer(
                    rejected_texts, 
                    return_tensors="pt", 
                    truncation=True,
                    max_length=512, 
                    padding="max_length",
                    return_attention_mask=True
                )
                
                # Clamp input_ids to valid vocab range to avoid index errors
                vocab_size = reward_model.model.config.vocab_size
                chosen_inputs['input_ids'] = torch.clamp(chosen_inputs['input_ids'], 0, vocab_size - 1)
                rejected_inputs['input_ids'] = torch.clamp(rejected_inputs['input_ids'], 0, vocab_size - 1)
                
                # Move to device and ensure proper dtype
                device = next(reward_model.model.parameters()).device
                chosen_inputs = {k: v.to(device) for k, v in chosen_inputs.items()}
                rejected_inputs = {k: v.to(device) for k, v in rejected_inputs.items()}
                
                # Forward pass
                try:
                    reward_chosen = reward_model(**chosen_inputs)
                    reward_rejected = reward_model(**rejected_inputs)
                except RuntimeError as e:
                    logger.error(f"Error in forward pass: {e}")
                    logger.error(f"Chosen input_ids shape: {chosen_inputs['input_ids'].shape}")
                    logger.error(f"Rejected input_ids shape: {rejected_inputs['input_ids'].shape}")
                    logger.error(f"Chosen input_ids max: {chosen_inputs['input_ids'].max()}, vocab size: {reward_model.model.config.vocab_size}")
                    raise
                
                # Ranking loss: -log sigmoid(reward_chosen - reward_rejected)
                loss = -F.logsigmoid(reward_chosen - reward_rejected).mean()
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            logger.info(f"  Epoch {epoch+1}/{self.reward_num_epochs}, Loss: {avg_loss:.4f}")
        
        reward_model.eval()
    
    def _initialize_assignments(self, preference_data: List[Dict]) -> Dict[int, int]:
        """
        Initialize cluster assignments randomly to ensure all clusters get some data.
        This prevents all users from being assigned to the same cluster in the first iteration.
        """
        # Group data by user
        user_data = {}
        for item in preference_data:
            user_id = item.get('user_id', item.get('annotator', 0))
            if user_id not in user_data:
                user_data[user_id] = []
            user_data[user_id].append(item)
        
        # Randomly assign users to clusters, ensuring roughly equal distribution
        user_ids = list(user_data.keys())
        np.random.shuffle(user_ids)  # Shuffle for randomness
        
        assignments = {}
        for idx, user_id in enumerate(user_ids):
            # Assign to cluster based on modulo to ensure distribution
            cluster_id = idx % self.num_clusters
            assignments[user_id] = cluster_id
        
        logger.info(f"Initialized {len(assignments)} users to {self.num_clusters} clusters")
        for cluster_id in range(self.num_clusters):
            count = sum(1 for uid, cid in assignments.items() if cid == cluster_id)
            logger.info(f"  Cluster {cluster_id}: {count} users")
        
        return assignments
    
    def fit(self, preference_data: List[Dict]) -> List[RewardModel]:
        """
        Run EM algorithm to learn reward models.
        
        Returns:
            List of trained reward models (one per cluster)
        """
        logger.info(f"Starting EM algorithm with {self.num_clusters} clusters")
        
        # Initialize assignments randomly to ensure all clusters get data
        prev_assignments = self._initialize_assignments(preference_data)
        
        # First M-step: Train reward models on initial random assignments
        logger.info("\n=== Initial M-step: Training reward models on random assignments ===")
        self.m_step(preference_data, prev_assignments)
        
        for iteration in range(self.max_iterations):
            logger.info(f"\n=== EM Iteration {iteration + 1}/{self.max_iterations} ===")
            
            # E-step: Assign users to clusters
            assignments = self.e_step(preference_data)
            
            # Check convergence
            if prev_assignments is not None:
                # Count how many assignments changed
                num_changes = sum(
                    1 for user_id in assignments 
                    if user_id in prev_assignments and assignments[user_id] != prev_assignments[user_id]
                )
                total_users = len(assignments)
                change_ratio = num_changes / total_users if total_users > 0 else 0.0
                
                logger.info(f"Assignment change ratio: {change_ratio:.4f}")
                
                if change_ratio < self.convergence_threshold:
                    logger.info("Converged!")
                    break
            
            prev_assignments = assignments.copy()
            
            # M-step: Update reward models
            self.m_step(preference_data, assignments)
        
        logger.info("EM algorithm completed")
        return self.reward_models


class EMDPOTrainer(DPOTrainer):
    """
    EM-DPO Trainer
    
    Uses EM algorithm to learn reward models and clusters, but then
    uses standard DPO (not MaxMin-DPO) for training.
    """
    
    def __init__(
        self,
        config: DPOConfig = None,
        model: PreTrainedModelWrapper = None,
        ref_model: Optional[PreTrainedModelWrapper] = None,
        tokenizer: PreTrainedTokenizerBase = None,
        dataset: Optional[Union[torch.utils.data.Dataset, Dataset]] = None,
        reward_models: Optional[List[RewardModel]] = None,
        user_assignments: Optional[Dict[int, int]] = None,
        num_clusters: int = 2,
        **kwargs
    ):
        super().__init__(
            config=config,
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            dataset=dataset,
            **kwargs
        )
        
        self.reward_models = reward_models or []
        self.user_assignments = user_assignments or {}
        self.num_clusters = num_clusters
        
        # Log cluster information
        if user_assignments:
            for cluster_id in range(num_clusters):
                count = sum(1 for uid, cid in user_assignments.items() if cid == cluster_id)
                logger.info(f"Cluster {cluster_id}: {count} users")
    
    def _compute_cluster_weights(
        self,
        queries: torch.LongTensor,
        responses_w: torch.LongTensor,
        responses_l: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Compute soft weights p(u|sample) for each cluster u and each sample.
        
        Scheme 2: Mix when uncertain - compute weights w_u for each cluster,
        then normalize to get p(u|sample).
        
        Returns:
            cluster_weights: [B, num_clusters] tensor of normalized weights p(u|sample)
        """
        batch_size = queries.size(0)
        device = queries.device
        num_clusters = len(self.reward_models) if self.reward_models else 1
        
        # If no reward models, return uniform weights
        if not self.reward_models:
            return torch.ones(batch_size, num_clusters, device=device) / num_clusters
        
        # Compute weights for each cluster and each sample
        # weights[i, u] = w(φ_u, x_i, y1_i, y2_i) = exp(r_φ_u(y1_i, x_i)) / (exp(r_φ_u(y1_i, x_i)) + exp(r_φ_u(y2_i, x_i)))
        weights = torch.zeros(batch_size, num_clusters, device=device, dtype=torch.float32)
        
        for i in range(batch_size):
            # Decode queries and responses
            query_text = self.tokenizer.decode(queries[i], skip_special_tokens=True)
            response_w_text = self.tokenizer.decode(responses_w[i], skip_special_tokens=True)
            response_l_text = self.tokenizer.decode(responses_l[i], skip_special_tokens=True)
            
            for u in range(num_clusters):
                reward_model = self.reward_models[u]
                try:
                    # Compute w(φ_u, x, y1, y2) = exp(r_φ_u(y1,x)) / (exp(r_φ_u(y1,x)) + exp(r_φ_u(y2,x)))
                    r_w = reward_model.compute_reward(query_text, response_w_text)
                    r_l = reward_model.compute_reward(query_text, response_l_text)
                    
                    # Use log-sum-exp trick for numerical stability
                    max_r = max(r_w, r_l)
                    exp_r_w = np.exp(r_w - max_r)
                    exp_r_l = np.exp(r_l - max_r)
                    w_u = exp_r_w / (exp_r_w + exp_r_l + 1e-10)
                    
                    weights[i, u] = w_u
                except Exception as e:
                    logger.warning(f"Error computing weight for sample {i}, cluster {u}: {e}, using uniform weight")
                    weights[i, u] = 1.0 / num_clusters
        
        # Normalize to get p(u|sample): p(u|sample) = w_u / sum_u' w_u'
        # Add small epsilon to avoid division by zero
        weights_sum = weights.sum(dim=1, keepdim=True) + 1e-10
        cluster_weights = weights / weights_sum  # [B, num_clusters]
        
        return cluster_weights
    
    def _compute_mixture_rewards(
        self,
        queries: torch.LongTensor,
        responses_w: torch.LongTensor,
        responses_l: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute mixture rewards using soft cluster weights (Scheme 2).
        
        For each sample, compute rewards from all clusters and mix them:
        r_mixed = sum_u p(u|sample) * r_φ_u
        
        Returns:
            rewards_w: [B] tensor of mixed rewards for chosen responses
            rewards_l: [B] tensor of mixed rewards for rejected responses
            cluster_weights: [B, num_clusters] tensor of cluster weights (for stats)
        """
        batch_size = queries.size(0)
        device = queries.device
        num_clusters = len(self.reward_models) if self.reward_models else 1
        
        # If no reward models, return zeros (fallback to standard DPO)
        if not self.reward_models:
            cluster_weights = torch.ones(batch_size, num_clusters, device=device) / num_clusters
            return torch.zeros(batch_size, device=device), torch.zeros(batch_size, device=device), cluster_weights
        
        # Compute cluster weights p(u|sample) for each sample
        # This also computes rewards internally, so we'll reuse them
        cluster_weights = self._compute_cluster_weights(queries, responses_w, responses_l)  # [B, num_clusters]
        
        # Compute rewards from each cluster
        cluster_rewards_w = torch.zeros(batch_size, num_clusters, device=device, dtype=torch.float32)
        cluster_rewards_l = torch.zeros(batch_size, num_clusters, device=device, dtype=torch.float32)
        
        for i in range(batch_size):
            query_text = self.tokenizer.decode(queries[i], skip_special_tokens=True)
            response_w_text = self.tokenizer.decode(responses_w[i], skip_special_tokens=True)
            response_l_text = self.tokenizer.decode(responses_l[i], skip_special_tokens=True)
            
            for u in range(num_clusters):
                reward_model = self.reward_models[u]
                try:
                    r_w = reward_model.compute_reward(query_text, response_w_text)
                    r_l = reward_model.compute_reward(query_text, response_l_text)
                    cluster_rewards_w[i, u] = r_w
                    cluster_rewards_l[i, u] = r_l
                except Exception as e:
                    logger.warning(f"Error computing reward for sample {i}, cluster {u}: {e}, using zero reward")
                    cluster_rewards_w[i, u] = 0.0
                    cluster_rewards_l[i, u] = 0.0
        
        # Mix rewards: r_mixed = sum_u p(u|sample) * r_φ_u
        rewards_w = (cluster_weights * cluster_rewards_w).sum(dim=1)  # [B]
        rewards_l = (cluster_weights * cluster_rewards_l).sum(dim=1)  # [B]
        
        return rewards_w, rewards_l, cluster_weights
    
    def _step(
        self,
        queries: torch.LongTensor,
        responses_w: torch.LongTensor,
        responses_l: torch.LongTensor,
        preference_mask: Optional[torch.BoolTensor] = None,
        user_ids: Optional[torch.Tensor] = None,
    ):
        """
        Cluster-conditioned DPO step with soft mixture (Scheme 2).
        
        Uses soft cluster weights p(u|sample) to mix rewards from all clusters:
        - Compute weights w_u = w(φ_u, x, y1, y2) for each cluster u
        - Normalize to get p(u|sample) = w_u / sum_u' w_u'
        - Mix rewards: r_mixed = sum_u p(u|sample) * r_φ_u
        - Compute DPO loss using mixed rewards
        
        The DPO loss becomes:
        L = E_{(x,y+,y-)} [ sum_u p(u|sample) * DPO_loss_u ]
        where DPO_loss_u uses reward model φ_u
        
        Advantages:
        - Not susceptible to errors from incorrect cluster selection
        - Results in a smoother process
        - No need to select cluster during test
        """
        input_ids_w = torch.cat((queries, responses_w), dim=1)  # [B, Lq+Lw]
        input_ids_l = torch.cat((queries, responses_l), dim=1)  # [B, Lq+Ll]
        pad_id = self.tokenizer.pad_token_id

        def process_input_ids(input_ids):
            # attention_mask: pad=0, others=1
            attention_mask = (input_ids != pad_id).long()  # [B, L]
            input_data = {"input_ids": input_ids, "attention_mask": attention_mask}

            logits, _, _ = self.model(**input_data)
            with torch.no_grad():
                old_logits, _, _ = self.ref_model(**input_data)
                old_logprobs = logprobs_from_logits(old_logits[:, :-1, :], input_ids[:, 1:])  # [B, L-1]
            logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])  # [B, L-1]
            attn_mask_shifted = attention_mask[:, 1:]  # [B, L-1]

            return logprobs, old_logprobs, attn_mask_shifted

        logprobs_w, old_logprobs_w, attn_shift_w = process_input_ids(input_ids_w)
        logprobs_l, old_logprobs_l, attn_shift_l = process_input_ids(input_ids_l)

        q_len = queries.size(1)
        mask_w = attn_shift_w.clone()  # [B, L-1]
        mask_l = attn_shift_l.clone()
        if q_len > 1:
            mask_w[:, :q_len - 1] = 0
            mask_l[:, :q_len - 1] = 0

        if preference_mask is not None:
            sample_mask = preference_mask.to(logprobs_w.device).float()  # [B]
        else:
            sample_mask = torch.ones(logprobs_w.size(0), device=logprobs_w.device)

        seq_logprob_w = (logprobs_w * mask_w).sum(dim=1)          # [B]
        seq_logprob_l = (logprobs_l * mask_l).sum(dim=1)          # [B]
        seq_old_logprob_w = (old_logprobs_w * mask_w).sum(dim=1)  # [B]
        seq_old_logprob_l = (old_logprobs_l * mask_l).sum(dim=1)  # [B]

        pi_logratios_seq = seq_logprob_w - seq_logprob_l               # [B]
        
        # Cluster-conditioned DPO with soft mixture (Scheme 2):
        # loss = sum_u p(u|sample) * loss_u
        # where loss_u uses reward model φ_u
        cluster_rewards_w = None
        cluster_rewards_l = None
        cluster_weights = None
        if self.reward_models:
            # Compute mixture rewards using soft cluster weights
            cluster_rewards_w, cluster_rewards_l, cluster_weights = self._compute_mixture_rewards(
                queries, responses_w, responses_l
            )
            
            # Use reward difference as reference logratio
            # Scale rewards to match logprob scale (optional, can be tuned)
            reward_scale = getattr(self.config, 'cluster_reward_scale', 1.0)
            ref_logratios_seq = reward_scale * (cluster_rewards_w - cluster_rewards_l)  # [B]
        else:
            # Fallback to standard DPO using reference model
            ref_logratios_seq = seq_old_logprob_w - seq_old_logprob_l      # [B]

        dpo_logit_seq = self.config.temperature * (pi_logratios_seq - ref_logratios_seq)  # [B]

        if self.config.ipo_loss:
            dpo_loss_vec = (dpo_logit_seq - 1.0 / (2 * self.config.temperature)) ** 2  # [B]
        else:
            dpo_loss_vec = -F.logsigmoid(dpo_logit_seq)  # [B]

        denom = sample_mask.sum()
        if denom.item() == 0:
            dpo_loss = dpo_loss_vec.mean()
        else:
            dpo_loss = (dpo_loss_vec * sample_mask).sum() / denom

        delta_w = seq_logprob_w - seq_old_logprob_w  # [B]
        delta_l = seq_logprob_l - seq_old_logprob_l  # [B]

        rewards_chosen = self.config.temperature * delta_w.detach()
        rewards_rejected = self.config.temperature * delta_l.detach()
        reward_margin = rewards_chosen - rewards_rejected

        stats = dict(
            loss=dict(
                dpo_loss=dpo_loss.detach(),
            ),
            policy=dict(
                rewards_chosen=rewards_chosen.mean().detach(),
                rewards_rejected=rewards_rejected.mean().detach(),
                reward_margin=reward_margin.mean().detach(),
                logprobs_w=seq_logprob_w.mean().detach(),
                logprobs_l=seq_logprob_l.mean().detach(),
                dpo_logit_mean=dpo_logit_seq.mean().detach(),
                classifier_accuracy=(reward_margin > 0).float().mean().detach(),
            ),
        )
        
        # Add cluster-specific stats if using cluster-conditioned DPO
        if cluster_rewards_w is not None and cluster_rewards_l is not None and cluster_weights is not None:
            stats['cluster'] = dict(
                cluster_rewards_chosen=cluster_rewards_w.mean().detach(),
                cluster_rewards_rejected=cluster_rewards_l.mean().detach(),
                cluster_reward_margin=(cluster_rewards_w - cluster_rewards_l).mean().detach(),
                cluster_weight_entropy=(-cluster_weights * torch.log(cluster_weights + 1e-10)).sum(dim=1).mean().detach(),  # Entropy of cluster distribution (higher = more uncertain)
                cluster_weight_max=cluster_weights.max(dim=1)[0].mean().detach(),  # Average max weight (higher = more confident)
            )
        
        return dpo_loss, flatten_dict(stats)
    
    def step(
        self,
        queries: torch.LongTensor,
        responses_w: torch.LongTensor,
        responses_l: torch.LongTensor,
        preference_mask: Optional[torch.BoolTensor] = None,
        user_ids: Optional[torch.Tensor] = None,
    ):
        """
        Cluster-conditioned DPO step with user_ids support.
        """
        assert queries.ndim == 2 and responses_w.ndim == 2 and responses_l.ndim == 2
        self.model.train()
        bs = self.config.batch_size
        sub_bs = self.config.mini_batch_size
        assert bs % sub_bs == 0
        
        all_stats = []
        for i in range(0, bs, sub_bs):
            queries_ = queries[i : i + sub_bs]
            responses_w_ = responses_w[i : i + sub_bs]
            responses_l_ = responses_l[i : i + sub_bs]
            preference_mask_ = preference_mask[i : i + sub_bs] if preference_mask is not None else None
            user_ids_ = user_ids[i : i + sub_bs] if user_ids is not None else None

            loss, stats = self._step(
                queries=queries_,
                responses_w=responses_w_,
                responses_l=responses_l_,
                preference_mask=preference_mask_,
                user_ids=user_ids_,
            )
            
            self.optimizer.zero_grad()
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.current_step += 1
            
            all_stats.append(stats)
        
        # Aggregate stats
        aggregated_stats = {}
        for key in all_stats[0].keys():
            values = [s[key] for s in all_stats if key in s]
            if values:
                if isinstance(values[0], torch.Tensor):
                    aggregated_stats[key] = torch.stack(values).mean()
                else:
                    aggregated_stats[key] = np.mean(values)
        
        return aggregated_stats

