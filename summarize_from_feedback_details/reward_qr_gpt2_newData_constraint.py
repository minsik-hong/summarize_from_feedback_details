import os
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from accelerate import Accelerator
from accelerate.utils import gather_object, broadcast
from datasets import load_dataset, Dataset
from rich.console import Console
from rich.pretty import pprint
from rich.table import Table
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    get_scheduler,
)
from huggingface_hub import HfApi

api = HfApi()


@dataclass
class Args:
    # common args
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    cuda: bool = True
    """Whether to use cuda if available."""
    run_name: Optional[str] = None
    """a unique name of this run"""
    load_from_cache_file: bool = False
    """Whether to load data from the local cache file in `dataset.map`"""
    deepspeed: bool = False
    """Whether to use deepspeed to train the model"""
    print_sample_output_freq: int = 220
    """How often to print sample output"""
    run_eval: bool = True
    """Whether to run evaluation"""

    # optimizer args
    eps: float = 1e-5
    """the epsilon value for the optimizer"""
    lr: float = 3e-6
    """the learning rate"""
    optimizer: Literal["adam", "adamw"] = "adamw"
    """Which optimizer to use"""
    scheduler: str = "cosine"
    """Which scheduler to use"""
    warm_up_steps: int = 0
    """Number of warm up steps for the scheduler"""

    # various batch sizes
    world_size: Optional[int] = None
    """The number of processes (GPUs) to use"""
    num_train_epochs: int = 5 # 1
    """Number of epochs to train"""
    num_updates: Optional[int] = None
    """The number of updates to train"""
    gradient_accumulation_steps: int = 64 #8
    """The number of gradient accumulation steps"""
    local_micro_batch_size: Optional[int] = 4 #1
    """The micro batch size per GPU (HF's `per_device_train_batch_size`)"""
    total_episodes: Optional[int] = 92832 #92832
    """The total number of episodes in the dataset"""
    micro_batch_size: Optional[int] = None
    """The micro batch size across devices (HF's `per_device_train_batch_size` * `world_size`)"""
    local_batch_size: Optional[int] = None
    """The batch size per GPU (HF's `per_device_train_batch_size` * `gradient_accumulation_steps`)"""
    batch_size: Optional[int] = None
    """The batch size across devices (HF's `per_device_train_batch_size` * `world_size` * `gradient_accumulation_steps`)"""
    local_eval_batch_size: int = 1
    """per rank eval batch size"""

    # other args
    # base_model: str = "EleutherAI/pythia-160m"
    base_model: str = "openai-community/gpt2-medium"
    """the name of the pretrained model to use"""
    # query_dataset: str = "vwxyzjn/summarize_from_feedback_tldr_3_filtered_oai_preprocessing_1706381144"
    query_dataset: str = "imminsik/summarize_from_feedback_tldr_3_filtered_oai_preprocessing_gpt2_1753693335"
    """the query dataset"""
    reward_model_path: str = ""
    """the path to the reward model"""
    # sft_model_path: str = "EleutherAI/pythia-160m"
    sft_model_path: str = "openai-community/gpt2-medium"
    """the path to the sft model"""
    # label_dataset: str = "vwxyzjn/summarize_from_feedback_oai_preprocessing_1706381144"
    label_dataset: str = "imminsik/summarize_from_feedback_oai_preprocessing_gpt2_1753693335"
    """the name of the dataset to use for labels in `https://huggingface.co/datasets/vwxyzjn/lm-human-preferences`"""

    # wandb and HF tracking configs
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "tldr_summarize"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    push_to_hub: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: Optional[str] = None
    """the user or org name of the model repository from the Hugging Face Hub"""
    hf_repo_id: Optional[str] = None
    """the id of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_revision: Optional[str] = None
    """the revision of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_url: Optional[str] = None
    """the url of the saved model in the Hugging Face Hub (will be autoset)"""
    output_dir: str = "models/reward_model"
    """Where to save the model"""


def parse_args() -> tuple[Args, Accelerator]:
    args = tyro.cli(Args)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    args.world_size = accelerator.num_processes
    args.local_batch_size = args.local_micro_batch_size * args.gradient_accumulation_steps
    args.micro_batch_size = int(args.local_micro_batch_size * args.world_size)
    args.batch_size = int(args.local_batch_size * args.world_size)
    time_tensor = torch.tensor(int(time.time()), device=accelerator.device)
    time_int = broadcast(time_tensor, 0).item()  # avoid different timestamps across processes
    args.run_name = f"{args.exp_name}__{args.seed}__{time_int}"
    if args.push_to_hub:
        if args.hf_repo_id is None: # auto-generate one
            args.hf_repo_id = f"{args.base_model.replace('/', '_')}__{args.exp_name}__tldr"
        if args.hf_entity is None:  # find the current user
            args.hf_entity = api.whoami()["name"]
        if "/" not in args.hf_repo_id: # prepend the current user
            args.hf_repo_id = f"{args.hf_entity}/{args.hf_repo_id}"
        if args.hf_repo_revision is None:  # auto-generate one
            args.hf_repo_revision = args.run_name
        args.hf_repo_url = f"https://huggingface.co/{args.hf_repo_id}/tree/{args.hf_repo_revision}"
    return args, accelerator


# taken from https://github.com/vwxyzjn/direct-preference-optimization/blob/f8b8c0f49dc92a430bae41585f9d467d3618fe2f/utils.py#L99
def disable_dropout(model: torch.nn.Module):
    """Disable dropout in a model."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


def print_rich_table(title: str, df: pd.DataFrame, console: Console) -> Table:
    table = Table(show_lines=True)
    for column in df.columns:
        table.add_column(column)
    for _, row in df.iterrows():
        table.add_row(*row.astype(str).tolist())
    console.rule(f"[bold red]{title}")
    console.print(table)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.normal_(layer.weight, std=std)
    torch.nn.init.constant_(layer.bias, val=bias_const)
    return layer


# class ScalarModelConfig(PretrainedConfig):
#     def __init__(
#         self,
#         base_model: str = "EleutherAI/pythia-160m",
#         base_config: PretrainedConfig = AutoConfig.from_pretrained("EleutherAI/pythia-160m"),
#         hidden_size: int = 768,
#         bias: float = 0.0,
#         **kwargs,
#     ):
#         super().__init__(**kwargs)
#         self.base_model = base_model
#         self.base_config = base_config
#         self.hidden_size = hidden_size
#         self.bias = bias

# # Scalar 버전
# class ScalarModel(PreTrainedModel):
#     config_class = ScalarModelConfig

#     def __init__(self, config: ScalarModelConfig):
#         super().__init__(config)
#         self.config = config
#         self.lm_backbone = AutoModel.from_pretrained(
#             config.base_model,
#             config=self.config.base_config,
#             trust_remote_code=True,
#         )
#         self.scalar_head = layer_init(
#             nn.Linear(self.config.hidden_size, 1),
#             std=1 / np.sqrt(self.config.hidden_size + 1),
#         )

#     def forward(self, **kwargs):
#         output = self.lm_backbone(**kwargs)
#         reward = self.scalar_head(output.hidden_states[-1]) - self.config.bias # [2B, T, 1] ?
#         return reward

# QR 버전
class QuantileModelConfig(PretrainedConfig):
    def __init__(
        self,
        base_model: str = "openai-community/gpt2-medium",
        base_config: PretrainedConfig = AutoConfig.from_pretrained("openai-community/gpt2-medium"),
        hidden_size: int = 768,
        bias: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.base_config = base_config
        self.hidden_size = hidden_size
        self.bias = bias

# QR 버전 
class QuantileModel(PreTrainedModel):
    config_class = QuantileModelConfig

    def __init__(self, config: QuantileModelConfig):
        super().__init__(config)
        self.config = config
        self.n_quantiles = 10  # 1이 아닌 config.n_quantiles 개수로 변경
        self.lm_backbone = AutoModel.from_pretrained(
            config.base_model,
            config=self.config.base_config,
            trust_remote_code=True,
        )
        self.quantile_head = layer_init(
            nn.Linear(self.config.hidden_size, self.n_quantiles), # 1이 아닌 config.n_quantiles 개수로 변경
            std=1 / np.sqrt(self.config.hidden_size + self.n_quantiles), # 1이 아닌 config.n_quantiles 개수로 변경
        )

    def forward(self, **kwargs):
        output = self.lm_backbone(**kwargs)
        reward = self.quantile_head(output.hidden_states[-1]) - self.config.bias  # [2B, T, N] ?
        
        # 디버깅 출력
        # print(f"reward shape: {reward.shape}")
        
        return reward



def first_true_indices(bools, dtype=torch.long):
    """
    Takes an N-dimensional bool tensor and returns an (N-1)-dimensional tensor of integers giving
    the position of the first True in each "row".

    Returns the length of the rows (bools.size(-1)) if no element is True in a given row.
    """
    row_len = bools.size(-1)
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=bools.device)
    return torch.min(zero_or_index, dim=-1).values


# def get_reward(model, query_responses, tokenizer, context_length=0):
#     attention_mask = query_responses != tokenizer.pad_token_id
#     # position_ids = attention_mask.cumsum(1) - attention_mask.long()  # exclusive cumsum
#     input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
#     reward_logits = model(
#         input_ids=input_ids,
#         attention_mask=attention_mask,
#         # position_ids=position_ids,
#         return_dict=True,
#         output_hidden_states=True,
#     )
#     sequence_lengths = first_true_indices(query_responses[:, context_length:] == tokenizer.pad_token_id) - 1 + context_length
#     # https://github.com/huggingface/transformers/blob/dc68a39c8111217683bf49a4912d0c9018bab33d/src/transformers/models/gpt2/modeling_gpt2.py#L1454
#     return reward_logits[torch.arange(reward_logits.size(0), device=reward_logits.device), sequence_lengths].squeeze(-1)

# QR 버전 

def get_quantile_expected_reward(model, query_responses, tokenizer, context_length=0):
    attention_mask = query_responses != tokenizer.pad_token_id
    # position_ids = attention_mask.cumsum(1) - attention_mask.long()  # exclusive cumsum
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)

    quantile_logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
        output_hidden_states=True,
    )  # [2B, T, N]

    # # quantile expected value -> [2B, T, 1], 
    # # quantile_logits = [2, 638, 10]
    # quantile_expected_logits = quantile_logits.mean(dim=-1, keepdim=True)
    # # quantile_expected_logits = [2, 638, 1]

    # # 마지막 유효 토큰 인덱스 계산
    # sequence_lengths = (
    #     first_true_indices(query_responses[:, context_length:] == tokenizer.pad_token_id)
    #     - 1 + context_length
    # )  # [2B]

    # # 마지막 토큰의 평균 reward 추출 -> [2B]
    # batch_indices = torch.arange(quantile_logits.size(0), device=quantile_logits.device)
    # quantile_expected_rewards = quantile_expected_logits[batch_indices, sequence_lengths].squeeze(-1)  # [2B]

    # 디버깅 출력
    # print(f"quantile_expected_rewards shape: {quantile_expected_rewards.shape}")
    
    # logits [2B, T, N] 형태로 반환
    return quantile_logits, context_length

## QR 버전 Loss 함수 
# def quantile_huber_loss(preds: torch.Tensor, targets: torch.Tensor):
#     td_errors = targets.unsqueeze(1) - preds  # [B, N]
#     huber_loss = torch.where(
#         td_errors.abs() <= 1.0,
#         0.5 * td_errors.pow(2),
#         td_errors.abs() - 0.5
#     )
#     device = preds.device
#     n_quantiles = preds.size(1)  # N 추출
#     taus = torch.linspace(0.5 / n_quantiles, 1 - 0.5 / n_quantiles, n_quantiles, device=device)
#     taus = taus.view(1, -1)  # [1, N] for broadcasting
#     weight = (taus - (td_errors.detach() < 0).float()).abs()  # [B, N]
#     loss = (weight * huber_loss).sum(-1).mean()
#     return loss

def quantile_huber_loss(preds: torch.Tensor, targets: torch.Tensor, taus: torch.Tensor):
    # preds: [B, N], targets: [B], taus: [N]
    td_errors = targets.unsqueeze(1) - preds  # [B, N]
    huber_loss = torch.where(
        td_errors.abs() <= 1.0, # condition
        0.5 * td_errors.pow(2), # true case
        td_errors.abs() - 0.5   # false case
    )   
    weight = (taus - (td_errors.detach() < 0).float()).abs()  # [B, N]
    loss = (weight * huber_loss).sum(-1).mean()
    return loss

def evaluate(args: Args, accelerator, tokenizer, model, dataloader, eval_split=int):
    
    model.eval()
    all_preferred, all_rejected = [], []  # Quantile 저장용 리스트

    with torch.no_grad():
        items = defaultdict(list)
        for data in tqdm(dataloader):
            query_responses = torch.cat((data["query_chosen_token"], data["query_rejected_token"]), dim=0)
            with accelerator.accumulate(model):

            
                # 기존 코드
                # predicted_reward = get_reward(model, query_responses, tokenizer)# [2B, T, 1]
                
                # =========QR 코드======= 
                # Quantile 기반 reward 기대값 추출
                quantile_logits, context_length_1 = get_quantile_expected_reward(model, query_responses, tokenizer)  

                # quantile expected value -> [2B, T, 1], 
                # quantile_logits = [2, 638, 10]
                quantile_expected_logits = quantile_logits.mean(dim=-1, keepdim=True)
                # quantile_expected_logits = [2, 638, 1]

                # 마지막 유효 토큰 인덱스 계산
                sequence_lengths = (
                    first_true_indices(query_responses[:, context_length_1:] == tokenizer.pad_token_id)
                    - 1 + context_length_1
                )  # [2B]

                # 마지막 토큰의 평균 reward 추출 -> [2B]
                batch_indices = torch.arange(quantile_logits.size(0), device=quantile_logits.device)
                quantile_expected_rewards = quantile_expected_logits[batch_indices, sequence_lengths].squeeze(-1)  # [2B]
                # ======================
                
                
                # === [추가] 보상 값 csv 저장용 ===
                quantile_last_token = quantile_logits[batch_indices, sequence_lengths]  # [2B, N]
                preferred_q = quantile_last_token[:data['query_chosen_token'].shape[0]].cpu().float().numpy()
                rejected_q = quantile_last_token[data['query_chosen_token'].shape[0]:].cpu().float().numpy()
                all_preferred.append(preferred_q)
                all_rejected.append(rejected_q)
                # =======================


                # 선택된 응답과 거부된 응답 분리
                chosen_rewards = quantile_expected_rewards[:data['query_chosen_token'].shape[0]]
                rejected_rewards = quantile_expected_rewards[data['query_chosen_token'].shape[0]:]
                accuracy = (chosen_rewards > rejected_rewards).float()

                # 수집
                accuracy = accelerator.gather(accuracy)
                chosen_rewards = accelerator.gather(chosen_rewards)
                rejected_rewards = accelerator.gather(rejected_rewards)
                
            for k in data:
                data[k] = gather_object(data[k])
            for i in range(len(accuracy)):
                items["query"].append(tokenizer.decode(data["query_token"][i], skip_special_tokens=True))
                items["chosen"].append(tokenizer.decode(data["chosen_token"][i]))
                items["rejected"].append(tokenizer.decode(data["rejected_token"][i]))
                items["batch"].append(data["batch"][i])
                items["split"].append(data["split"][i])
                items["confidence"].append(data["extra.confidence"][i].item())
                items["choice"].append(data["choice"][i].item())
                items["policies"].append(data["policies"][i])
                items["chosen_policy"].append(data["chosen_policy"][i])
                items["rejected_policy"].append(data["rejected_policy"][i])
                items["accuracy"].append(accuracy[i].item())
                items["chosen_rewards"].append(chosen_rewards[i].item())
                items["rejected_rewards"].append(rejected_rewards[i].item())
                
            
            # ===== [추가] Quantile CSV 저장 =====
            # if (
            #     accelerator.is_main_process and len(all_preferred) > 0
            #     and global_step % 10 == 0
            # ):
            # if accelerator.is_main_process and len(all_preferred) > 0:

            preferred_full = np.concatenate(all_preferred, axis=0)  # [전체 B, N]
            rejected_full = np.concatenate(all_rejected, axis=0)    # [전체 B, N]

            save_dir = f"eval_tables/{args.run_name}/quantile_logs"
            os.makedirs(save_dir, exist_ok=True)

            pd.DataFrame(preferred_full, columns=[f"q{i}" for i in range(preferred_full.shape[1])]) \
                .to_csv(f"{save_dir}/preferred_rewards_quantile_step_{eval_split}_test.csv", index=False)
            pd.DataFrame(rejected_full, columns=[f"q{i}" for i in range(rejected_full.shape[1])]) \
                .to_csv(f"{save_dir}/rejected_rewards_quantile_step_{eval_split}_test.csv", index=False)
                    
            # ===== [추가 끝] =====

    model.train()
    return pd.DataFrame(items)


# def train(args: Args):
if __name__ == "__main__":
    args, accelerator = parse_args()
    local_seed = args.seed + accelerator.process_index * 100003  # Prime

    # load dataset
    dataset = load_dataset(args.label_dataset, split="train")
    dataset = dataset.shuffle(seed=local_seed)
    dataset = dataset.select(range(args.total_episodes))
    dataset = dataset.with_format(
        "torch",
        columns=[
            "query_token",
            "chosen_token",
            "query_chosen_token",
            "rejected_token",
            "query_rejected_token",
            "batch",
            "split",
        ],
    )
    dataloader = DataLoader(dataset, batch_size=args.local_micro_batch_size)
    eval_datasets = []
    eval_dataloaders = {}
    for split in ["validation", "validation_cnndm"]:
        validation_dataset = load_dataset(args.label_dataset, split=split).flatten()
        validation_dataset = validation_dataset.with_format(
            "torch",
            columns=[
                "query_token",
                "choice",
                "chosen_token",
                "query_chosen_token",
                "rejected_token",
                "query_rejected_token",
                "batch",
                "split",
                "extra.confidence",
                "chosen_policy",
                "rejected_policy",
                "policies",
            ],
        )
        eval_datasets.append(validation_dataset)
        eval_dataloaders[split] = DataLoader(validation_dataset, batch_size=args.local_eval_batch_size)
        accelerator.print("The number of samples in validation_dataset", len(validation_dataset))
    accelerator.print("The number of samples in dataset", len(dataset))
    args.total_episodes = len(dataset)
    args.num_updates = args.total_episodes // args.batch_size

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        padding_side="right",
        trust_remote_code=True,
    )
    # we use the padding token manually but do not resize the token embedding of the model
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    console = Console(force_terminal=True)
    writer = SimpleNamespace()  # dummy writer
    writer.add_scalar = lambda x, y, z: None
    if accelerator.is_main_process:
        if args.track:
            import wandb

            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=asdict(args),
                name=args.run_name,
                save_code=True,
            )
            file_extensions = [".toml", ".lock", ".py", ".sh", ".yaml"]
            wandb.run.log_code(".", include_fn=lambda path: any([path.endswith(ext) for ext in file_extensions]))
        writer = SummaryWriter(f"runs/{args.run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        pprint(args)
    device = accelerator.device
    random.seed(local_seed)
    np.random.seed(local_seed)
    torch.manual_seed(local_seed)
    torch.backends.cudnn.deterministic = True
    model_config = AutoConfig.from_pretrained(args.sft_model_path)
    quantile_model_config =QuantileModelConfig(
        base_model=args.sft_model_path,
        base_config=model_config,
        hidden_size=model_config.hidden_size,
    )
    if len(args.reward_model_path) == 0:
        model: PreTrainedModel = QuantileModel(quantile_model_config)
    else:
        model: PreTrainedModel = QuantileModel.from_pretrained(
            args.reward_model_path,
            config=quantile_model_config,
            trust_remote_code=True,
        )
    disable_dropout(model)
    if accelerator.is_main_process:
        pprint(model_config)
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=args.eps)
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, eps=args.eps)
    scheduler = get_scheduler(
        args.scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.warm_up_steps,
        num_training_steps=args.num_updates * args.num_train_epochs,
    )

    # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
    # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
    torch.manual_seed(args.seed)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    eval_dataloaders = {split: accelerator.prepare(eval_dataloader) for split, eval_dataloader in eval_dataloaders.items()}
    torch.manual_seed(local_seed)  # reset the local seed again

    accelerator.print("===training model===")
    losses = torch.zeros((args.gradient_accumulation_steps,), device=device)
    pairwise_losses = torch.zeros((args.gradient_accumulation_steps,), device=device) # pairwise loss
    accuracies = torch.zeros((args.gradient_accumulation_steps,), device=device)
    reward_preferreds = torch.zeros((args.gradient_accumulation_steps,), device=device)
    reward_rejecteds = torch.zeros((args.gradient_accumulation_steps,), device=device)
    model.train()
    gradient_accumulation_idx = 0
    global_step = 0
    update = 0
    for epoch in range(args.num_train_epochs):
        accelerator.print(f"epoch: {epoch}")
        for data in dataloader:
            update += 1
            global_step += args.micro_batch_size
            query_responses = torch.cat((data["query_chosen_token"], data["query_rejected_token"]), dim=0)
            with accelerator.accumulate(model):
                # [2B, T, N] -> 전체 quantile logits
                quantile_logits, context_length = get_quantile_expected_reward(model, query_responses, tokenizer)  

                # 마지막 토큰 기준 추출
                sequence_lengths = (
                    first_true_indices(query_responses[:, context_length:] == tokenizer.pad_token_id) - 1 + context_length
                )  # [2B]
                
                batch_indices = torch.arange(quantile_logits.size(0), device=quantile_logits.device)
                quantile_last_token = quantile_logits[batch_indices, sequence_lengths]  # [2B, N]

                quantile_chosen = quantile_last_token[:data['query_chosen_token'].shape[0]]  # [B, N]
                quantile_rejected = quantile_last_token[data['query_chosen_token'].shape[0]:]  # [B, N]

                # Pairwise Loss용 scalar reward = 각 샘플 quantile 평균 (scalar 용도)
                chosen_rewards = quantile_chosen.mean(dim=-1)  # [B]
                rejected_rewards = quantile_rejected.mean(dim=-1)  # [B]

                pairwise_loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
                accuracy = (chosen_rewards > rejected_rewards).float().mean()

                # Quantile Huber Loss — target [B, N]
                # [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
                #
                num_quantiles = model.n_quantiles
                taus = torch.linspace(
                    0 + 1 / (2 * num_quantiles),
                    1 - 1 / (2 * num_quantiles),
                    num_quantiles,
                    device=quantile_chosen.device
                )


                # # expected QR target
                # target_chosen = chosen_rewards.detach()
                # target_rejected = rejected_rewards.detach()
                
                
                # # gradients 흐름을 차단하여 설정
                # with torch.no_grad():
                #     target_chosen = chosen_rewards
                #     target_rejected = rejected_rewards
                    
                #     pred_quantile_chosen = quantile_chosen.detach()
                #     pred_quantile_rejected = quantile_rejected.detach()

                # # Quantile Huber Loss
                # chosen_qr_loss = quantile_huber_loss(quantile_chosen, chosen_rewards, taus)
                # rejected_qr_loss = quantile_huber_loss(quantile_rejected, rejected_rewards, taus)
                
                # qr_weight = 0.1
                # qr_loss = (chosen_qr_loss + rejected_qr_loss) / 2
                # loss = qr_weight * qr_loss + pairwise_loss
                # loss = pairwise_loss

                #
                # # 최종 Loss 조합 (필요 시 pairwise loss 포함)
                # loss = qr_loss + pairwise_loss

                nc_penalty_1 = torch.relu(quantile_chosen[:, :-1] - quantile_chosen[:, 1:]).sum()
                nc_penalty_2 = torch.relu(quantile_rejected[:, :-1] - quantile_rejected[:, 1:]).sum()
                gamma = 0.01
                loss = pairwise_loss + gamma * 0.5 * (nc_penalty_1 + nc_penalty_2)

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                
            losses[gradient_accumulation_idx] = loss

            pairwise_losses[gradient_accumulation_idx] = pairwise_loss

            accuracies[gradient_accumulation_idx] = accuracy.mean() # multi gpu
            reward_preferreds[gradient_accumulation_idx] = chosen_rewards.mean()
            reward_rejecteds[gradient_accumulation_idx] = rejected_rewards.mean()
            gradient_accumulation_idx = (gradient_accumulation_idx + 1) % args.gradient_accumulation_steps
            if update > 1 and (update - 1) % args.gradient_accumulation_steps == 0:
                scheduler.step()
                train_accuracy = accelerator.gather(accuracies).mean().item()
                writer.add_scalar("train/rm/loss", accelerator.gather(losses).mean().item(), global_step)
                writer.add_scalar("train/loss/nc_penalty_1", accelerator.gather(nc_penalty_1).mean().item(), global_step)
                writer.add_scalar("train/loss/nc_penalty_2", accelerator.gather(nc_penalty_2).mean().item(), global_step)
                # writer.add_scalar("train/loss/pairwise_loss", accelerator.gather(pairwise_loss).mean().item(),
                #                   global_step)
                writer.add_scalar("train/loss/pairwise_loss", accelerator.gather(pairwise_losses).mean().item(),
                                  global_step)

                writer.add_scalar("train/rm/accuracy", train_accuracy, global_step)
                writer.add_scalar(
                    "train/rm/chosen_rewards", accelerator.gather(reward_preferreds).mean().item(), global_step
                )
                writer.add_scalar("train/rm/rejected_rewards", accelerator.gather(reward_rejecteds).mean().item(), global_step)
                writer.add_scalar("train/rm/lr", scheduler.get_last_lr()[0], global_step)
                accelerator.print(
                    f"{train_accuracy=}, {scheduler.get_last_lr()=}, {optimizer.param_groups[0]['lr']=}, {update=}"
                )


    if args.run_eval:
        accelerator.print("===evaluating start===")
        print(f"global_step: {global_step}, update: {update}")
        for eval_split in eval_dataloaders:
            evaluate_df = evaluate(args, accelerator, tokenizer, model, eval_dataloaders[eval_split], eval_split=eval_split)
            for split, row in evaluate_df[["split", "accuracy"]].groupby(["split"]).mean().iterrows():
                writer.add_scalar(f"eval/rm/{eval_split}/accuracy/split/{split}", row["accuracy"], global_step)
                accelerator.print(f"eval/rm/{eval_split}/accuracy/split/{split}: {row['accuracy']}")
            for batch, row in evaluate_df[["batch", "accuracy"]].groupby(["batch"]).mean().iterrows():
                writer.add_scalar(f"eval/rm/{eval_split}/accuracy/batch/{batch}", row["accuracy"], global_step)
                accelerator.print(f"eval/rm/{eval_split}/accuracy/batch/{batch}: {row['accuracy']}")
            for confi, row in evaluate_df[["confidence", "accuracy"]].groupby(["confidence"]).mean().iterrows():
                writer.add_scalar(f"eval/rm/{eval_split}/accuracy/confidence/{confi}", row["accuracy"], global_step)
                accelerator.print(f"eval/rm/{eval_split}/accuracy/confidence/{confi}: {row['accuracy']}")
            writer.add_scalar(f"eval/rm/{eval_split}/accuracy", evaluate_df["accuracy"].mean(), global_step)
            accelerator.print(f"eval/rm/{eval_split}/accuracy: {evaluate_df['accuracy'].mean()}")
            if accelerator.is_main_process:
                os.makedirs(f"eval_tables/{args.run_name}", exist_ok=True)
                evaluate_df.to_csv(f"eval_tables/{args.run_name}/eval_{eval_split}_{update}.csv")
                if args.track:
                    wandb.log({f"samples/{eval_split}/query_responses": wandb.Table(dataframe=evaluate_df)}, step=update)
            del evaluate_df
            torch.cuda.empty_cache()

    norm_dataset = load_dataset(args.query_dataset, split="train")
    norm_dataset = norm_dataset.with_format("torch", columns=["query_token", "reference_response_token", "query_reference_response_token"])
    norm_dataset = norm_dataset.shuffle(seed=local_seed)
    norm_dataloader = DataLoader(norm_dataset, batch_size=args.local_eval_batch_size)
    items = defaultdict(list)
    norm_dataloader = accelerator.prepare(norm_dataloader)
    rtol = 1e-2
    with torch.no_grad():
        for data in tqdm(norm_dataloader):
            reference_responses = data["reference_response_token"].to(device, non_blocking=True)
            queries = data["query_token"].to(device, non_blocking=True)
            query_responses = data["query_reference_response_token"].to(device, non_blocking=True)
            cat_query_responses = torch.cat((queries, reference_responses), dim=1)

            # 기존 코드
            # cat_predicted_reward = get_reward(model, cat_query_responses, tokenizer, context_length=queries.shape[1])
            # predicted_reward = get_reward(model, query_responses, tokenizer) [2B, T, 1]

            # =========QR 코드=======
            quantile_logits, cat_context_length = get_quantile_expected_reward(model, cat_query_responses, tokenizer, context_length=queries.shape[1])
            # quantile_logits = [2, 638, 10]
            
            # quantile expected value -> [2B, T, 1], 
            
            quantile_expected_logits = quantile_logits.mean(dim=-1, keepdim=True)
            # quantile_expected_logits = [2, 638, 1]

            # 마지막 유효 토큰 인덱스 계산
            sequence_lengths = (
                first_true_indices(cat_query_responses[:, cat_context_length:] == tokenizer.pad_token_id)
                - 1 + cat_context_length
            )  # [2B]

            # 마지막 토큰의 평균 reward 추출 -> [2B]
            batch_indices = torch.arange(quantile_logits.size(0), device=quantile_logits.device)
            cat_quantile_expected_rewards = quantile_expected_logits[batch_indices, sequence_lengths].squeeze(-1)  # [2B]
            # ======================

            
            # =========QR 코드=======
            quantile_logits, context_length_3 = get_quantile_expected_reward(model, query_responses, tokenizer) 

            
            # quantile expected value -> [2B, T, 1], 
            # quantile_logits = [2, 638, 10]
            quantile_expected_logits = quantile_logits.mean(dim=-1, keepdim=True)
            # quantile_expected_logits = [2, 638, 1]

            # 마지막 유효 토큰 인덱스 계산
            sequence_lengths = (
                first_true_indices(query_responses[:, context_length_3:] == tokenizer.pad_token_id)
                - 1 + context_length_3
            )  # [2B]

            # 마지막 토큰의 평균 reward 추출 -> [2B]
            batch_indices = torch.arange(quantile_logits.size(0), device=quantile_logits.device)
            quantile_expected_rewards = quantile_expected_logits[batch_indices, sequence_lengths].squeeze(-1)  # [2B]
            # ======================
            

            unexpecte_reward_diff = quantile_expected_rewards - cat_quantile_expected_rewards
            unexpecte_reward_diff_gt_rtol = unexpecte_reward_diff.abs() > rtol
            unexpecte_reward_diff = accelerator.gather(unexpecte_reward_diff)
            unexpecte_reward_diff_gt_rtol = accelerator.gather(unexpecte_reward_diff_gt_rtol)
            quantile_expected_rewards = accelerator.gather(quantile_expected_rewards)
            queries = accelerator.gather(queries)
            reference_responses = accelerator.gather(reference_responses)
            for i in range(len(quantile_expected_rewards)):
                items["query"].append(tokenizer.decode(queries[i], skip_special_tokens=True))
                items["reference_response"].append(tokenizer.decode(reference_responses[i]))
                items["quantile_expected_rewards"].append(quantile_expected_rewards[i].item())
                items["unexpecte_reward_diff"].append(unexpecte_reward_diff[i].item())
                items["unexpecte_reward_diff_gt_rtol"].append(unexpecte_reward_diff_gt_rtol[i].item())

    if accelerator.is_main_process:
        norm_df = pd.DataFrame(items)
        os.makedirs(f"eval_tables/{args.run_name}", exist_ok=True)
        norm_ds = Dataset.from_pandas(norm_df)
        # norm_df.to_csv(f"eval_tables/{args.run_name}/eval_{update}_normalized.csv")
        norm_ds.save_to_disk(f"eval_tables/{args.run_name}/eval_{update}_normalized")
        if args.track:
            wandb.log({"samples/normalized": wandb.Table(dataframe=norm_df)}, step=update)
        stats = {
            "mean": norm_df["quantile_expected_rewards"].mean(),
            "std": norm_df["quantile_expected_rewards"].std(),
            "max": norm_df["quantile_expected_rewards"].max(),
            "min": norm_df["quantile_expected_rewards"].min(),
            "unexpecte_reward_diff_mean": norm_df["unexpecte_reward_diff"].mean(),
            "unexpecte_reward_diff_gt_rtol_mean": norm_df["unexpecte_reward_diff_gt_rtol"].mean(),
        }
        for stat_name, stat_value in stats.items():
            writer.add_scalar(f"eval/rm/normalized_{stat_name}", stat_value, global_step)
            accelerator.print(f"Normalized Reward {stat_name.capitalize()}: {stat_value}")

    # save model
    if args.output_dir and args.num_train_epochs > 0:
        accelerator.print("===saving model start===")
        os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
        time_tensor = torch.tensor([int(time.time())], device=device)
        time_int = accelerator.gather(time_tensor)[0].item()  # avoid different timestamps across processes
        repo_name = f"{args.base_model.replace('/', '_')}__{args.exp_name}__tldr"
        repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name

        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                tokenizer.push_to_hub(repo_id=args.hf_repo_id, revision=args.hf_repo_revision)
        unwrapped: PreTrainedModel = accelerator.unwrap_model(model)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            # unwrapped.config.bias = norm_df["predicted_reward"].mean()
            unwrapped.config.bias = norm_df["quantile_expected_rewards"].mean()
            unwrapped.save_pretrained(
                args.output_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=accelerator.get_state_dict(model),
                safe_serialization=False,
            )
            if args.push_to_hub:
                unwrapped.push_to_hub(repo_id=args.hf_repo_id, revision=args.hf_repo_revision, safe_serialization=False)
                accelerator.print(f"🔥 pushed to {args.hf_repo_url}")
