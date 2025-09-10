import copy
import json
import multiprocessing
import os
import time
from dataclasses import dataclass, field
from pprint import pformat
from typing import Dict, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import tyro
from datasets import DatasetDict, load_dataset
from huggingface_hub import HfApi
from huggingface_hub.repocard import RepoCard
from rich.pretty import pprint
from transformers import AutoTokenizer

api = HfApi()

"""
ORIGINAL CLI EXAMPLES (unchanged)
poetry run python -i summarize_from_feedback_details/tldr_dataset.py \
    --base_model=EleutherAI/pythia-1b-deduped \
    --tldr_params.max_sft_response_length=53 \
    --tldr_params.max_sft_query_response_length=562 \
    --tldr_params.max_rm_response_length=169 \
    --tldr_params.max_rm_query_response_length=638 \
    --cnndm_params.max_rm_response_length=155 \
    --cnndm_params.max_rm_query_response_length=2021 \
    --push_to_hub \

# make gpt2 data
python summarize_from_feedback_details/tldr_dataset_gpt2.py \
  --base_model=openai-community/gpt2 \
  --tldr_params.max_sft_response_length 53 \
  --tldr_params.max_sft_query_response_length 562 \
  --tldr_params.max_rm_response_length 169 \
  --tldr_params.max_rm_query_response_length 638 \
  --cnndm_params.max_rm_response_length 155 \
  --cnndm_params.max_rm_query_response_length 1024 \
  --push_to_hub
"""

# ================================
# Dataclasses (original)
# ================================
@dataclass
class TaskQueryHParams:
    length: Optional[int] = None
    format_str: Optional[str] = None
    truncate_field: Optional[str] = None
    truncate_text: Optional[str] = None
    padding: Optional[Literal["empty_space", "pad_token"]] = None
    pad_token: Optional[str] = None
    pad_side: Optional[str] = None
    max_sft_response_length: Optional[int] = None
    max_sft_query_response_length: Optional[int] = None
    max_rm_response_length: Optional[int] = None
    max_rm_query_response_length: Optional[int] = None


@dataclass
class Args:
    base_model: str = "openai-community/gpt2"  # "gpt2"
    hf_entity: str = "imminsik"
    push_to_hub: bool = True
    check_length_correctness: bool = True
    debug: bool = False

    # === NEW: axis selection & files ===
    # -----------------------------------
    # \u25B6\uFE0F MOD: axis-related options added
    axis: Optional[Literal["accuracy", "coherence", "coverage"]] = None
    axis_dir: Optional[str] = None  # directory containing accuracy_sorted.json, coherence_sorted.json, coverage_sorted.json
    drop_ties: bool = False  # if True, drop pairs with equal axis scores; else keep original choice on ties

    tldr_params: TaskQueryHParams = field(
        default_factory=lambda: TaskQueryHParams(
            length=512,
            format_str="SUBREDDIT: r/{subreddit}\n\nTITLE: {title}\n\nPOST: {post}\n\nTL;DR:",
            truncate_field="post",
            truncate_text="\n",
            padding="pad_token",
            pad_side="left",
            max_sft_response_length=53,
            max_sft_query_response_length=562,
            max_rm_response_length=169,
            max_rm_query_response_length=638,
        )
    )
    cnndm_params: TaskQueryHParams = field(
        default_factory=lambda: TaskQueryHParams(
            length=1024 - 128,
            format_str="Article:\n{article}\n\nTL;DR:\n",
            truncate_field="article",
            truncate_text="\n",
            padding="pad_token",
            pad_side="left",
            max_rm_response_length=155,
            max_rm_query_response_length=1024,
        )
    )


# ================================
# Helpers (original)
# ================================

def _ensure_length(toks, l, pad_sequence=None, pad_side=None, truncate_side=None):
    assert pad_side in (None, "left", "right")
    assert truncate_side in (None, "left", "right")
    if len(toks) < l:
        assert pad_sequence is not None
        pad_amt = l - len(toks)
        assert len(pad_sequence) >= pad_amt, f"{len(pad_sequence)} < {pad_amt}"
        if pad_side is None:
            assert len(toks) == l, f"Needed to pad! {len(toks)} < {l}"
            return toks
        elif pad_side == "left":
            return pad_sequence[-pad_amt:] + toks
        else:
            assert pad_side == "right"
            return toks + pad_sequence[:pad_amt]
    if truncate_side is None:
        assert len(toks) == l, f"Needed to truncate! {len(toks)} > {l}"
        return toks
    elif truncate_side == "left":
        return toks[-l:]
    else:
        assert truncate_side == "right"
        return toks[:l]


def _get_query_padding_for_task(encoder, hparams: TaskQueryHParams):
    return hparams.pad_token * hparams.length


def process_query(query_info: Dict[str, str], *, encoder, hparams: TaskQueryHParams, pad_sequence=None):
    if pad_sequence is None:
        pad_sequence = _get_query_padding_for_task(encoder, hparams)
    if isinstance(query_info, str):
        query_info = dict(query=query_info)
    else:
        # copy to avoid mutating input
        query_info = dict(**query_info)

    format_str = hparams.format_str or "{query}"
    query_tokens = encoder.encode(format_str.format(**query_info))
    truncate_field = hparams.truncate_field or "query"

    if truncate_field not in query_info:
        raise ValueError(f"Could not truncate field {truncate_field}, found fields: {query_info.keys()}!")
    while len(query_tokens) > hparams.length:
        if not len(query_info[truncate_field]):
            raise ValueError("Could not truncate enough!")

        i = -1  # default to just remove one character
        if hparams.truncate_text:
            try:
                i = query_info[truncate_field].rindex(hparams.truncate_text)
            except ValueError:
                pass
        query_info[truncate_field] = query_info[truncate_field][:i]
        query_tokens = encoder.encode(format_str.format(**query_info))

    query_token = _ensure_length(query_tokens, hparams.length, pad_side=hparams.pad_side, pad_sequence=pad_sequence)
    query = encoder.decode(query_token, skip_special_tokens=True).lstrip()
    return dict(
        query_token=query_token,
        query=query,
    )


def ceil_div(a, b):
    return (a - 1) // b + 1


# ================================
# NEW: Axis utilities
# ================================
# \u25B6\uFE0F MOD: functions to load axis json and compute preferred summary per pair

def _axis_file_name(axis: str) -> str:
    return f"{axis}_sorted.json"


def load_axis_pairs(axis: str, axis_dir: str):
    """Load axis file -> mapping from frozenset({s0,s1}) to (score0, score1).
    The json is a list of items with "matched_summaries": [{"text":..., "axes": {axis: int}}, ...]
    """
    path = os.path.join(axis_dir, _axis_file_name(axis))
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mapping: Dict[frozenset, Tuple[int, int]] = {}
    for item in data:
        ms = item.get("matched_summaries", [])
        if len(ms) != 2:
            continue
        t0 = ms[0]["text"].strip()
        t1 = ms[1]["text"].strip()
        s0 = int(ms[0]["axes"][axis])
        s1 = int(ms[1]["axes"][axis])
        mapping[frozenset({t0, t1})] = (s0, s1)
    return mapping


from datasets import DatasetDict, Dataset, Features, Value

def annotate_axis_override(ds_dict: DatasetDict, *, axis: str, axis_dir: str, drop_ties: bool = False) -> DatasetDict:
    """
    axis_jsons(accuracy_sorted.json 등)에 근거해 override choice(0/1)를 기록한다.
    - 적용 대상이 아니면 -1 저장
    - 동률(drop_ties=True)이면 -1 저장하여 제외 신호로 사용
    """
    assert axis in {"accuracy", "coherence", "coverage"}
    # axis 파일 로드: order -> (choice_override, score_chosen, score_rejected)
    import json, os

    axis_path = os.path.join(axis_dir, _axis_file_name(axis))
    with open(axis_path, "r", encoding="utf-8") as f:
        axis_items = json.load(f)

    # ex) order 기준으로 override 판단하는 매핑 구성
    # (order는 원본 comparisons의 샘플 순서/인덱스라고 가정. 인덱스 기반이면 with_indices=True로 접근)
    order2override = {}  # order -> (override_choice or -1, chosen_score, rejected_score)
    for item in axis_items:
        ms = item["matched_summaries"]
        # 두 요약의 점수 추출
        s0, s1 = ms[0]["axes"][axis], ms[1]["axes"][axis]
        if s0 == s1:
            if drop_ties:
                order2override[item["order"]] = (-1, s0, s1)  # 동률 드랍
            else:
                order2override[item["order"]] = (None, s0, s1)  # 동률이면 원래 choice 유지
        else:
            # 높은 점수가 chosen(=1)인지, reject(=0)인지 axis 기준으로 재지정
            override_choice = 0 if s0 > s1 else 1
            order2override[item["order"]] = (override_choice, s0, s1)

    new_dict = DatasetDict()
    for split, ds in ds_dict.items():
        # 1) 기본 컬럼을 먼저 추가 (모든 row가 동일 스키마 유지)
        if "_override_choice" not in ds.column_names:
            ds = ds.add_column("_override_choice", [-1] * len(ds))
        if "_axis_score_chosen" not in ds.column_names:
            ds = ds.add_column("_axis_score_chosen", [-1] * len(ds))
        if "_axis_score_rejected" not in ds.column_names:
            ds = ds.add_column("_axis_score_rejected", [-1] * len(ds))

        # 2) order(=인덱스) 기반으로 값 업데이트
        def mark_and_keep(example, idx):
            # 기본값
            override = -1
            sc0 = -1
            sc1 = -1
            if idx in order2override:
                ov, s0, s1 = order2override[idx]
                sc0, sc1 = s0, s1
                if ov is None:
                    # 동률 but keep original -> override는 -1로 두고 나중에 원래 choice 사용
                    override = -1
                else:
                    override = ov  # 0 or 1
            # 항상 동일한 컬럼 반환! (스키마 일관성)
            example["_override_choice"] = override
            example["_axis_score_chosen"] = sc0
            example["_axis_score_rejected"] = sc1
            return example

        ds = ds.map(mark_and_keep, with_indices=True, num_proc=1)  # num_proc>1 가능하나 디버그는 1 권장
        new_dict[split] = ds

    return new_dict



# ================================
# Main
# ================================
if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.hf_entity is None:
        args.hf_entity = api.whoami()["name"]
        assert isinstance(args.hf_entity, str)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # post init
    if args.tldr_params.padding == "empty_space":
        args.tldr_params.pad_token = tokenizer.encode(" ")
    else:
        args.tldr_params.pad_token = [tokenizer.pad_token_id]
    if args.cnndm_params.padding == "empty_space":
        args.cnndm_params.pad_token = tokenizer.encode(" ")
    else:
        args.cnndm_params.pad_token = [tokenizer.pad_token_id]
    pprint(args)

    timestamp = int(time.time())

    # ================================
    # SFT dataset (original pipeline)
    # ================================
    sft_ds = load_dataset("vwxyzjn/summarize_from_feedback_tldr_3_filtered")

    def process_query_data(x):
        # x['summary'] has NO leading space in this dataset -> add one and append <|endoftext|>
        reference_response = f" {x['summary']}<|endoftext|>"
        y = {
            **process_query(x, encoder=tokenizer, hparams=args.tldr_params),
            "reference_response": reference_response,
            "reference_response_token": tokenizer.encode(
                reference_response,
                padding="max_length",
                max_length=args.tldr_params.max_sft_response_length,
                truncation=True,
            ),
            "reference_response_token_len": len(tokenizer.encode(reference_response)),
        }
        y["query_reference_response"] = y["query"].strip() + y["reference_response"]
        if args.tldr_params.padding == "empty_space":
            y["query_reference_response_token"] = y["query_token"] + y["reference_response_token"]
        else:
            y["query_reference_response_token"] = tokenizer.encode(
                y["query_reference_response"],
                padding="max_length",
                max_length=args.tldr_params.max_sft_query_response_length,
                truncation=True,
            )
        y["query_reference_response_token_response_label"] = copy.deepcopy(y["query_reference_response_token"])
        unpadded_query_token = [t for t in y["query_token"] if t != tokenizer.pad_token_id]
        y["query_reference_response_token_response_label"][: len(unpadded_query_token)] = [
            tokenizer.pad_token_id for _ in range(len(unpadded_query_token))
        ]
        y["query_reference_response_token_len"] = len(tokenizer.encode(y["query_reference_response"]))
        return y

    sft_ds = sft_ds.map(process_query_data, load_from_cache_file=False, num_proc=1 if args.debug else multiprocessing.cpu_count())

    if args.push_to_hub:
        sft_dataset_hf_path = f"{args.hf_entity}/summarize_from_feedback_tldr_3_filtered_oai_preprocessing_gpt2_{timestamp}"
        sft_ds.push_to_hub(sft_dataset_hf_path)
        sft_ds.save_to_disk(f"/home/hail/Distribution_RLHF/minsik_huggigface_dataset/summarize_from_feedback_tldr_3_filtered_oai_preprocessing_gpt2_{timestamp}")
        sft_card = RepoCard.load(sft_dataset_hf_path, repo_type="dataset")
        sft_card.text = f"""\
# TL;DR SFT Dataset for OpenAI's [Summarize from Feedback](https://openai.com/blog/summarization/) task

The dataset is directly taken from https://github.com/openai/summarize-from-feedback/tree/700967448d10004279f138666442bf1497d0e705#reddit-tldr-dataset

These columns are taken directly from the aforementioned dataset:

* **id**: unique identifier for the post
* **subreddit**: subreddit the post was taken from
* **title**: title of the post
* **post**: body of the post
* **summary**: summary of the post
* **reference_response**: reference response for the post

These columns are added by this preprocessing script:
* **query**: length-limited query for summarization
* **query_token**: tokenized version of `query`
* **reference_response_token**: tokenized version of `reference_response`
* **reference_response_token_len**: length of `reference_response_token`
* **query_reference_response**: concatenation of `query.strip()` and `reference_response`
* **query_reference_response_token**: tokenized version of `query_reference_response`
* **query_reference_response_token_len**: length of `query_reference_response_token`

# Args
```python
{pformat(vars(args))}
```
"""
        sft_card.push_to_hub(sft_dataset_hf_path, repo_type="dataset")

    # ================================
    # RM dataset (comparisons) + filters
    # ================================
    cnndm_batches = ["batch0_cnndm", "cnndm0", "cnndm2"]
    label_ds = load_dataset("openai/summarize_from_feedback", "comparisons")

    # --- Keep CNNDM validation aside (original behavior) ---
    print("Split out 'validation_cnndm' from original validation...")
    label_ds["validation_cnndm"] = label_ds["validation"].filter(
        lambda x: x["batch"] in cnndm_batches,
        num_proc=1 if args.debug else multiprocessing.cpu_count(),
        load_from_cache_file=False, 
    )
    label_ds["validation"] = label_ds["validation"].filter(
        lambda x: x["batch"] not in cnndm_batches,
        num_proc=1 if args.debug else multiprocessing.cpu_count(),
        load_from_cache_file=False, 
    )

    
    # === NEW: keep only {train=validation, validation_cnndm} =====================
    # 기존 train/test 제거하고, validation을 train으로 치환
    from datasets import DatasetDict

    _keep = {}
    if "validation" in label_ds and "validation_cnndm" in label_ds:
        _keep["train"] = label_ds["validation"]                 # validation → train
        _keep["validation_cnndm"] = label_ds["validation_cnndm"] # CNN/DM validation만 유지
    else:
        raise RuntimeError("Expected 'validation' and 'validation_cnndm' splits not found.")

    label_ds = DatasetDict(_keep)
    # ============================================================================ 

    # --- NEW: axis matching + override choice ---
    # ▶️ MOD: train에만 axis override 적용
    if args.axis and args.axis_dir:
        print(f"Applying axis filter/override for axis='{args.axis}' using dir='{args.axis_dir}' (drop_ties={args.drop_ties})...")
        # train만 따로 감싸서 처리
        label_ds["train"] = annotate_axis_override(
            DatasetDict(train=label_ds["train"]),
            axis=args.axis, axis_dir=args.axis_dir, drop_ties=args.drop_ties
        )["train"]
    else:
        print("No axis filtering applied (run with --axis <name> --axis_dir <dir> to enable).")

    # ================== FILTER: 축 파일에 매칭된 샘플만 남기기 ==================
    def _keep_example(ex):
        in_axis = ex.get("_axis_score_chosen", -1) != -1
        if not in_axis:
            return False
        if args.drop_ties:
            return ex.get("_override_choice", -1) in (0, 1)
        return True

    # MOD: axis가 지정된 경우에만, train에만 필터 적용
    if args.axis and args.axis_dir and "train" in label_ds:
        label_ds["train"] = label_ds["train"].filter(_keep_example)




    def process_response_data(x):
        # pick possibly overridden choice first
        # --- override 적용 ---
        # MOD: CNNDM 배치는 override 무시하고 원래 choice 사용
        if x["batch"] in cnndm_batches:
            ov = -1
        else:
            ov = x.get("_override_choice", -1)

        if ov in (0, 1):
            choice = ov
        else:
            choice = x["choice"]
        # ----------------------
        chosen = f"{x['summaries'][choice]['text']}<|endoftext|>"
        rejected = f"{x['summaries'][1 - choice]['text']}<|endoftext|>"

        chosen_policy = x["summaries"][choice]["policy"]
        rejected_policy = x["summaries"][1 - choice]["policy"]
        policies = "--".join(sorted([chosen_policy, rejected_policy]))
        format_params = args.cnndm_params if x["batch"] in cnndm_batches else args.tldr_params
        max_rm_response_length = (
            args.cnndm_params.max_rm_response_length
            if x["batch"] in cnndm_batches
            else args.tldr_params.max_rm_response_length
        )
        max_rm_query_response_length = (
            args.cnndm_params.max_rm_query_response_length
            if x["batch"] in cnndm_batches
            else args.tldr_params.max_rm_query_response_length
        )
        y = {
            **process_query(x["info"], encoder=tokenizer, hparams=format_params),
            "chosen": chosen,
            "chosen_token": tokenizer.encode(
                chosen, padding="max_length", max_length=max_rm_response_length, truncation=True
            ),
            "chosen_token_len": len(tokenizer.encode(chosen)),
            "rejected": rejected,
            "rejected_token": tokenizer.encode(
                rejected, padding="max_length", max_length=max_rm_response_length, truncation=True
            ),
            "rejected_token_len": len(tokenizer.encode(rejected)),
            "chosen_policy": chosen_policy,
            "rejected_policy": rejected_policy,
            "policies": policies,
        }
        y["query_chosen"] = y["query"].strip() + y["chosen"]
        if args.tldr_params.padding == "empty_space":
            y["query_chosen_token"] = y["query_token"] + y["chosen_token"]
        else:
            y["query_chosen_token"] = tokenizer.encode(
                y["query_chosen"], padding="max_length", max_length=max_rm_query_response_length, truncation=True
            )
        y["query_chosen_token_len"] = len(tokenizer.encode(y["query_chosen"]))
        y["query_rejected"] = y["query"].strip() + y["rejected"]
        if args.tldr_params.padding == "empty_space":
            y["query_rejected_token"] = y["query_token"] + y["rejected_token"]
        else:
            y["query_rejected_token"] = tokenizer.encode(
                y["query_rejected"], padding="max_length", max_length=max_rm_query_response_length, truncation=True
            )
        y["query_rejected_token_len"] = len(tokenizer.encode(y["query_rejected"]))
        y["query_token_len"] = len(tokenizer.encode(y["query"]))
        unpadded_query_token = [t for t in y["query_token"] if t != tokenizer.pad_token_id]
        y["query_chosen_token_response_label"] = copy.deepcopy(y["query_chosen_token"])
        y["query_chosen_token_response_label"][: len(unpadded_query_token)] = [
            tokenizer.pad_token_id for _ in range(len(unpadded_query_token))
        ]
        y["query_rejected_token_response_label"] = copy.deepcopy(y["query_rejected_token"])
        y["query_rejected_token_response_label"][: len(unpadded_query_token)] = [
            tokenizer.pad_token_id for _ in range(len(unpadded_query_token))
        ]
        return y

    label_ds = label_ds.map(
        process_response_data, load_from_cache_file=False, num_proc=1 if args.debug else multiprocessing.cpu_count()
    )

    # 두 스플릿의 공통 컬럼만 유지
    common_cols = None
    for split in label_ds.keys():
        cols = set(label_ds[split].column_names)
        common_cols = cols if common_cols is None else (common_cols & cols)

    for split in list(label_ds.keys()):
        to_drop = [c for c in label_ds[split].column_names if c not in common_cols]
        if to_drop:
            label_ds[split] = label_ds[split].remove_columns(to_drop)

    if args.push_to_hub:
        axis_suffix = f"_{args.axis}" if args.axis else ""
        rm_dataset_hf_path = f"{args.hf_entity}/summarize_from_feedback_oai_preprocessing_gpt2{axis_suffix}_{timestamp}"
        label_ds.push_to_hub(rm_dataset_hf_path)
        label_ds.save_to_disk(f"/home/hail/Distribution_RLHF/minsik_huggigface_dataset/summarize_from_feedback_oai_preprocessing_gpt2{axis_suffix}_{timestamp}")
    ####################################
    # visualize token length distribution
    ####################################
    calculated_tldr_params = TaskQueryHParams(
        max_sft_query_response_length=0,
        max_sft_response_length=0,
        max_rm_response_length=0,
        max_rm_query_response_length=0,
    )
    calculated_cnndm_params = TaskQueryHParams(
        max_rm_query_response_length=0,
        max_rm_response_length=0,
    )

    os.makedirs("dataset_visuals", exist_ok=True)
    num_sft_visuals = 2
    num_label_visuals = 5
    num_subplots = len(sft_ds) * num_sft_visuals + len(label_ds) * num_label_visuals
    num_cols = 3
    print(f"{num_subplots=}")
    fig, axs = plt.subplots(ceil_div(num_subplots, num_cols), num_cols, figsize=(16, 16))
    axs = axs.flatten()
    j = 0
    for _, key in enumerate(sft_ds.keys()):
        df = sft_ds[key].to_pandas()
        axs[j].hist(df["reference_response_token_len"], bins=100)
        axs[j].set_title(f"{key} split: reference response token length\nmax_length={max(df['reference_response_token_len'])}")
        axs[j + 1].hist(df["query_reference_response_token_len"], bins=100)
        axs[j + 1].set_title(
            f"{key} split: query.strip() + reference response token length\nmax_length={max(df['query_reference_response_token_len'])}"
        )
        calculated_tldr_params.max_sft_response_length = max(
            calculated_tldr_params.max_sft_response_length, max(df["reference_response_token_len"])
        )
        calculated_tldr_params.max_sft_query_response_length = max(
            calculated_tldr_params.max_sft_query_response_length, max(df["query_reference_response_token_len"])
        )
        j += num_sft_visuals

    for _, split in enumerate(label_ds.keys()):
        df = label_ds[split].to_pandas()
        axs[j].hist(df["chosen_token_len"], bins=100)
        axs[j].set_title(f"{split} split: chosen token length\nmax_length={max(df['chosen_token_len'])}")
        axs[j + 1].hist(df["rejected_token_len"], bins=100)
        axs[j + 1].set_title(f"{split} split: rejected token length\nmax_length={max(df['rejected_token_len'])}")
        axs[j + 2].hist(df["query_chosen_token_len"], bins=100)
        axs[j + 2].set_title(
            f"{split} split: query.strip() + chosen token length\nmax_length={max(df['query_chosen_token_len'])}"
        )
        axs[j + 3].hist(df["query_rejected_token_len"], bins=100)
        axs[j + 3].set_title(
            f"{split} split: query.strip() + rejected token length\nmax_length={max(df['query_rejected_token_len'])}"
        )
        axs[j + 4].hist(df["query_token_len"], bins=100)
        axs[j + 4].set_title(f"{split} split: query token length\nmax_length={max(df['query_token_len'])}")
        if split in ["train", "validation"]:
            calculated_tldr_params.max_rm_response_length = max(
                calculated_tldr_params.max_rm_response_length, max(df["chosen_token_len"]), max(df["rejected_token_len"])
            )
            calculated_tldr_params.max_rm_query_response_length = max(
                calculated_tldr_params.max_rm_query_response_length,
                max(df["query_chosen_token_len"]),
                max(df["query_rejected_token_len"]),
            )
        elif split == "validation_cnndm":
            calculated_cnndm_params.max_rm_response_length = max(
                calculated_cnndm_params.max_rm_response_length, max(df["chosen_token_len"]), max(df["rejected_token_len"])
            )
            calculated_cnndm_params.max_rm_query_response_length = max(
                calculated_cnndm_params.max_rm_query_response_length,
                max(df["query_chosen_token_len"]),
                max(df["query_rejected_token_len"]),
            )
        else:
            raise ValueError(f"Unknown dataset split: {split}")
        j += num_label_visuals
    fig.suptitle(f"{args.base_model} Tokenizer: Token length distribution")
    fig.tight_layout()
    fig.savefig("dataset_visuals/token_len.png")

    pprint({"calculated_tldr_params": calculated_tldr_params})
    pprint({"calculated_cnndm_params": calculated_cnndm_params})

    # visualize confidence distribution
    fig, axs = plt.subplots(len(label_ds), 1, figsize=(8, 8))
    axs = axs.flatten()
    label_ds = label_ds.flatten()
    for i, split in enumerate(label_ds.keys()):
        df = label_ds[split].to_pandas()
        axs[i].hist(df["extra.confidence"])
        axs[i].set_title(f"{split} split: confidence distribution")
    fig.suptitle("Confidence distribution")
    fig.tight_layout()
    fig.savefig("dataset_visuals/confidence.png")

    # visualize policies used
    fig, axs = plt.subplots(1, len(label_ds), figsize=(8, 12))
    axs = axs.flatten()
    label_ds = label_ds.flatten()
    for i, split in enumerate(label_ds.keys()):
        df = label_ds[split].to_pandas()
        cat = pd.concat([df["chosen_policy"], df["rejected_policy"]], axis=0)
        cat.hist(ax=axs[i], xrot=90, orientation="horizontal")
        axs[i].set_title(f"{split} split: policy distribution")
    fig.suptitle("Policy distribution")
    fig.tight_layout()
    fig.savefig("dataset_visuals/policies.png")

    # visualize comparison distribution
    fig, axs = plt.subplots(1, len(label_ds), figsize=(24, 30))
    axs = axs.flatten()
    label_ds = label_ds.flatten()
    for i, split in enumerate(label_ds.keys()):
        df = label_ds[split].to_pandas()
        df["policies"].hist(ax=axs[i], xrot=90, orientation="horizontal")
        axs[i].set_title(f"{split} split: policy comparison distribution")
    fig.suptitle("Policy comparison distribution")
    fig.tight_layout()
    fig.savefig("dataset_visuals/policy_comparisons.png")

    if args.push_to_hub:
        # upload the `dataset_visuals`
        api.upload_folder(
            folder_path="dataset_visuals",
            path_in_repo="dataset_visuals",
            repo_id=rm_dataset_hf_path if args.axis else f"{args.hf_entity}/summarize_from_feedback_oai_preprocessing_gpt2_{timestamp}",
            repo_type="dataset",
        )
        # upload current file
        print(f"{__file__=}")
        api.upload_file(
            path_or_fileobj=__file__,
            path_in_repo="create_dataset.py",
            repo_id=rm_dataset_hf_path if args.axis else f"{args.hf_entity}/summarize_from_feedback_oai_preprocessing_gpt2_{timestamp}",
            repo_type="dataset",
        )
        if args.axis:
            print(f"✨ Pushed to hub (RM + axis): https://huggingface.co/datasets/{rm_dataset_hf_path}")
        else:
            print(f"✨ Pushed to hub (RM): https://huggingface.co/datasets/{args.hf_entity}/summarize_from_feedback_oai_preprocessing_gpt2_{timestamp}")
            print(f"✨ Pushed to hub (SFT): https://huggingface.co/datasets/{sft_dataset_hf_path}")
