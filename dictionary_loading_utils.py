from collections import namedtuple
from dictionary_learning import AutoEncoder, JumpReluAutoEncoder
from dictionary_learning.dictionary import IdentityDict
from attribution import Submodule
from loading_utils import TranscoderSubmodule
from typing import Literal
import torch as t
from huggingface_hub import list_repo_files, hf_hub_download
from tqdm import tqdm
import numpy as np
import os

DICT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/dictionaries"

DictionaryStash = namedtuple("DictionaryStash", ["embed", "attns", "mlps", "resids"])


def _load_pythia_saes_and_submodules(
    model,
    thru_layer: int | None = None,
    separate_by_type: bool = False,
    include_embed: bool = True,
    neurons: bool = False,
    dtype: t.dtype = t.float32,
    device: t.device = t.device("cpu"),
):
    assert (
        len(model.gpt_neox.layers) == 6
    ), "Not the expected number of layers for pythia-70m-deduped"
    if thru_layer is None:
        thru_layer = len(model.gpt_neox.layers)

    attns = []
    mlps = []
    resids = []
    dictionaries = {}
    if include_embed:
        embed = Submodule(
            name="embed",
            submodule=model.gpt_neox.embed_in,
        )
        if not neurons:
            dictionaries[embed] = AutoEncoder.from_pretrained(
                f"{DICT_DIR}/pythia-70m-deduped/embed/10_32768/ae.pt",
                dtype=dtype,
                device=device,
            )
        else:
            dictionaries[embed] = IdentityDict(512, device=device, dtype=dtype)
    else:
        embed = None
    for i, layer in enumerate(model.gpt_neox.layers[: thru_layer + 1]):
        attns.append(
            attn := Submodule(
                name=f"attn_{i}",
                submodule=layer.attention,
                is_tuple=True,
            )
        )
        mlps.append(
            mlp := Submodule(
                name=f"mlp_{i}",
                submodule=layer.mlp,
            )
        )
        resids.append(
            resid := Submodule(
                name=f"resid_{i}",
                submodule=layer,
                is_tuple=True,
            )
        )
        if not neurons:
            dictionaries[attn] = AutoEncoder.from_pretrained(
                f"{DICT_DIR}/pythia-70m-deduped/attn_out_layer{i}/10_32768/ae.pt",
                dtype=dtype,
                device=device,
            )
            dictionaries[mlp] = AutoEncoder.from_pretrained(
                f"{DICT_DIR}/pythia-70m-deduped/mlp_out_layer{i}/10_32768/ae.pt",
                dtype=dtype,
                device=device,
            )
            dictionaries[resid] = AutoEncoder.from_pretrained(
                f"{DICT_DIR}/pythia-70m-deduped/resid_out_layer{i}/10_32768/ae.pt",
                dtype=dtype,
                device=device,
            )
        else:
            dictionaries[attn] = IdentityDict(512, device=device, dtype=dtype)
            dictionaries[mlp] = IdentityDict(512, device=device, dtype=dtype)
            dictionaries[resid] = IdentityDict(512, device=device, dtype=dtype)

    if separate_by_type:
        return DictionaryStash(embed, attns, mlps, resids), dictionaries
    else:
        submodules = ([embed] if include_embed else []) + [
            x
            for layer_dictionaries in zip(attns, mlps, resids)
            for x in layer_dictionaries
        ]
        return submodules, dictionaries


def load_gemma_sae(
    submod_type: Literal["embed", "attn", "mlp", "resid"],
    layer: int,
    width: Literal["16k", "65k"] = "16k",
    neurons: bool = False,
    dtype: t.dtype = t.float32,
    device: t.device = t.device("cpu"),
):
    if neurons:
        if submod_type != "attn":
            return IdentityDict(2304, device=device, dtype=dtype)
        else:
            return IdentityDict(2048, device=device, dtype=dtype)

    repo_id = "google/gemma-scope-2b-pt-" + (
        "res"
        if submod_type in ["embed", "resid"]
        else "att" if submod_type == "attn" else "mlp"
    )
    if submod_type != "embed":
        directory_path = f"layer_{layer}/width_{width}"
    else:
        directory_path = "embedding/width_4k"

    files_with_l0s = [
        (f, int(f.split("_")[-1].split("/")[0]))
        for f in list_repo_files(repo_id, repo_type="model", revision="main")
        if f.startswith(directory_path) and f.endswith("params.npz")
    ]
    optimal_file = min(files_with_l0s, key=lambda x: abs(x[1] - 100))[0]
    optimal_file = optimal_file.split("/params.npz")[0]
    return JumpReluAutoEncoder.from_pretrained(
        load_from_sae_lens=True,
        release=repo_id.split("google/")[-1],
        sae_id=optimal_file,
        dtype=dtype,
        device=device,
    )


def _load_gemma_saes_and_submodules(
    model,
    thru_layer: int | None = None,
    separate_by_type: bool = False,
    include_embed: bool = True,
    neurons: bool = False,
    dtype: t.dtype = t.float32,
    device: t.device = t.device("cpu"),
):
    assert (
        len(model.model.layers) == 26
    ), "Not the expected number of layers for Gemma-2-2B"
    if thru_layer is None:
        thru_layer = len(model.model.layers)

    attns = []
    mlps = []
    resids = []
    dictionaries = {}
    if include_embed:
        embed = Submodule(
            name="embed",
            submodule=model.model.embed_tokens,
        )
        dictionaries[embed] = load_gemma_sae(
            "embed", 0, neurons=neurons, dtype=dtype, device=device
        )
    else:
        embed = None
    for i, layer in tqdm(
        enumerate(model.model.layers[: thru_layer + 1]),
        total=thru_layer + 1,
        desc="Loading Gemma SAEs",
    ):
        attns.append(
            attn := Submodule(
                name=f"attn_{i}", submodule=layer.self_attn.o_proj, use_input=True
            )
        )
        dictionaries[attn] = load_gemma_sae(
            "attn", i, neurons=neurons, dtype=dtype, device=device
        )
        mlps.append(
            mlp := Submodule(
                name=f"mlp_{i}",
                submodule=layer.post_feedforward_layernorm,
            )
        )
        dictionaries[mlp] = load_gemma_sae(
            "mlp", i, neurons=neurons, dtype=dtype, device=device
        )
        resids.append(
            resid := Submodule(
                name=f"resid_{i}",
                submodule=layer,
                is_tuple=True,
            )
        )
        dictionaries[resid] = load_gemma_sae(
            "resid", i, neurons=neurons, dtype=dtype, device=device
        )

    if separate_by_type:
        return DictionaryStash(embed, attns, mlps, resids), dictionaries
    else:
        submodules = ([embed] if include_embed else []) + [
            x
            for layer_dictionaries in zip(attns, mlps, resids)
            for x in layer_dictionaries
        ]
        return submodules, dictionaries


def load_saes_and_submodules(
    model,
    thru_layer: int | None = None,
    separate_by_type: bool = False,
    include_embed: bool = True,
    neurons: bool = False,
    dtype: t.dtype = t.float32,
    device: t.device = t.device("cpu"),
):
    model_name = model.config._name_or_path

    if model_name == "EleutherAI/pythia-70m-deduped":
        return _load_pythia_saes_and_submodules(
            model,
            thru_layer=thru_layer,
            separate_by_type=separate_by_type,
            include_embed=include_embed,
            neurons=neurons,
            dtype=dtype,
            device=device,
        )
    elif model_name == "google/gemma-2-2b":
        return _load_gemma_saes_and_submodules(
            model,
            thru_layer=thru_layer,
            separate_by_type=separate_by_type,
            include_embed=include_embed,
            neurons=neurons,
            dtype=dtype,
            device=device,
        )
    else:
        raise ValueError(f"Model {model_name} not supported")


def load_gemma_transcoder(
    layer: int,
    width: Literal["16k"] = "16k",
    dtype: t.dtype = t.float32,
    device: t.device = t.device("cpu"),
) -> JumpReluAutoEncoder:
    """
    Load a single Gemma-2-2b transcoder from google/gemma-scope-2b-pt-transcoders.

    Transcoders have the same JumpReLU architecture as SAEs but map MLP *inputs* to MLP *outputs*:
        encode(pre_feedforward_layernorm_output) -> features
        decode(features)                         -> post_feedforward_layernorm_output approximation
    """
    repo_id = "google/gemma-scope-2b-pt-transcoders"
    directory_path = f"layer_{layer}/width_{width}"

    files_with_l0s = [
        (f, int(f.split("_")[-1].split("/")[0]))
        for f in list_repo_files(repo_id, repo_type="model", revision="main")
        if f.startswith(directory_path) and f.endswith("params.npz")
    ]
    optimal_file = min(files_with_l0s, key=lambda x: abs(x[1] - 100))[0]
    path = hf_hub_download(repo_id, optimal_file, repo_type="model")

    data = np.load(path)
    activation_dim, dict_size = data["W_enc"].shape  # (2304, 16384)
    tc = JumpReluAutoEncoder(activation_dim, dict_size)
    tc.W_enc.data = t.from_numpy(data["W_enc"].copy())
    tc.b_enc.data = t.from_numpy(data["b_enc"].copy())
    tc.W_dec.data = t.from_numpy(data["W_dec"].copy())
    tc.b_dec.data = t.from_numpy(data["b_dec"].copy())
    tc.threshold.data = t.from_numpy(data["threshold"].copy())
    return tc.to(dtype=dtype, device=device)


def load_gemma_transcoders_and_submodules(
    model,
    thru_layer: int | None = None,
    dtype: t.dtype = t.float32,
    device: t.device = t.device("cpu"),
):
    """
    Load Gemma-2-2b transcoders for MLP layers 0..thru_layer.

    Returns (submodules, dictionaries) where each submodule is a TranscoderSubmodule
    that reads from pre_feedforward_layernorm output and writes to post_feedforward_layernorm output.
    """
    assert (
        len(model.model.layers) == 26
    ), "Not the expected number of layers for Gemma-2-2B"
    if thru_layer is None:
        thru_layer = len(model.model.layers) - 1

    submodules = []
    dictionaries = {}

    for i, layer in tqdm(
        enumerate(model.model.layers[: thru_layer + 1]),
        total=thru_layer + 1,
        desc="Loading Gemma transcoders",
    ):
        tc_submod = TranscoderSubmodule(
            name=f"tc_mlp_{i}",
            pre_feedforward_ln=layer.pre_feedforward_layernorm,
            post_feedforward_ln=layer.post_feedforward_layernorm,
        )
        submodules.append(tc_submod)
        dictionaries[tc_submod] = load_gemma_transcoder(
            i, dtype=dtype, device=device
        )

    return submodules, dictionaries
