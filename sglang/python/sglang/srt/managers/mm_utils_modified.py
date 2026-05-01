"""
Multi-modality utils modified
"""

from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import nn

import sglang.srt.managers.mm_utils as mm_utils
from sglang.srt.managers.mm_utils import (
    DataEmbeddingFunc,
    MultimodalInputs,
    _adjust_embedding_length,
    _get_multimodal_mask,
    _get_precomputed_embedding,
    _is_npu,
    get_embedding_chunk,
)
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.mem_cache.multimodal_cache import EmbeddingResult, MultiModalStaticCache
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.multimodal.evs import EVSEmbeddingResult
from sglang.srt.utils import print_warning_once
from sglang.srt.utils.common import flatten_nested_list
from sglang.utils import logger


def _get_chunked_prefill_embedding(
    data_embedding_func: DataEmbeddingFunc,
    embedding_items: List[MultimodalDataItem],
    items_size: List[int],
    prefix_length: List[int],
    extend_length: List[int],
    items_offset_list: List[List[Tuple[int, int]]],
    input_ids: torch.Tensor,
    forward_batch: ForwardBatch,
    indices: List[int],
) -> tuple[torch.Tensor | None, torch.Tensor]:
    # Calculate embedding for each request, try to get it from cache to avoid repeated calculation
    embedding_list = []
    # FIXME(Xinyuan): temporary workaround for eagle3, which may have len(items_size) > len(prefix_length)
    max_iterations = min(len(items_size) - 1, len(prefix_length))
    for i in range(max_iterations):
        if items_size[i] == items_size[i + 1]:
            continue
        embedding_items_per_req = embedding_items[items_size[i] : items_size[i + 1]]
        indices_per_req = indices[items_size[i] : items_size[i + 1]]
        items_offset = items_offset_list[i]
        assert items_offset is not None, items_offset
        # if all items has been prefixed, we do not need to calculate embedding
        if all([offset_end < prefix_length[i] for _, offset_end in items_offset]):
            continue
        item_hashes = [item.hash for item in embedding_items_per_req]
        embedding_items_hash = MultiModalStaticCache.combine_hashes(item_hashes)
        embedding_per_req = mm_utils.embedding_cache.get(item_hashes)
        if embedding_per_req is None:
            embedding = data_embedding_func(
                embedding_items_per_req,
                forward_batch,
                indices_per_req,
            )
            embedding_per_req = (
                EmbeddingResult(embedding=embedding)
                if isinstance(embedding, torch.Tensor)
                else embedding
            )
            if not mm_utils.embedding_cache.set(
                embedding_items_hash, embedding_per_req
            ):
                print_warning_once(
                    "Multimodal embedding cache is full. This typically occurs when a single "
                    "embedding exceeds the cache size limit. Consider increasing the "
                    "`SGLANG_VLM_CACHE_SIZE_MB` environment variable or reducing the input "
                    "embedding size."
                )

        extend_prefix_len = prefix_length[i]
        extend_seq_len = extend_length[i] if i < len(extend_length) else 0

        if isinstance(embedding_per_req, EVSEmbeddingResult):
            item = embedding_items_per_req[0]
            input_ids, items_offset = (
                embedding_per_req.redistribute_pruned_frames_placeholders(
                    input_ids,
                    items_offset,
                    item=item,
                    extend_prefix_len=extend_prefix_len,
                    extend_seq_len=extend_seq_len,
                )
            )

        embedding_per_req_chunk, _, _ = get_embedding_chunk(
            embedding=embedding_per_req.embedding,
            extend_prefix_len=extend_prefix_len,
            extend_seq_len=extend_seq_len,
            items_offset=items_offset,
        )
        embedding_list.append(embedding_per_req_chunk)
    if len(embedding_list) == 0:
        return None, input_ids
    return torch.concat(embedding_list, dim=0), input_ids


def get_embedding_and_mask(
    data_embedding_func: DataEmbeddingFunc,
    embedding_items: List[MultimodalDataItem],
    placeholder_tensor: torch.Tensor,
    input_ids: torch.Tensor,
    items_size: List[int],
    prefix_length: List[int],
    extend_length: List[int],
    items_offset_list: List[List[Tuple[int, int]]],
    forward_batch: ForwardBatch,
    indices: List[int],
) -> Tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
    """
    Generate multimodal embeddings and create a mask for identifying their positions in the input sequence.

    Args:
        data_embedding_func: Function that generates embeddings for multimodal items
        embedding_items: List of multimodal items to embed
        placeholder_tensor: Tensor containing token IDs that serve as placeholders for multimodal content
        input_ids: The input token IDs tensor
        items_size: Cumulative sizes of multimodal items per request
        prefix_length: Prefix lengths for each request
        extend_length: Sequence lengths for each request
        items_offset_list: List of offset ranges for multimodal items in each request
        indices: List of indices for the multimodal items
    Returns:
        A tuple containing:
        - The generated embeddings tensor
        - A boolean mask tensor indicating where these embeddings should be placed
        - If EVS is used, the pruned input ids tensor; otherwise, the original input ids tensor
    """
    # 1. Get embedding
    embedding = _get_precomputed_embedding(
        embedding_items, prefix_length, extend_length, items_offset_list
    )
    if embedding is None:
        embedding, input_ids = _get_chunked_prefill_embedding(
            data_embedding_func,
            embedding_items,
            items_size,
            prefix_length,
            extend_length,
            items_offset_list,
            input_ids,
            forward_batch,
            indices,
        )
        if embedding is None:
            return None, None, input_ids
    # 2. Get mask
    if _is_npu:
        torch.npu.current_stream().synchronize()
    special_multimodal_mask = _get_multimodal_mask(input_ids, placeholder_tensor)
    # 3. Adjust embedding length if needed
    embedding = _adjust_embedding_length(embedding, special_multimodal_mask, logger)
    return embedding, special_multimodal_mask, input_ids


def embed_mm_inputs(
    forward_batch: ForwardBatch,
    mm_inputs_list: List[MultimodalInputs],
    mm_input_indices: List[int],
    extend_prefix_lens: List[int],
    extend_seq_lens: List[int],
    input_ids: torch.Tensor,
    vocab_size: int = 152697,
    input_embedding: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    multimodal_model: nn.Module = None,
    data_embedding_func_mapping: Dict[
        Modality, Callable[[List[MultimodalDataItem]], torch.Tensor]
    ] = None,
    use_deepstack: Dict[Modality, bool] = {},
) -> Optional[torch.Tensor]:
    """
    Embed multimodal inputs and integrate them with text token embeddings.

    Args:
        mm_inputs_list: List of multimodal inputs to process
        extend_prefix_lens: Prefix lengths for each request
        extend_seq_lens: Sequence lengths for each request
        input_ids: Input token IDs tensor
        input_embedding: Embedding layer for text tokens
        placeholder_tokens: Token IDs for multimodal placeholders (uses pad_values if None)

    Returns:
        Combined embedding tensor with multimodal content integrated
    """
    other_info = {}
    if mm_inputs_list is None:
        return None

    # 1. Calculate the multimodal data which exists in input_ids, with the help of pad_values
    # we assume that multimodal data are represented with its pad_values in input_ids
    item_flatten_list = []
    indices_flatten_list = []
    for indice, mm_inputs in zip(mm_input_indices, mm_inputs_list):
        item_flatten_list += [item for item in mm_inputs.mm_items if item is not None]
        indices_flatten_list += [indice] * len(mm_inputs.mm_items)

    # deepstack_embeddings: per-modality
    modalities, embeddings, masks, deepstack_embeddings = [], [], [], []

    # 2. Get multimodal embedding separately
    # Try get mm embedding if any
    for modality in Modality.all():
        items = [
            item for item in item_flatten_list if item.is_modality(modality=modality)
        ]
        indices = [
            indices_flatten_list[i]
            for i, item in enumerate(item_flatten_list)
            if item.is_modality(modality=modality)
        ]
        embedder = (
            None
            if data_embedding_func_mapping is None
            else data_embedding_func_mapping.get(modality, None)
        )
        if embedder is None:
            # "image", "video", etc
            modality_id = modality.name.lower()
            embedder = getattr(multimodal_model, f"get_{modality_id}_feature", None)
        if len(items) != 0:
            assert embedder is not None, f"no embedding method found for {modality}"
            placeholder_tensor = torch.as_tensor(
                [item.pad_value for item in items],
                device=input_ids.device,
            )
            # calculate per request items length offset
            items_size = torch.zeros(len(mm_inputs_list) + 1, dtype=int)
            items_offsets = []
            for i, mm_inputs in enumerate(mm_inputs_list):
                mm_items = [
                    item
                    for item in mm_inputs.mm_items
                    if item.is_modality(modality=modality)
                ]
                items_size[i + 1] = len(mm_items)
                items_offsets.append(
                    flatten_nested_list([item.offsets for item in mm_items])
                )
            items_size = torch.cumsum(items_size, dim=0).tolist()

            embedding, mask, input_ids = get_embedding_and_mask(
                data_embedding_func=embedder,
                embedding_items=items,
                placeholder_tensor=placeholder_tensor,
                input_ids=input_ids,
                items_size=items_size,
                prefix_length=extend_prefix_lens,
                extend_length=extend_seq_lens,
                items_offset_list=items_offsets,
                forward_batch=forward_batch,
                indices=indices,
            )

            if use_deepstack.get(modality, None) and embedding is not None:
                embedding, deepstack_embedding = (
                    multimodal_model.separate_deepstack_embeds(embedding)
                )
                deepstack_embeddings += [deepstack_embedding]
            modalities += [modality]
            embeddings += [embedding]
            masks += [mask]

    # 3. Get input embeddings
    # Important: clamp after getting original multimodal regions
    # Clamp input ids. This is because the input_ids for the multimodal tokens are
    # filled with the hash values of the multimodal for the prefix matching in the radix attention.
    # There values are useless because their embeddings will be replaced by vision embeddings anyway.
    input_ids.clamp_(min=0, max=vocab_size - 1)
    input_embeds = input_embedding(input_ids)

    # 4. scatter embeddings into input embedding
    for i, modality, embedding, mask in zip(
        range(len(embeddings)), modalities, embeddings, masks
    ):
        if embedding is None or mask is None:
            continue
        # in-place update
        indices = torch.where(mask.squeeze(dim=-1))[0]
        input_embeds[indices] = embedding.to(input_embeds.device, input_embeds.dtype)

    return input_embeds, other_info
