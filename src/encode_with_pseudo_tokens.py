from typing import Union

import clip
import clip.model
import open_clip.transformer
import timm
import torch
import torch.nn as nn
import transformers
from open_clip.transformer import _expand_token, text_global_pool

from src.SLIP.models import SLIP

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def get_encode_with_pseudo_tokens(clip_model):
    if isinstance(clip_model, clip.model.CLIP) or isinstance(clip_model, SLIP):
        return encode_with_pseudo_tokens_openai
    elif isinstance(clip_model, open_clip.model.CLIP):
        if isinstance(clip_model.transformer, open_clip.transformer.Transformer):
            return encode_with_pseudo_tokens_openclip
        else:
            raise NotImplementedError(
                f"When using OpenCLIP, the transformer must be of type open_clip.transformer.Transformer")
    elif isinstance(clip_model, open_clip.model.CustomTextCLIP):
        if isinstance(clip_model.text, open_clip.model.TextTransformer):
            return encode_with_pseudo_tokens_customtextclip
        else:
            raise NotImplementedError(
                f"When using CustomTextCLIP, the text model must be of type open_clip.model.TextTransformer")
    else:
        raise NotImplementedError(f"Clip model {clip_model} not implemented")


def get_encode_image_with_pseudo_tokens(clip_model):
    if isinstance(clip_model, clip.model.CLIP):
        return encode_image_with_pseudo_tokens_openai
    elif isinstance(clip_model, SLIP):
        return encode_image_with_pseudo_tokens_slip
    elif isinstance(clip_model.visual, open_clip.transformer.VisionTransformer):
        return encode_image_with_pseudo_tokens_openclip
    elif isinstance(clip_model.visual, open_clip.timm_model.TimmModel):
        if isinstance(clip_model.visual.trunk, timm.models.vision_transformer.VisionTransformer):
            return encode_image_with_pseudo_tokens_timmvit
        else:
            raise NotImplementedError(
                f"When using OpenCLIP Timm Model, the trunk must be of type timm.models.vision_transformer.VisionTransformer")
    else:
        raise NotImplementedError(f"Clip model {clip_model} not implemented")


def encode_with_pseudo_tokens_openai(clip_model: clip.model.CLIP, tokenized_text: torch.Tensor,
                                     pseudo_tokens: torch.tensor, num_pseudo_tokens=1):
    special_token_idx = get_special_token_idx(clip_model)
    x = clip_model.token_embedding(tokenized_text).type(clip_model.dtype)  # [batch_size, n_ctx, d_model]
    # 49407
    _, counts = torch.unique((tokenized_text == special_token_idx).nonzero(as_tuple=True)[0], return_counts=True)

    cum_sum = torch.cat((torch.zeros(1, device=pseudo_tokens.device).int(), torch.cumsum(counts, dim=0)[:-1]))
    first_vstar_indexes = (tokenized_text == special_token_idx).nonzero()[cum_sum][:, 1]
    rep_idx = torch.cat([(first_vstar_indexes + n).unsqueeze(0) for n in range(num_pseudo_tokens)])

    if pseudo_tokens.shape[0] == x.shape[0]:
        if len(pseudo_tokens.shape) == 2:
            pseudo_tokens = pseudo_tokens.unsqueeze(1)
        x[torch.arange(x.shape[0]).repeat_interleave(
            num_pseudo_tokens).reshape(x.shape[0], num_pseudo_tokens), rep_idx.T] = pseudo_tokens.to(x.dtype)
    else:
        first_vstar_indexes = (tokenized_text == special_token_idx).nonzero()[
                                  torch.arange(0, x.shape[0] * num_pseudo_tokens, num_pseudo_tokens)][:, 1]
        rep_idx = torch.cat([(first_vstar_indexes + n).unsqueeze(0) for n in range(num_pseudo_tokens)])
        x[torch.arange(x.shape[0]).repeat_interleave(num_pseudo_tokens).reshape(
            x.shape[0], num_pseudo_tokens), rep_idx.T] = pseudo_tokens.repeat(x.shape[0], 1, 1).to(x.dtype)

    x = x + clip_model.positional_embedding.type(clip_model.dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = clip_model.ln_final(x).type(clip_model.dtype)

    # x.shape = [batch_size, n_ctx, transformer.width]
    # take features from the eot embedding (eot_token is the highest number in each sequence)
    x = x[torch.arange(x.shape[0]), tokenized_text.argmax(dim=-1)] @ clip_model.text_projection
    return x


def encode_with_pseudo_tokens_openclip(clip_model: open_clip.model.CLIP, tokenized_text: torch.Tensor,
                                       pseudo_tokens: torch.tensor, num_pseudo_tokens=1):
    special_token_idx = get_special_token_idx(clip_model)

    cast_dtype = clip_model.transformer.get_cast_dtype()

    x = clip_model.token_embedding(tokenized_text).type(cast_dtype)  # [batch_size, n_ctx, d_model]
    # 49407

    _, counts = torch.unique((tokenized_text == special_token_idx).nonzero(as_tuple=True)[0], return_counts=True)
    cum_sum = torch.cat((torch.zeros(1, device=pseudo_tokens.device).int(), torch.cumsum(counts, dim=0)[:-1]))
    first_vstar_indexes = (tokenized_text == special_token_idx).nonzero()[cum_sum][:, 1]
    rep_idx = torch.cat([(first_vstar_indexes + n).unsqueeze(0) for n in range(num_pseudo_tokens)])

    if pseudo_tokens.shape[0] == x.shape[0]:
        if len(pseudo_tokens.shape) == 2:
            pseudo_tokens = pseudo_tokens.unsqueeze(1)
        x[torch.arange(x.shape[0]).repeat_interleave(
            num_pseudo_tokens).reshape(x.shape[0], num_pseudo_tokens), rep_idx.T] = pseudo_tokens.to(x.dtype)
    else:
        first_vstar_indexes = (tokenized_text == special_token_idx).nonzero()[
                                  torch.arange(0, x.shape[0] * num_pseudo_tokens, num_pseudo_tokens)][:, 1]
        rep_idx = torch.cat([(first_vstar_indexes + n).unsqueeze(0) for n in range(num_pseudo_tokens)])
        x[torch.arange(x.shape[0]).repeat_interleave(num_pseudo_tokens).reshape(
            x.shape[0], num_pseudo_tokens), rep_idx.T] = pseudo_tokens.repeat(x.shape[0], 1, 1).to(x.dtype)

    x = x + clip_model.positional_embedding.to(cast_dtype)
    x = clip_model.transformer(x, attn_mask=clip_model.attn_mask)
    x = clip_model.ln_final(x)  # [batch_size, n_ctx, transformer.width]
    x, _ = text_global_pool(x, tokenized_text, clip_model.text_pool_type)
    if clip_model.text_projection is not None:
        if isinstance(clip_model.text_projection, nn.Linear):
            x = clip_model.text_projection(x)
        else:
            x = x @ clip_model.text_projection

    return x


def encode_with_pseudo_tokens_customtextclip(clip_model: open_clip.model.CustomTextCLIP, tokenized_text: torch.Tensor,
                                             pseudo_tokens: torch.tensor, num_pseudo_tokens=1):
    special_token_idx = get_special_token_idx(clip_model)

    cast_dtype = clip_model.text.transformer.get_cast_dtype()
    seq_len = tokenized_text.shape[1]

    x = clip_model.text.token_embedding(tokenized_text).to(cast_dtype)  # [batch_size, n_ctx, d_model]
    attn_mask = clip_model.text.attn_mask
    if clip_model.text.cls_emb is not None:
        seq_len += 1
        x = torch.cat([x, _expand_token(clip_model.text.cls_emb, x.shape[0])], dim=1)
        cls_mask = clip_model.text.build_cls_mask(tokenized_text, cast_dtype)
        if attn_mask is not None:
            attn_mask = attn_mask[None, :seq_len, :seq_len] + cls_mask[:, :seq_len, :seq_len]

    _, counts = torch.unique((tokenized_text == special_token_idx).nonzero(as_tuple=True)[0], return_counts=True)
    cum_sum = torch.cat((torch.zeros(1, device=pseudo_tokens.device).int(), torch.cumsum(counts, dim=0)[:-1]))
    first_vstar_indexes = (tokenized_text == special_token_idx).nonzero()[cum_sum][:, 1]
    rep_idx = torch.cat([(first_vstar_indexes + n).unsqueeze(0) for n in range(num_pseudo_tokens)])

    if pseudo_tokens.shape[0] == x.shape[0]:
        if len(pseudo_tokens.shape) == 2:
            pseudo_tokens = pseudo_tokens.unsqueeze(1)
        x[torch.arange(x.shape[0]).repeat_interleave(
            num_pseudo_tokens).reshape(x.shape[0], num_pseudo_tokens), rep_idx.T] = pseudo_tokens.to(x.dtype)
    else:
        first_vstar_indexes = (tokenized_text == special_token_idx).nonzero()[
                                  torch.arange(0, x.shape[0] * num_pseudo_tokens, num_pseudo_tokens)][:, 1]
        rep_idx = torch.cat([(first_vstar_indexes + n).unsqueeze(0) for n in range(num_pseudo_tokens)])
        x[torch.arange(x.shape[0]).repeat_interleave(num_pseudo_tokens).reshape(
            x.shape[0], num_pseudo_tokens), rep_idx.T] = pseudo_tokens.repeat(x.shape[0], 1, 1).to(x.dtype)

    x = x + clip_model.text.positional_embedding[:seq_len].to(cast_dtype)
    x = clip_model.text.transformer(x, attn_mask=attn_mask)

    # x.shape = [batch_size, n_ctx, transformer.width]
    if clip_model.text.cls_emb is not None:
        # presence of appended cls embed (CoCa) overrides pool_type, always take last token
        pooled, tokens = text_global_pool(x, pool_type='last')
        pooled = clip_model.text.ln_final(pooled)  # final LN applied after pooling in this case
    else:
        x = clip_model.text.ln_final(x)
        pooled, tokens = text_global_pool(x, tokenized_text, pool_type=clip_model.text.pool_type)

    if clip_model.text.text_projection is not None:
        if isinstance(clip_model.text.text_projection, nn.Linear):
            pooled = clip_model.text.text_projection(pooled)
        else:
            pooled = pooled @ clip_model.text.text_projection

    return pooled


def encode_image_with_pseudo_tokens_openai(clip_model: clip.model.CLIP, pseudo_tokens: torch.Tensor,
                                           num_pseudo_tokens=1):
    # x = clip_model.conv1(x)  # shape = [*, width, grid, grid]
    # x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
    # x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

    bs, npt, emb_dim = pseudo_tokens.shape
    assert npt == num_pseudo_tokens
    num_tokens = clip_model.visual.positional_embedding.shape[0] - 1  # subtract 1 for the CLIP class token

    # interpolate pseudo tokens to match the number of tokens
    pseudo_tokens_dim = pseudo_tokens.shape[-1]
    pseudo_tokens = pseudo_tokens.unsqueeze(2).transpose(1, 2)
    pseudo_tokens = torch.nn.functional.interpolate(pseudo_tokens, size=(num_tokens, pseudo_tokens_dim),
                                                    mode='nearest')
    pseudo_tokens = pseudo_tokens.squeeze(1)

    x = torch.cat(
        [clip_model.visual.class_embedding.to(
            pseudo_tokens.dtype) + torch.zeros(bs, 1, emb_dim, dtype=pseudo_tokens.dtype,
                                               device=pseudo_tokens.device), pseudo_tokens],
        dim=1)  # shape = [*, grid ** 2 + 1, width]
    x = x + clip_model.visual.positional_embedding.to(x.dtype)
    x = clip_model.visual.ln_pre(x)

    x = x.permute(1, 0, 2)  # NLD -> LND
    x = clip_model.visual.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD

    x = clip_model.visual.ln_post(x[:, 0, :])

    if clip_model.visual.proj is not None:
        x = x @ clip_model.visual.proj

    return x


def encode_image_with_pseudo_tokens_openclip(clip_model: Union[open_clip.model.CLIP, open_clip.model.CustomTextCLIP],
                                             pseudo_tokens: torch.Tensor, num_pseudo_tokens=1):
    # x = clip_model.conv1(x)  # shape = [*, width, grid, grid]
    # x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
    # x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

    bs, npt, emb_dim = pseudo_tokens.shape
    assert npt == num_pseudo_tokens
    num_tokens = clip_model.visual.positional_embedding.shape[0] - 1  # subtract 1 for the CLIP class token

    # interpolate pseudo tokens to match the number of tokens
    pseudo_tokens_dim = pseudo_tokens.shape[-1]
    pseudo_tokens = pseudo_tokens.unsqueeze(2).transpose(1, 2)
    pseudo_tokens = torch.nn.functional.interpolate(pseudo_tokens, size=(num_tokens, pseudo_tokens_dim),
                                                    mode='nearest')
    pseudo_tokens = pseudo_tokens.squeeze(1)

    x = torch.cat(
        [clip_model.visual.class_embedding.to(
            pseudo_tokens.dtype) + torch.zeros(bs, 1, emb_dim, dtype=pseudo_tokens.dtype,
                                               device=pseudo_tokens.device), pseudo_tokens],
        dim=1)  # shape = [*, grid ** 2 + 1, width]
    x = x + clip_model.visual.positional_embedding.to(x.dtype)

    x = clip_model.visual.patch_dropout(x)
    x = clip_model.visual.ln_pre(x)
    x = clip_model.visual.transformer(x)

    if clip_model.visual.attn_pool is not None:
        if clip_model.visual.attn_pool_contrastive is not None:
            x = clip_model.visual.ln_post(x)
            tokens = clip_model.visual.attn_pool(x)
            if clip_model.visual.attn_pool_type == 'parallel':
                pooled = clip_model.visual.attn_pool_contrastive(x)
            else:
                assert clip_model.visual.attn_pool_type == 'cascade'
                pooled = clip_model.visual.attn_pool_contrastive(tokens)
        else:
            x = clip_model.visual.attn_pool(x)
            x = clip_model.visual.ln_post(x)
            pooled, tokens = clip_model.visual._global_pool(x)
    elif clip_model.visual.final_ln_after_pool:
        pooled, tokens = clip_model.visual._global_pool(x)
        pooled = clip_model.visual.ln_post(pooled)
    else:
        x = clip_model.visual.ln_post(x)
        pooled, tokens = clip_model.visual._global_pool(x)

    if clip_model.visual.proj is not None:
        pooled = pooled @ clip_model.visual.proj

    return pooled


def encode_image_with_pseudo_tokens_timmvit(clip_model: Union[open_clip.model.CLIP, open_clip.model.CustomTextCLIP],
                                            pseudo_tokens: torch.Tensor, num_pseudo_tokens=1):
    bs, npt, emb_dim = pseudo_tokens.shape
    assert npt == num_pseudo_tokens
    num_tokens = clip_model.visual.trunk.patch_embed.num_tokens

    # interpolate pseudo tokens to match the number of tokens
    pseudo_tokens_dim = pseudo_tokens.shape[-1]
    pseudo_tokens = pseudo_tokens.unsqueeze(2).transpose(1, 2)
    pseudo_tokens = torch.nn.functional.interpolate(pseudo_tokens, size=(num_tokens, pseudo_tokens_dim),
                                                    mode='nearest')
    pseudo_tokens = pseudo_tokens.squeeze(1)

    x = clip_model.visual.trunk._pos_embed(pseudo_tokens)
    x = clip_model.visual.trunk.patch_drop(x)
    x = clip_model.visual.trunk.norm_pre(x)

    x = clip_model.visual.trunk.blocks(x)
    x = clip_model.visual.trunk.norm(x)

    x = clip_model.visual.trunk.forward_head(x)

    x = clip_model.visual.head(x)
    return x


def encode_image_with_pseudo_tokens_slip(clip_model: SLIP, pseudo_tokens: torch.Tensor,
                                         num_pseudo_tokens=1):
    bs, npt, emb_dim = pseudo_tokens.shape
    assert npt == num_pseudo_tokens
    num_tokens = clip_model.visual.patch_embed.num_tokens

    # interpolate pseudo tokens to match the number of tokens
    pseudo_tokens_dim = pseudo_tokens.shape[-1]
    pseudo_tokens = pseudo_tokens.unsqueeze(2).transpose(1, 2)
    pseudo_tokens = torch.nn.functional.interpolate(pseudo_tokens, size=(num_tokens, pseudo_tokens_dim),
                                                    mode='nearest')
    pseudo_tokens = pseudo_tokens.squeeze(1)

    x = clip_model.visual._pos_embed(pseudo_tokens)
    x = clip_model.visual.patch_drop(x)
    x = clip_model.visual.norm_pre(x)
    x = clip_model.visual.blocks(x)
    x = clip_model.visual.norm(x)

    x = clip_model.visual.forward_head(x)

    x = x @ clip_model.image_projection
    return x


def get_special_token_idx(clip_model, special_character="£") -> int:
    if isinstance(clip_model, clip.model.CLIP):
        return clip_model.tokenizer(special_character)[0][1].item()
    if isinstance(clip_model, SLIP):
        return clip_model.tokenizer(special_character)[0][1].item()
    elif isinstance(clip_model, open_clip.model.CLIP):
        if isinstance(clip_model.tokenizer, open_clip.tokenizer.SimpleTokenizer):
            return clip_model.tokenizer(special_character)[0][1].item()
    elif isinstance(clip_model, open_clip.model.CustomTextCLIP):
        if isinstance(clip_model.tokenizer.tokenizer,
                      transformers.models.t5.tokenization_t5_fast.T5TokenizerFast):  # SigLIP
            return clip_model.tokenizer(special_character)[0][0].item()

    raise NotImplementedError(
        f"Could not find special token index for model {clip_model} and tokenizer {clip_model.tokenizer}")
