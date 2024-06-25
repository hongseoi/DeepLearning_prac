"""
 * Copyright (c) 2023, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
 * Based on huggingface code base
 * https://github.com/huggingface/transformers/blob/v4.15.0/src/transformers/models/bert
"""

import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch
from torch import Tensor, device, dtype, nn
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.file_utils import (
    ModelOutput,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.utils import logging
from transformers.models.bert.configuration_bert import BertConfig

logger = logging.get_logger(__name__)

# ? BERT 모델에서 입력 임베딩 생성: 단어 임베딩과 위치 임베딩, 쿼리 임베딩 결합해 입력 시퀸스 표현
class BertEmbeddings(nn.Module):
    """Construct the embeddings from word and position embeddings."""

    def __init__(self, config):
        super().__init__()
        
        # 단어 임베딩
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        
        # 위치 임베딩
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        # 레이저 정규화 정의
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 레이어 드롭아웃 정의
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer( # position_ids를 생성하고 버퍼로 등록
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1))
        )
        
        # position embedding 타입 absolute로 설정
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )

        self.config = config

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        query_embeds=None,
        past_key_values_length=0,
    ):
        # input_ids 주어지면 입력 시퀸스의 길이 계산
        if input_ids is not None:
            seq_length = input_ids.size()[1]
        else:
            seq_length = 0

        # position_ids가 없으면 생성
        if position_ids is None:
            position_ids = self.position_ids[
                :, past_key_values_length : seq_length + past_key_values_length
            ].clone()

        # inpur_ids가 주어지면 word_embedding 계산
        if input_ids is not None:
            embeddings = self.word_embeddings(input_ids)
            
            # position_embedding type이 absolute인 경우: word embedding + position_embedding
            if self.position_embedding_type == "absolute":
                position_embeddings = self.position_embeddings(position_ids)
                embeddings = embeddings + position_embeddings 

            # query_embedding이 주어지는 경우: word embedding + position_embedding + query_embedding
            if query_embeds is not None:
                embeddings = torch.cat((query_embeds, embeddings), dim=1)
        
        # input_ids가 없는 경우: qeury_embedding
        else:
            embeddings = query_embeds

        # 임베딩 정규화 및 드롭아웃
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

# ? BERT의 Self attention 구현
class BertSelfAttention(nn.Module):
    def __init__(self, config, is_cross_attention):
        super().__init__()
        self.config = config # config 매개변수를 인스턴스 변수로 저장
        
        # size 안 맞을 경우 에러
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        # attention head 수, 크기 설정
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # linear 레이어를 query 변수에 설정
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        
        # cross attention인 경우, key와 value 생성
        if is_cross_attention:
            self.key = nn.Linear(config.encoder_width, self.all_head_size)
            self.value = nn.Linear(config.encoder_width, self.all_head_size)
        
        # cross attention 아닌 경우 key와 value 생성
        else:
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        # 드롭아웃 및 위치 임베딩 정의
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )
        
        # ! relative 위치 임베딩 사용하는 경우 
        if (
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(
                2 * config.max_position_embeddings - 1, self.attention_head_size
            )
        self.save_attention = False

    # attention gradient 저장
    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    # 저장된 attention gradient 불러오기
    def get_attn_gradients(self):
        return self.attn_gradients

    # attention map 저장
    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    # attention map 불러오기
    def get_attention_map(self):
        return self.attention_map

    # attention score 계산을 위한 x의 shape 변환
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        # x 크기 변경
        x = x.view(*new_x_shape)
        # x의 차원 변경
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        # cross attention인 경우 key value 설정
        if is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        
        # 과거 key-value가 있는 경우
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            
            # 과거 key-value와 현재 key-value를 합침
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
            
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        # hidden state에 query(선형변환) 적용
        mixed_query_layer = self.query(hidden_states)

        # shape 변경
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # 현재 key-value를 past_key_value에 저장
        past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # position embedding type이 relative key 인 경우
        # ! 상대적 위치 임베딩???
        if (
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            # 입력 시퀸스 길이
            seq_length = hidden_states.size()[1]
            
            # 왼쪽 위치 인덱스. sequence 길이 만큼의 1차원 텐서 생성 후 (seq_length, 1)로 크기 변환
            position_ids_l = torch.arange(
                seq_length, dtype=torch.long, device=hidden_states.device
            ).view(-1, 1)
            
            # 오른쪽 위치 인텍스. 시퀸스 길이만큼의 1차원 텐서 생성 후 (1, seq_length)로 크기 변환
            position_ids_r = torch.arange(
                seq_length, dtype=torch.long, device=hidden_states.device
            ).view(1, -1)
            
            # 두 position index간 차이 계산해 각 위치쌍의 상대적 거리 나타내는 텐서 생성
            distance = position_ids_l - position_ids_r
            
            # ! position embedding 계산: 상대적 거리를 정수 값으로 변환하여 임베딩 레이어에 입력합니다. 이는 거리 값이 음수가 되지 않도록 조정하는 과정
            positional_embedding = self.distance_embedding(
                distance + self.max_position_embeddings - 1
            )
            
            # position embedding과 query_layer의 dtype 맞춤 
            positional_embedding = positional_embedding.to(
                dtype=query_layer.dtype
            )  # fp16 compatibility

            # position_embedding type이 relative key인 경우
            if self.position_embedding_type == "relative_key":
                # einsum 표기법을 사용해 query layer와 positional embedding 간 dot product 계산해서 상대적 위치 score
                relative_position_scores = torch.einsum( # ! einsum이 뭐여
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                
                # 기존 attention score + 상대적 위치 score
                attention_scores = attention_scores + relative_position_scores
                
            # position embedding type이 relative_key_query인 경우    
            elif self.position_embedding_type == "relative_key_query":
                
                # einsum 표기법을 사용해 query layer와 positional embedding 간 dot product 계산해서 key, query의 상대적 위치 score                
                relative_position_scores_query = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                relative_position_scores_key = torch.einsum(
                    "bhrd,lrd->bhlr", key_layer, positional_embedding
                )
                attention_scores = (
                    attention_scores
                    + relative_position_scores_query
                    + relative_position_scores_key
                )
        # ateention score 계산
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # ! attention mask 있으면 attention mask를 적용하여 attention score 다시 계산
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # cross attention이 활성화되어있고 save_attention이 지정된 경우
        if is_cross_attention and self.save_attention:
            self.save_attention_map(attention_probs) # 계산된 attention probs 저장
            attention_probs.register_hook(self.save_attn_gradients) # attention probs에 대한 gradient를 저장하기 위한 hook 등록

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # 과적합 방지를 위해 어텐션 확률의 일부를 무작위로 0으로 설정
        attention_probs_dropped = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            # 특정 어텐션 헤드 무시
            attention_probs_dropped = attention_probs_dropped * head_mask

        # attention prob와 value를 곱해서 context layer 생성
        context_layer = torch.matmul(attention_probs_dropped, value_layer)

        # context layer의 shape 변경
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )
        
        # ! ??? 왜 context_layer + past_key_value ?
        outputs = outputs + (past_key_value,)
        return outputs

# ? self attention의 출력 부분 처리: dense-dropout-layernorm
class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

# !!!!! BERT의 전체 Attention 모듈 구현
class BertAttention(nn.Module):
    def __init__(self, config, is_cross_attention=False):
        super().__init__()
        self.self = BertSelfAttention(config, is_cross_attention) # self attention
        self.output = BertSelfOutput(config) # hidden state output
        self.pruned_heads = set() # 제거된 attention head 추적

    # 특정 Attention head 제거
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices( # !! 주어진 head 리스트에서 이미 제거된 head를 제외하고 실제로 제거할 head를 식별하여 해당 인덱스 반환
            heads, # 제거할 head의 인덱스 리스트
            self.self.num_attention_heads, # 레이어의 attention head 수
            self.self.attention_head_size, # 각 attention head의 size
            self.pruned_heads, # 이미 제거된 head
        )

        # 해당 head의 qkv 제거 
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1) # 관련 선형 레이어 제거

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = (
            self.self.attention_head_size * self.self.num_attention_heads
        )
        self.pruned_heads = self.pruned_heads.union(heads) # pruned_heads에 제거된 head index 리스트 추가

    def forward(
        self,
        hidden_states,
        attention_mask=None, # 입력 시퀸스의 padding 토큰 제거하기 위해 사용
        head_mask=None, # 특정 attention head 마스킹 위해 사용
        encoder_hidden_states=None, # decoder에서 cross attention 수행할 때 사용
        encoder_attention_mask=None,
        past_key_value=None, # 이전 timestep의 qkv (캐싱을 이용해 속도 높이기 위해 사용)
        output_attentions=False, # attention 출력 반환여부
    ):
        
        # self attention 수행
        self_outputs = self.self( # self: BertSelfAttention
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        
        # linear - dropout - layernorm 수행
        attention_output = self.output(self_outputs[0], hidden_states) # hidden state, input tensor

        outputs = (attention_output,) + self_outputs[
            1:
        ]  # add attentions if we output them
        return outputs

# ? BERT의 FeedForward Network의 첫번째 부분 구현: dense-activation
class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        
        # FFN에서 사용할 act_fn 설정
        if isinstance(config.hidden_act, str): # hidden_act가 str인지 확인
            self.intermediate_act_fn = ACT2FN[config.hidden_act] # 해당 문자열에 해당하는 활성화함수 가져옴. ACTFN은 활성화 함수 이름을 실제 함수로 매핑하는 딕셔너리
        else: # 이미 함수 객체인 경우
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states) # 활성화함수 적용
        return hidden_states

# ? BERT의 FeedForward Network 두번째 부분: dense-dropout-residual+layernrom
class BertOutput(nn.Module): # ! BERFSELFOUTPUT()과 다른 점? 똑같음
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor) # residual connection 및 layernorm
        return hidden_states

# ? BERT 단일 레이어 구현. attention과 피드포워드 신경망 레이어를 결합
class BertLayer(nn.Module):
    def __init__(self, config, layer_num):
        super().__init__()
        self.config = config
        self.chunk_size_feed_forward = config.chunk_size_feed_forward #config에서 불러와서 인스턴스 변수로 저장. ffn에서 입력을 chnnk 단위로 처리할 때 사용
        self.seq_len_dim = 1 # 시퀸스 길이 차원(입력 텐서 (batch_size, sequence_length, hidden_size) 중 sequence length는 1번째라는뜻!)
        self.attention = BertAttention(config) # attention 담당
        self.layer_num = layer_num # 레이어 개수? index?
        
        # add_cross_attention이 True이고 layer_num을 cross attention frequency로 나눈 나머지가 0인 경우 cross attention 수행
        if (
            self.config.add_cross_attention
            and layer_num % self.config.cross_attention_freq == 0
        ):
            self.crossattention = BertAttention(
                config, is_cross_attention=self.config.add_cross_attention
            )
            self.has_cross_attention = True
        
        else:
            self.has_cross_attention = False
        
        # 계산    
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

        self.intermediate_query = BertIntermediate(config)
        self.output_query = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        query_length=0,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        # 과거 key, value 값 가져오기
        self_attn_past_key_value = (
            past_key_value[:2] if past_key_value is not None else None
        )
        
        # self attention 수행
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:-1]

        # 현재 key value 값
        present_key_value = self_attention_outputs[-1]

        if query_length > 0:
            query_attention_output = attention_output[:, :query_length, :]

            # cross attention 수행
            if self.has_cross_attention:
                assert (
                    encoder_hidden_states is not None
                ), "encoder_hidden_states must be given for cross-attention layers"
                cross_attention_outputs = self.crossattention(
                    query_attention_output,
                    attention_mask,
                    head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions=output_attentions,
                )
                query_attention_output = cross_attention_outputs[0]
                outputs = (
                    outputs + cross_attention_outputs[1:-1]
                )  # add cross attentions if we output attention weights

            # query feed forward 수행
            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk_query,
                self.chunk_size_feed_forward,
                self.seq_len_dim,
                query_attention_output,
            )
            
            # query_length보다 긴 경우 잘라내서 수행
            if attention_output.shape[1] > query_length:
                layer_output_text = apply_chunking_to_forward(
                    self.feed_forward_chunk,
                    self.chunk_size_feed_forward,
                    self.seq_len_dim,
                    attention_output[:, query_length:, :],
                )
                layer_output = torch.cat([layer_output, layer_output_text], dim=1)
        
        # cross attention 안하는 경우
        else:
            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk,
                self.chunk_size_feed_forward,
                self.seq_len_dim,
                attention_output,
            )
        outputs = (layer_output,) + outputs

        outputs = outputs + (present_key_value,)

        return outputs

    # Feed forward 연산
    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

    # query feed forward 연산
    def feed_forward_chunk_query(self, attention_output):
        intermediate_output = self.intermediate_query(attention_output)
        layer_output = self.output_query(intermediate_output, attention_output)
        return layer_output

# ? 여러가지 BERT 레이어를 결합하여 인코더를 구성
class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # BERTLAYER 객체를 num_hidden_layers 만큼 생성해서 ModuleList에 저장
        self.layer = nn.ModuleList(
            [BertLayer(config, i) for i in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        query_length=0,
    ):
        # 각 변수 초기화
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = (
            () if output_attentions and self.config.add_cross_attention else None
        )

        # 디코더 캐시 저장 변수 초기화
        next_decoder_cache = () if use_cache else None

        # 각 레이어에 대해 순차적 연산 수행
        for i in range(self.config.num_hidden_layers):
            layer_module = self.layer[i]    # 현재 레이어
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 현재 레이어에서의 head_mask와 past kv 설정
            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # gradient_checkpointing가 False이고 training 중인 경우
            if getattr(self.config, "gradient_checkpointing", False) and self.training:
                    
                if use_cache:
                    logger.warn(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                # torch.utils.checkpoint와 함께 사용될 커스텀 forward 함수 생성
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(
                            *inputs, past_key_value, output_attentions, query_length
                        )

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            # gradient_checkpointing가 True 또는 training 중이 아닌 경우
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    query_length,
                )

            # 현재 레이어 출력인 hidden state 업데이터
            hidden_states = layer_outputs[0]
            
            # 캐시 저장
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
                
            # self attention과 cross attention 가중치 저장
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # hidden state 저장
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # dictionary가 아니라 tuple로 반환
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        
        # return_dict==True인 경우
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

# ? 인코더의 출력을 POOLING하여 고정된 크기의 벡터로 변환 (일반적으로 문장 분류 작업에 사용)
class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # BERT encoder의 출력 hidden state의 첫번째 토큰의 hidden state 출력
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0] 
        pooled_output = self.dense(first_token_tensor) 
        pooled_output = self.activation(pooled_output)
        return pooled_output

# BERT의 언어 모델링 헤드(MLM) 구현. 다음단어 예측에 사용
class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    # hidden state에 변환 적용
    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

# BERT의 언어 모델링 헤드(MLM) 구현. 다음단어 예측에 사용
class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

# pretrained BERT 모델 다루는 클래스
class BertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BertConfig
    base_model_prefix = "bert"
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
# =================================================================
# BERT의 전체 모델 구현
class BertModel(BertPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
    input to the forward pass.
    
    동일 모델이 encoder로도, decoder로도 사용 가능
    encoder: self attention만 사용하는 경우
    decoder: self attention + cross attention 사용하는 경우
    
    """

    def __init__(self, config, add_pooling_layer=False):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)

        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.init_weights()

    # word embedding 반환
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    # word embedding 설정
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    # ??? 특정 attention head는 왜 제거함?
    # 특정 attention head 제거
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # ! attention mask 생성(이부분이 중요한것으로 예상)
    def get_extended_attention_mask(
        self,
        attention_mask: Tensor,
        input_shape: Tuple[int], #모델의 입력 데이터 형태
        device: device, # 모델에 입력되는 데이터 디바이스
        is_decoder: bool,
        has_query: bool = False, # 쿼리 있는지 여부 나타내는 flag
    ) -> Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.

        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        
        ****************************************************************
        broadcastable attention and causal masks를 만들어서 미래의 토큰과 마스킹된 토큰이 무시되도록 합니다.
        - broadcastable attention: 마스크를 여러 attention head에 적용하기 쉽도록 만든 형태. 즉 동일 mask를 여러 attention에 적용할 수 있도록 텐서 차원 확장
        - casual mask: 주로 decoder에서 사용되며 모델이 미래 토큰을 보지못하게 함
        인자:
            attention_mask (:obj:`torch.Tensor`):
                주의해야 할 토큰을 나타내는 1과 무시해야 할 토큰을 나타내는 0으로 구성된 마스크.
                어떤 토큰을 집중할지, 무시할지 결정
            input_shape (:obj:`Tuple[int]`):
                모델에 입력되는 데이터의 형태.
            device: (:obj:`torch.device`):
                모델에 입력되는 데이터의 디바이스.

        return:
            :obj:`torch.Tensor` 확장된 어텐션 마스크로, :obj:`attention_mask.dtype`과 같은 데이터 타입을 가집니다.

        """
        # ? We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ? ourselves in which case we just need to make it broadcastable to all heads.
        # [batch_size, from_seq_length, to_seq_length] 형태로 3차원
        if attention_mask.dim() == 3:
            # broadcasting을 이용해 attention mask shape를 [batch_size, 1, from_seq_length, to_seq_length] 형태로 확장하여 여러 어텐션 헤드에 브로드캐스트할 수 있게 만듦
            extended_attention_mask = attention_mask[:, None, :, :]
        
        # attention mask가 2차원: [batch_size, seq_length]
        elif attention_mask.dim() == 2:
            # ?Provided a padding mask of dimensions [batch_size, seq_length]
            # ? - if the model is a decoder, apply a causal mask in addition to the padding mask
            # ? - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            
            # decoder인 경우: casual mask + padding mask
            if is_decoder:
                batch_size, seq_length = input_shape

                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = (
                    seq_ids[None, None, :].repeat(batch_size, seq_length, 1)
                    <= seq_ids[None, :, None]
                )

                # add a prefix ones mask to the causal mask
                # causal and attention masks must have same type with pytorch version < 1.3
                causal_mask = causal_mask.to(attention_mask.dtype)

                if causal_mask.shape[1] < attention_mask.shape[1]:
                    prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
                    if has_query:  # UniLM style attention mask
                        causal_mask = torch.cat(
                            [
                                torch.zeros(
                                    (batch_size, prefix_seq_len, seq_length),
                                    device=device,
                                    dtype=causal_mask.dtype,
                                ),
                                causal_mask,
                            ],
                            axis=1,
                        )
                    causal_mask = torch.cat(
                        [
                            torch.ones(
                                (batch_size, causal_mask.shape[1], prefix_seq_len),
                                device=device,
                                dtype=causal_mask.dtype,
                            ),
                            causal_mask,
                        ],
                        axis=-1,
                    )
                extended_attention_mask = (
                    causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
                )
            # encoder 인 경우: 마스크를 [batch_size, 1, 1, seq_length] 형태로 확장하여 여러 어텐션 헤드에 브로드캐스트할 수 있게 함
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            # attention mask가 1차원 > 오류임
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        
        # attention_mask가 주의해야 할 위치에 대해 1.0, 마스킹된 위치에 대해 0.0
        # 아래 연산은 주의해야 할 위치에 대해 0.0을, 마스킹된 위치에 대해 -10000.0을 가지는 텐서를 생성합니다.
        # 이 값을 소프트맥스 이전의 raw score에 더하기 때문에, 이는 실질적으로 마스킹된 위치를 완전히 제거하는 것과 동일한 효과를 줍니다.

        extended_attention_mask = extended_attention_mask.to(
            dtype=self.dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0 #주의해야할 위치: (1.0-1.0)*-10000, 마스킹된 위치: (1.0-0.0)*-100000
        # 주의할 위치는 0, 마스킹 위치는 -100000으로 설정하여 SOFTMAX 적용시 확률을 거의 0으로 만든다
        return extended_attention_mask

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        query_embeds=None, # decoder에서 사용되는 경우 입력 시퀸스 외부의 추가정보, 즉 이미지에 대한 정보 제공
        encoder_hidden_states=None, # encoder의 마지막 layer의 hidden state. 모델이 decoder로 사용되는 경우 cross attention에 사용
        encoder_attention_mask=None, # [1,0]으로 표현되는 attention mask. 모델이 decoder로 사용되는 경우 cross attention에 사용
        past_key_values=None, # attention에서 미리 계산된 kv의 hidden state
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        is_decoder=False,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """
        
        # config를 이용한 파라미터 기본값 설정
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # use_cache = use_cache if use_cache is not None else self.config.use_cache

        # input_ids가 none인 경우 query_embeds 반드시 필요
        if input_ids is None:
            assert (
                query_embeds is not None
            ), "You have to specify query_embeds when input_ids is None"

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] - self.config.query_length # pask_key_value의 길이 - qeury_length
            if past_key_values is not None
            else 0 # past_key_value 없으면 0
        )

        # query 길이 계산
        query_length = query_embeds.shape[1] if query_embeds is not None else 0

        # embedding 생성
        embedding_output = self.embeddings(
            input_ids=input_ids, # 입력 토큰 아이디
            position_ids=position_ids, # 입력 토큰 위치 id
            query_embeds=query_embeds, # ! learned query
            past_key_values_length=past_key_values_length,#이전 kv 길이(현재단계의 위치 인덱스 조정에 사용)
        )

        input_shape = embedding_output.size()[:-1] # embedding output에서 마지막 차원 제외 가져옴. [batch size, seq length]
        batch_size, seq_length = input_shape
        device = embedding_output.device

        if attention_mask is None: # attention mask 없는 경우 모든 위치가 1인 mask 생성
            attention_mask = torch.ones(
                ((batch_size, seq_length + past_key_values_length)), device=device
            )

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if is_decoder: # decoder인 경우
            extended_attention_mask = self.get_extended_attention_mask( # ??? extended mask의 역할 casual mask, padding mask로 attention mask 구성
                attention_mask,
                input_ids.shape,
                device,
                is_decoder,
                has_query=(query_embeds is not None),
            )
        else:
            extended_attention_mask = self.get_extended_attention_mask(
                attention_mask, input_shape, device, is_decoder
            )

        # cross attention 적용시 사용
        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if encoder_hidden_states is not None: # hidden state 존재할 경우 cross attention에 다음 정보 활용
            if type(encoder_hidden_states) == list:
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states[
                    0
                ].size()
            else: # encoder_hidden_state == None인 경우 크기 직접 가져옴
                (
                    encoder_batch_size,
                    encoder_sequence_length,
                    _,
                ) = encoder_hidden_states.size()
                
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)

            # encoder_attention_mask == list이면 각 mask를 invert_attention_mask()를 이용해 변환
            if type(encoder_attention_mask) == list:
                encoder_extended_attention_mask = [
                    self.invert_attention_mask(mask) for mask in encoder_attention_mask
                ]
            
            # encoder_attention_mask == None 이면 ones 이용해 직접 생성 후 변환
            elif encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
                encoder_extended_attention_mask = self.invert_attention_mask(
                    encoder_attention_mask
                )
            # 그외 
            else:
                encoder_extended_attention_mask = self.invert_attention_mask( # attention mask 값 반전
                    encoder_attention_mask
                )
        # encoder_hidden_state ==None인 경우
        else:
            encoder_extended_attention_mask = None

        # ! Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # encoder
        encoder_outputs = self.encoder(
            embedding_output, # input embedding tensor
            attention_mask=extended_attention_mask, #입력 시퀸스의 패딩 토큰 무시
            head_mask=head_mask,# 특정 attention head 무시
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask, # 디코더에서 cross attention 수행시 인코더의 padding token 무시 위해 사용
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            query_length=query_length,
        )
        
        # encoder의 첫번째 요소, 즉 encoder의 마지막 hidden state를 저장
        sequence_output = encoder_outputs[0]
        
        # pooling
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

# 언어모델링을 위한 BERT 모델 구현
class BertLMHeadModel(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        query_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        past_key_values=None,
        use_cache=True,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        return_logits=False,
        is_decoder=True,
        reduction="mean",
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are
            ignored (masked), the loss is only computed for the tokens with labels n ``[0, ..., config.vocab_size]``
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        Returns:
        Example::
            >>> from transformers import BertTokenizer, BertLMHeadModel, BertConfig
            >>> import torch
            >>> tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            >>> config = BertConfig.from_pretrained("bert-base-cased")
            >>> model = BertLMHeadModel.from_pretrained('bert-base-cased', config=config)
            >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
            >>> outputs = model(**inputs)
            >>> prediction_logits = outputs.logits
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        # labels가 주어지는 경우, 즉 학습모드인경우 캐시 저장 안함
        if labels is not None:
            use_cache = False
        # past kv가 제공되면 이전 시퀸스의 정보 사용하므로 query embeds 없어도 ok
        if past_key_values is not None:
            query_embeds = None

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            query_embeds=query_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            is_decoder=is_decoder,
        )

        sequence_output = outputs[0]
        
        # query_embeds가 제공되었다면, sequence_output의 첫번째 query_embeds만 남겨서 실제 시퀸스의 출력만 남김
        if query_embeds is not None:
            sequence_output = outputs[0][:, query_embeds.shape[1] :, :]

        # 예측 수행
        prediction_scores = self.cls(sequence_output)

        if return_logits:
            return prediction_scores[:, :-1, :].contiguous()

        lm_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            # 예측 점수와 label을 한 위치씩 이동
            shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss(reduction=reduction, label_smoothing=0.1)
            # 교차 엔트로피 이용해 loss 계산 및 업데이트
            lm_loss = loss_fct(
                shifted_prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1),
            )
            if reduction == "none":
                lm_loss = lm_loss.view(prediction_scores.size(0), -1).sum(1)

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=lm_loss,
            logits=prediction_scores,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, query_embeds, past=None, attention_mask=None, **model_kwargs
    ):
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)
        
        # query_embeds와 shape가 동일하고 모든 요소가 1인 새 텐서 생성    
        query_mask = input_ids.new_ones(query_embeds.shape[:-1])
        
        # query mask + attention mask로 최종 attention mask 생성 ??? 이 부분이 task에 따라 다르게 적용되는 부분인 것 같은데 왜 prepare_inputs_for_generation()를 사용한 부분이 없는지
        attention_mask = torch.cat([query_mask, attention_mask], dim=-1)

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:] # past가 제공되면 input_ids에서 마지막 위치의 토큰만 사용

        return {
            "input_ids": input_ids,
            "query_embeds": query_embeds,
            "attention_mask": attention_mask,
            "past_key_values": past,
            "encoder_hidden_states": model_kwargs.get("encoder_hidden_states", None),
            "encoder_attention_mask": model_kwargs.get("encoder_attention_mask", None),
            "is_decoder": True,
        }

    def _reorder_cache(self, past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx) for past_state in layer_past
                ),
            )
        return reordered_past

# ? 마스크드 언어 모델링(MLM)을 위한 BERT 모델을 구현
class BertForMaskedLM(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        query_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        return_logits=False,
        is_decoder=False,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            query_embeds=query_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            is_decoder=is_decoder,
        )

        if query_embeds is not None:
            sequence_output = outputs[0][:, query_embeds.shape[1] :, :]
        prediction_scores = self.cls(sequence_output)

        if return_logits:
            return prediction_scores

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)
            )

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (
                ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            )

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )