import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from parlai.agents.hy_lib.common_modules import FeedForward
from parlai.core.torch_generator_agent import TorchGeneratorModel
from parlai.core.utils import neginf


LAYER_NORM_EPS = 1e-12  # Epsilon for layer norm.

# TorchGeneratorWithDialogEvalAgent
def _normalize(tensor, norm_layer):
    """Broadcast layer norm"""
    size = tensor.size()
    return norm_layer(tensor.view(-1, size[-1])).view(size)

def _build_encoder(opt,reduction_type='mean'):
    return TransformerEncoder(
        n_heads=opt['policy_n_heads'],
        n_layers=opt['policy_n_layers'],
        embedding_size=opt['policy_embedding_size'],
        ffn_size=opt['policy_ffn_size'],
        dropout=opt['dropout'],
        attention_dropout=opt['policy_attention_dropout'],
        relu_dropout=opt['policy_relu_dropout'],
        reduction_type=reduction_type,
        activation=opt['policy_activation'],
    )

def gelu(tensor):
    return 0.5 * tensor * (1.0 + torch.erf(tensor / math.sqrt(2.0)))

class TransformerEncoder(nn.Module):
    """
    Transformer encoder module.

    :param int n_heads: the number of multihead attention heads.
    :param int n_layers: number of transformer layers.
    :param int embedding_size: the embedding sizes. Must be a multiple of n_heads.
    :param int ffn_size: the size of the hidden layer in the FFN
    :param embedding: an embedding matrix for the bottom layer of the transformer.
        If none, one is created for this encoder.
    :param float dropout: Dropout used around embeddings and before layer
        layer normalizations. This is used in Vaswani 2017 and works well on
        large datasets.
    :param float attention_dropout: Dropout performed after the multhead attention
        softmax. This is not used in Vaswani 2017.
    :param float relu_attention: Dropout used after the ReLU in the FFN. Not used
        in Vaswani 2017, but used in Tensor2Tensor.
    :param bool learn_positional_embeddings: If off, sinusoidal embeddings are
        used. If on, position embeddings are learned from scratch.
    :param bool reduction: If true, returns the mean vector for the entire encoding
        sequence.
    :param int n_positions:
        Size of the position embeddings matrix.
    :param activation:
        Type of nonlinear activation. Can be relu or gelu.
    """

    def __init__(
            self,
            n_heads,
            n_layers,
            embedding_size,
            ffn_size,
            dropout=0.0,
            attention_dropout=0.0,
            relu_dropout=0.0,
            reduction_type='mean',
            activation='relu',
    ):
        super(TransformerEncoder, self).__init__()

        self.embedding_size = embedding_size
        self.ffn_size = ffn_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dim = embedding_size
        self.reduction_type = reduction_type
        # this is --dropout, not --relu-dropout or --attention-dropout
        self.dropout = nn.Dropout(p=dropout)
        self.out_dim = embedding_size

        # build the model
        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(TransformerEncoderLayer(
                n_heads, embedding_size, ffn_size,
                attention_dropout=attention_dropout,
                relu_dropout=relu_dropout,
                dropout=dropout,
                activation=activation,
            ))

    def forward(self, input):
        """
            input data is a FloatTensor of shape [batch, seq_len, dim]
            mask is a ByteTensor of shape [batch, seq_len], filled with 1 when
            inside the sequence and 0 outside.
        """
        # --dropout on the embeddings
        tensor = self.dropout(input)
        for i in range(self.n_layers):
            tensor = self.layers[i](tensor)

        if self.reduction_type == 'first':
            return tensor[:, 0, :]
        elif self.reduction_type == 'max':
            return tensor.max(dim=1)[0]
        elif self.reduction_type == 'mean':
            divisor = mask.float().sum(dim=1).unsqueeze(-1).clamp(min=1).type_as(tensor)
            output = tensor.sum(dim=1) / divisor
            return output
        elif self.reduction_type == 'none' or self.reduction_type is None:
            output = tensor
            return output
        else:
            raise ValueError(
                "Can't handle --reduction-type {}".format(self.reduction_type)
            )

class TransformerEncoderLayer(nn.Module):
    def __init__(
            self,
            n_heads,
            embedding_size,
            ffn_size,
            attention_dropout=0.0,
            relu_dropout=0.0,
            dropout=0.0,
            activation='relu',
    ):
        super().__init__()
        self.dim = embedding_size
        self.ffn_dim = ffn_size
        self.activation = activation
        self.attention = MultiHeadAttention(
            n_heads, embedding_size,
            dropout=attention_dropout,  # --attention-dropout
        )
        self.norm1 = nn.LayerNorm(embedding_size, eps=LAYER_NORM_EPS)
        self.ffn = TransformerFFN(embedding_size, ffn_size,
                                  relu_dropout=relu_dropout,
                                  activation=self.activation)
        self.norm2 = nn.LayerNorm(embedding_size, eps=LAYER_NORM_EPS)
        self.dropout = nn.Dropout(p=dropout)
        self.attention_with_states = MultiHeadAttention(
            n_heads, embedding_size,
            dropout=attention_dropout,  # --attention-dropout
        )

    def forward(self, tensor):
        # Multi-head attention
        # Add
        # print('tensor', tensor)
        # print('self.attention(tensor)', self.attention(tensor))
        # print('self.dropout(self.attention(tensor)', self.dropout(self.attention(tensor)))
        tensor = tensor + self.dropout(self.attention(tensor))
        # print('tensor', tensor)
        # Normalization
        tensor = _normalize(tensor, self.norm1)
        # FFN and add
        tensor = tensor + self.dropout(self.ffn(tensor))
        # Normalization
        tensor = _normalize(tensor, self.norm2)

        return tensor

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, dim, dropout=0):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.attn_dropout = nn.Dropout(p=dropout)  # --attention-dropout
        self.out_lin = nn.Linear(dim, dim)
        nn.init.xavier_normal_(self.out_lin.weight)

    def forward(self, query, key=None, value=None):
        batch_size, query_len, dim = query.size()
        n_heads = self.n_heads
        dim_per_head = dim // n_heads
        scale = math.sqrt(dim_per_head)

        # q, k, v are the transformed values
        if key is None and value is None:
            # self attention
            key = value = query
        elif value is None:
            # key and value are the same, but query differs
            # self attention
            value = key

        q = query
        k = key
        v = value

        dot_prod = q.div_(scale).bmm(k.transpose(1, 2))
        attn_weights = F.softmax(dot_prod, dim=-1).type_as(query)
        attn_weights = self.attn_dropout(attn_weights)  # --attention-dropout

        attentioned = attn_weights.bmm(v)
        attentioned = (
            attentioned.type_as(query)
                .view(batch_size, n_heads, query_len, dim_per_head)
                .transpose(1, 2).contiguous()
                .view(batch_size, query_len, dim)
        )
        out = self.out_lin(attentioned)

        return out

class TransformerFFN(nn.Module):
    def __init__(self, dim, dim_hidden, relu_dropout=0, activation='relu'):
        super(TransformerFFN, self).__init__()
        self.relu_dropout = nn.Dropout(p=relu_dropout)
        if activation == 'relu':
            self.nonlinear = F.relu
        elif activation == 'gelu':
            self.nonlinear = gelu
        else:
            raise ValueError(
                "Don't know how to handle --activation {}".format(activation)
            )
        self.lin1 = nn.Linear(dim, dim_hidden)
        self.lin2 = nn.Linear(dim_hidden, dim)
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.xavier_uniform_(self.lin2.weight)
        # TODO: initialize biases to 0

    def forward(self, x):
        x = self.nonlinear(self.lin1(x))
        x = self.relu_dropout(x)  # --relu-dropout
        x = self.lin2(x)
        return x

####################################

class PolicyNet_MLP(nn.Module):

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.policy = FeedForward(state_dim, action_dim, hidden_sizes=(128, 64))

    def forward(self, state):
        action_score = self.policy(state)
        action_prob = F.softmax(action_score, dim=-1)

        return action_prob

class PolicyNet_Transformer(nn.Module):

    def __init__(self, opt, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.history_encoder = _build_encoder(opt, reduction_type=None)
        self.history_ffn = FeedForward(opt['policy_embedding_size'], action_dim, hidden_sizes=(128, 64))
        self.policy = FeedForward(state_dim, action_dim, hidden_sizes=(128, 64, 32))
        self.last_ffn = FeedForward(action_dim*2, action_dim, hidden_sizes=(128, 64))

    def forward(self, state, history_mean_emb):
        # print('PolicyNet_Transformer history_mean_emb', history_mean_emb.size())
        pad_history_mean_emb = F.pad(input=history_mean_emb, pad=(0, 0, 0, 0, 0, 10-history_mean_emb.size()[0]), mode='constant', value=0) #(10,64,300)
        # print('PolicyNet_Transformer pad_history_mean_emb', pad_history_mean_emb.size())
        encoder_states = self.history_encoder(pad_history_mean_emb)
        # print('encoder_states', encoder_states.size()) # (10, 64, 300)
        encoder_states = torch.unsqueeze(torch.mean(torch.sum(encoder_states, 0), 0), 0)
        # print('encoder_states', encoder_states.size()) # (1,300)
        encoder_states = self.history_ffn(encoder_states)
        # print(encoder_states.size()) # (1,5)
        state_score = self.policy(state)
        # print('state_score', state_score.size()) # (1,5)
        states = self.last_ffn(torch.cat((encoder_states, state_score), dim=1))
        # print('states', states.size()) # (1,10)->(1,5)
        action_prob = F.softmax(states, dim=-1)
        # print('action_prob', action_prob.size()) # (1,5)
        # exit()
        return action_prob

class CriticNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.critic = FeedForward(state_dim + action_dim, 1, hidden_sizes=(128, 64))

    def forward(self, state_actions):
        val = self.critic(state_actions)
        return val
