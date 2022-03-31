from typing import Optional, Tuple

import torch
import torch.nn as nn

class Config:
    def __init__(
        self,
        tokenIn=100,
        embSize=1024,
        positionMax=100,
        layer_norm_eps=1.0,
        encoder_lin=1024,
        encoder_lout=1024,
        ffn_out=4096,
        layerOut=1024,
        num_attention_heads=16,
        hidden_size=1024,
        attention_probs_dropout_prob=0.1,
        is_decoder=False,
        hidden_dropout_prob=0.1,
        intermediate_size=4096,
        seq=1024,
    ):
        self.tokenIn = tokenIn
        self.embSize = embSize
        self.positionMax = positionMax
        self.layer_norm_eps = layer_norm_eps
        self.encoder_lin = encoder_lin
        self.encoder_lout = encoder_lout
        self.ffn_out = ffn_out
        self.layerOut = layerOut

        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.is_decoder = is_decoder

        self.hidden_dropout_prob = hidden_dropout_prob
        self.intermediate_size = intermediate_size
        self.seq = seq


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads

        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )

        self.is_decoder = config.is_decoder
        self.scale = torch.randn(1).item()

        self.smax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        newView = (x.size(0), x.size(2), x.size(1), x.size(3))
        return x.view(newView)

    def changeViewForFirstMult(self, inp):
        batch, c, a = inp.size()
        return inp.view((batch * c, a))

    def changeViewAtTheEnd(self, inp):
        newView = (inp.size(0), inp.size(2), inp.size(1), inp.size(3))
        return inp.view(newView)

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
        baseSize = hidden_states.size()
        hidden_states = self.changeViewForFirstMult(hidden_states)

        qi = self.query(hidden_states).view(baseSize)
        ki = self.key(hidden_states).view(baseSize)
        vi = self.value(hidden_states).view(baseSize)

        key_layer = self.transpose_for_scores(ki)
        value_layer = self.transpose_for_scores(vi)
        query_layer = self.transpose_for_scores(qi)

        attention_scores = query_layer @ key_layer.transpose(-1, -2)
        attention_scores = attention_scores / (self.scale * self.scale)
      
       
        attention_probs = self.smax(attention_scores)
        
      
        attention_probs = self.dropout(attention_probs)
       
        context_layer = attention_scores @ value_layer
        context_layer = self.changeViewAtTheEnd(context_layer)
       
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class SelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        

    def changeViewForFirstMult(self, inp):
        batch, c, a = inp.size()
        return inp.view((batch * c, a))

    def forward(self, inp):

        hidden_states = inp[0]
        input_tensor = inp[1]
        originSize = hidden_states.size()
        hidden_states = self.changeViewForFirstMult(hidden_states)
       
        hidden_states = self.dense(hidden_states).view(originSize)
       
        hidden_states = self.dropout(hidden_states)
        
        hidden_states = self.changeViewForFirstMult(hidden_states + input_tensor)
       
        hidden_states = self.norm(hidden_states).view(originSize)
       
        return hidden_states


class Intermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.ReLU()
        self.intermediate_size = config.intermediate_size

    def changeViewForFirstMult(self, inp):
        batch, c, a = inp.size()
        return inp.view((batch * c, a))

    def forward(self, hidden_states):
        oS = hidden_states.size()
        hidden_states = self.changeViewForFirstMult(hidden_states)

        newView = (oS[0], oS[1], self.intermediate_size)
        
        hidden_states = self.dense(hidden_states).view(newView)

        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class Output(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.hidden_size = config.hidden_size

    def changeViewForFirstMult(self, inp):
        batch, c, a = inp.size()
        return inp.view((batch * c, a))

    def forward(self, hidden_states):
        originSize = hidden_states.size()
        newView = (originSize[0], originSize[1], self.hidden_size)
        hidden_states = self.changeViewForFirstMult(hidden_states)
        
        hidden_states = self.dense(hidden_states)
        
        hidden_states = self.dropout(hidden_states)
       
        hidden_states = self.norm(hidden_states).view(newView)
        return hidden_states


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.l0 = MultiHeadAttention(config)
        self.l1 = SelfOutput(config)
        self.l2 = Intermediate(config)
        self.l3 = Output(config)

    def forward(self, x0):
        x = self.l0(x0)
        x = self.l1([x, x0])
        x = self.l2(x)
        x = self.l3(x)
        return x


class TransformerBasedModel(nn.Module):
    def __init__(self, config, num_transformer_layers):
        super().__init__()
        
        self.embedding = nn.Embedding(config.tokenIn, config.embSize)
        
        self.transformerList = []
        for _ in range(num_transformer_layers):
            self.transformerList.append(Transformer(config))

        self.transformers = nn.Sequential(*self.transformerList)


    def forward(self, x):
        x = self.embedding(x)
        return self.transformers(x)


def main(args):
    config = Config(tokenIn=250001, embSize=1024)
    number_of_transformers = 12
    x = torch.randint(low=0, high=config.tokenIn, size=(4, 1024))

    m = TransformerBasedModel(config, number_of_transformers)
    m(x)
    return 0


if __name__ == '__main__':
    main(None)