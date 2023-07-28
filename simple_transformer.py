import torch
import torch.nn as nn
import math


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, attention_head_size, num_attention_heads,
            attention_probs_dropout_prob=0.1, use_structure_matrix=False):
        super().__init__()

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = attention_head_size
        
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

        if use_structure_matrix:
            self.structure_map_query = nn.Sequential(
                nn.Linear(self.attention_head_size, self.attention_head_size))
            self.structure_map_key = nn.Sequential(
                nn.Linear(self.attention_head_size, self.attention_head_size))
            pass


    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (-1, self.attention_head_size)
        x = x.reshape(*new_x_shape)
        return x.permute(0, 2, 1, 3).contiguous()

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        structure_matrix=None,
    ):
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        if structure_matrix is not None:
            structure_key = self.structure_map_key(structure_matrix)
            structure_query = self.structure_map_query(structure_matrix)
            structure_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, structure_query)
            structure_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, structure_key)
            attention_scores = attention_scores + structure_scores_query + \
                structure_scores_key
            pass
        
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:    
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape).contiguous()

        outputs = context_layer

        return outputs, attention_scores

    pass



class Attention(nn.Module):
    def __init__(self, hidden_size, attention_head_size, num_attention_heads, 
            hidden_dropout_prob, attention_probs_dropout_prob, use_structure_matrix=False):
        super().__init__()
        self.self = SelfAttention(
            hidden_size, attention_head_size, num_attention_heads, attention_probs_dropout_prob, use_structure_matrix)
        self.dense = nn.Linear(num_attention_heads * attention_head_size, hidden_size)
        self.layer_norm_output = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)


    def forward(
        self,
        hidden_states,
        attention_mask=None,
        structure_matrix=None,
    ):
        self_outputs, attention_scores = self.self(
            hidden_states,
            attention_mask,
            structure_matrix=structure_matrix,
        )

        output = self.dense(self_outputs)
        output = self.dropout(output)
        output = self.layer_norm_output(output + hidden_states)
        return output, attention_scores

    pass



class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, attention_head_size, num_attention_heads,
            hidden_dropout_prob, attention_probs_dropout_prob, use_structure_matrix=False):
        super().__init__()
        self.attention = Attention(
            hidden_size, attention_head_size, num_attention_heads,
            hidden_dropout_prob, attention_probs_dropout_prob, use_structure_matrix)
        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = nn.GELU()
        self.dense_output = nn.Linear(intermediate_size, hidden_size)
        self.layer_norm_output = nn.LayerNorm(hidden_size)
        self.dropout_output = nn.Dropout(hidden_dropout_prob)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        structure_matrix=None,
    ):     
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            structure_matrix=structure_matrix,
        )
        
        attention_output, attention_scores = self_attention_outputs
        x_intermediate = self.intermediate_act_fn(self.intermediate(attention_output))
        output = self.dense_output(x_intermediate)
        output = self.dropout_output(output)
        output = self.layer_norm_output(output + attention_output)
        return output, attention_scores
    pass


class MultiLayerTransformer(nn.Module):
    def __init__(self, num_layers, hidden_size, intermediate_size=None, attention_head_size=None,
            num_attention_heads=None, hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
            reduction='top', use_structure_matrix=False):
        super().__init__()
        self.bias = nn.init.uniform_(
            nn.Parameter(torch.zeros(hidden_size)))

        if intermediate_size is None:
            intermediate_size = 4 * hidden_size
            pass

        if attention_head_size is None:
            attention_head_size = hidden_size // 4
            pass

        if num_attention_heads is None:
            num_attention_heads = hidden_size // attention_head_size
        
        self.layers = nn.ModuleList([
            TransformerLayer(
                hidden_size, intermediate_size, attention_head_size, num_attention_heads,
                hidden_dropout_prob, attention_probs_dropout_prob, use_structure_matrix)
            for _ in range(num_layers)])

        self.reduction = reduction
        
        pass

    def forward(self, hidden_states, attention_mask=None, structure_matrix=None):

        hidden_states = self.forward_(hidden_states, attention_mask, structure_matrix)

        if self.reduction == 'top':
            return hidden_states[:, 0, :]
        elif self.reduction == 'mean':
            if attention_mask is not None:
                hidden_states = hidden_states * attention_mask[:, :, None]
                hidden_states = torch.sum(hidden_states, dim=1) / torch.sum(attention_mask, dim=1, keepdim=True)
                pass
            
            return hidden_states
        elif self.reduction is None:
            return hidden_states
        else:
            raise RuntimeError(f'cannot recognize reduction: {self.reduction}')
        pass


    def forward_(self, hidden_states, attention_mask=None, structure_matrix=None):
        if attention_mask is not None:
            attention_mask_ = (1.0 - attention_mask[:, None, None, :]) * -10000000
            pass
        else:
            attention_mask_ = None
            pass

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask_, structure_matrix)[0]
            pass


        return hidden_states
    pass