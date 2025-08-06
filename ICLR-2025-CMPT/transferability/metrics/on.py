'''
Transferability measurements used in paper: On Transferability of Prompt Tuning for Natural Language Processing
ArXiv: https://arxiv.org/abs/2111.06719
On uses the overlapping rate between activations of the neurons in the feed-forward layers of Transformers as the
transferability measurement.
'''

import torch
import torch.nn.functional as F

def fake_attention_layer_forward(encoder, idx, hidden_states):

    head_mask = None
    output_attentions = False

    self = encoder.layer[idx]

    self_attention_outputs = self.attention(
        self.layernorm_before(hidden_states),  # in ViT, layernorm is applied before self-attention
        head_mask,
        output_attentions=output_attentions,
    )

    attention_output = self_attention_outputs[0]
    outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

    # first residual connection
    hidden_states = attention_output + hidden_states

    # in ViT, layernorm is also applied after self-attention
    layer_output = self.layernorm_after(hidden_states)
    activation = self.intermediate(layer_output)

    # second residual connection is done here
    layer_output = self.output(activation, hidden_states)

    outputs = (layer_output,) + outputs

    return outputs, activation

# Load the vit model
from transformers import ViTModel
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
encoder = vit_model.encoder

def model_stimulation_similarity(src_prompt, tgt_prompt, device='cpu'):

    # src_prompt = src_prompt.to('cpu')
    # tgt_prompt = tgt_prompt.to('cpu')

    vit_model.to(device)

    # Get the [CLS] embedding
    vit_cls = vit_model.embeddings.cls_token

    # Cat the [CLS] embedding to the end of source and target prompts
    src_prompt = torch.cat([src_prompt.unsqueeze(0), vit_cls], dim=1)
    tgt_prompt = torch.cat([tgt_prompt.unsqueeze(0), vit_cls], dim=1)


    with torch.no_grad():

        # Forward pass to get the hidden states of all layers
        src_output = encoder(src_prompt, output_hidden_states=True)
        tgt_output = encoder(tgt_prompt, output_hidden_states=True)

        # Get the activations
        src_activations = []
        tgt_activations = []
        for idx in (-4, -3, -2):

            src_hs = src_output['hidden_states'][idx]
            tgt_hs = tgt_output['hidden_states'][idx]

            src_layer_opt, src_activation = fake_attention_layer_forward(encoder, idx + 1, src_hs)

            # Sanity check, the outputs should be equal to the hidden state of the next layer
            assert (src_layer_opt[0] == src_output['hidden_states'][idx + 1]).all()

            # Deal with the source activations
            src_activation = src_activation[:, -1, :].squeeze(0)
            src_activation[src_activation <= 0] = 0
            src_activation[src_activation > 0] = 1

            tgt_layer_opt, tgt_activation = fake_attention_layer_forward(encoder, idx + 1, tgt_hs)

            # Sanity check
            assert (tgt_layer_opt[0] == tgt_output['hidden_states'][idx + 1]).all()

            # Deal with the target activations
            tgt_activation = tgt_activation[:, -1, :].squeeze(0)
            tgt_activation[tgt_activation <= 0] = 0
            tgt_activation[tgt_activation > 0] = 1

            src_activations.append(src_activation)
            tgt_activations.append(tgt_activation)

        src_activations = torch.cat(src_activations).unsqueeze(0)
        tgt_activations = torch.cat(tgt_activations).unsqueeze(0)

    return F.cosine_similarity(src_activations, tgt_activations).mean()

if __name__ == '__main__':

    src_prompt = torch.rand((100, 768))
    tgt_prompt = torch.rand((100, 768))

    print(model_stimulation_similarity(src_prompt, tgt_prompt))




