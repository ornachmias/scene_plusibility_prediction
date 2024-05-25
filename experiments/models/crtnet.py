import os.path
from logging import Logger

from torch import optim

from experiments.experiment_type import ExperimentType
from experiments.models.base_model import BaseModel

import math

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import gdown


class CrtNet(BaseModel):
    def __init__(self, logger: Logger, experiment_type: ExperimentType, output_dir: str, device):
        super().__init__(logger, 'CRTNet', 224, experiment_type, output_dir, device)

    def optimizer(self):
        params_to_update = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)

        return optim.Adam(params_to_update, lr=1.0e-05)

    def init_model(self):
        pretrained_dir = os.path.join(self.output_dir, 'pretrained')
        os.makedirs(pretrained_dir, exist_ok=True)
        pretrained_path = os.path.join(self.output_dir, 'checkpoint_ocd_compatible.tar')
        if not os.path.isfile(pretrained_path):
            gdown.download(id='1XXCkgs7zhGg6jaD7J-n8MCT2vsFsY3fx', output=pretrained_path)

        checkpoint = torch.load(pretrained_path)
        model = Model()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.uncertainty_gate.target_classifier = nn.Linear(model.NUM_ENCODER_FEATURES, self.experiment_type.n_classes)
        model.classifier = nn.Linear(model.NUM_TOKEN_FEATURES, self.experiment_type.n_classes)
        return model

    def criterion(self):
        return CrtLoss(self.experiment_type)

    def forward(self, inputs):
        output_uncertainty_branch , output_main_branch, output_weighted, uncertainty = \
            self.model(inputs['context_image'], inputs['target_image'], inputs['target_bbox'])
        return {
            'output_uncertainty_branch': output_uncertainty_branch,
            'output_main_branch': output_main_branch,
            'output_weighted': output_weighted,
            'uncertainty': uncertainty
        }

    def predictions(self, outputs):
        if self.experiment_type.name == ExperimentType.reg:
            return outputs['output_main_branch'].clone().detach()
        else:
            return outputs['output_main_branch'].clone().detach().max(1).indices


class CrtLoss(nn.Module):
    def __init__(self, experiment_type: ExperimentType):
        super().__init__()
        if experiment_type.name == ExperimentType.bc:
            self.loss = nn.BCEWithLogitsLoss()
        elif experiment_type.name == ExperimentType.mcc:
            self.loss = nn.CrossEntropyLoss()
        elif experiment_type.name == ExperimentType.reg:
            self.loss = nn.L1Loss()

        self.last_input = None
        self.last_labels = None

    def forward(self, inputs, labels):
        self.last_input = inputs
        self.last_labels = labels
        return self

    def backward(self):
        loss_uncertainty_estimator = self.loss(self.last_input['output_weighted'], self.last_labels)
        loss_uncertainty_estimator.backward(retain_graph=True)

        loss_uncertainty_branch = self.loss(self.last_input['output_uncertainty_branch'], self.last_labels)
        loss_uncertainty_branch.backward(retain_graph=True)

        loss_main_branch = self.loss(self.last_input['output_main_branch'], self.last_labels)
        loss_main_branch.backward()

    def item(self):
        total_loss = 0
        loss_uncertainty_estimator = self.loss(self.last_input['output_weighted'], self.last_labels)
        total_loss += loss_uncertainty_estimator.item()

        loss_uncertainty_branch = self.loss(self.last_input['output_uncertainty_branch'], self.last_labels)
        total_loss += loss_uncertainty_branch.item()

        loss_main_branch = self.loss(self.last_input['output_main_branch'], self.last_labels)
        total_loss += loss_main_branch.item()
        return total_loss


class Model(nn.Module):
    def __init__(self):
        # Override configuration based on OCD training config
        # We want to model to be self-contained to fit in our code design
        super().__init__()

        num_classes = 15
        num_decoder_heads = 8
        num_decoder_layers = 6
        uncertainty_gate_type = 'learned'
        uncertainty_threshold = 0.0
        weighted_prediction = False
        extended_output = False
        gpu_streams = False

        self.NUM_CLASSES = num_classes

        self.context_encoder = Encoder()
        self.target_encoder = Encoder()

        self.CONTEXT_IMAGE_SIZE = self.context_encoder.IMAGE_SIZE
        self.TARGET_IMAGE_SIZE = self.target_encoder.IMAGE_SIZE
        self.NUM_ENCODER_FEATURES = self.context_encoder.NUM_FEATURES

        assert(self.context_encoder.NUM_FEATURES == self.target_encoder.NUM_FEATURES), "Context and target encoder must extract the same number of features."

        self.UNCERTAINTY_GATE_TYPE = uncertainty_gate_type
        self.UNCERTAINTY_THRESHOLD = uncertainty_threshold
        self.weighted_prediction = weighted_prediction
        self.uncertainty_gate = build_uncertainty_gate(self.NUM_ENCODER_FEATURES, self.NUM_CLASSES)

        self.tokenizer = Tokenizer()

        self.NUM_CONTEXT_TOKENS = self.tokenizer.NUM_CONTEXT_TOKENS
        self.NUM_TOKEN_FEATURES = self.NUM_ENCODER_FEATURES
        self.NUM_DECODER_HEADS = num_decoder_heads
        self.NUM_DECODER_LAYERS = num_decoder_layers

        assert(self.NUM_TOKEN_FEATURES % self.NUM_DECODER_HEADS == 0), "NUM_TOKEN_FEATURES must be divisible by NUM_DECODER_HEADS."

        self.positional_encoding = PositionalEncoding(self.NUM_CONTEXT_TOKENS, self.NUM_TOKEN_FEATURES)

        self.decoder_layers = TransformerDecoderLayerWithMap(self.NUM_TOKEN_FEATURES, nhead=self.NUM_DECODER_HEADS)
        self.decoder = TransformerDecoderWithMap(self.decoder_layers, self.NUM_DECODER_LAYERS)

        self.classifier = nn.Linear(self.NUM_TOKEN_FEATURES, self.NUM_CLASSES)

        self.initialize_weights()

        # cuda streams for parallel encoding
        self.gpu_streams = True if gpu_streams and torch.cuda.is_available() else False
        self.target_stream = torch.cuda.Stream(priority=-1) if self.gpu_streams else None # set target stream as high priority because context encoding may not be necessary due to uncertainty gating
        self.context_stream = torch.cuda.Stream() if self.gpu_streams else None

        self.extended_output = extended_output

    def forward(self, context_images, target_images, target_bbox):
        # Encoding of both streams
        if self.gpu_streams:
            torch.cuda.synchronize()

        with torch.cuda.stream(self.target_stream):
            target_encoding = self.target_encoder(target_images)

            # Uncertainty gating for target
            uncertainty_gate_prediction, uncertainty = self.uncertainty_gate(target_encoding.detach()) # Predictions and associated confidence metrics. Detach because encoder is trained via main branch only.

            # During inference, return uncertainty_gate_prediction if uncertainty is below the specified uncertainty threshold.
            # Note: The current implementation makes the gating decision on a per-batch basis. We expect/recommend that a batch size of 1 is used for inference.
            if not self.training and not self.extended_output and torch.all(uncertainty < self.UNCERTAINTY_THRESHOLD).item():
                return uncertainty_gate_prediction

        with torch.cuda.stream(self.context_stream):
            context_encoding = self.context_encoder(context_images)

        if self.gpu_streams:
            torch.cuda.synchronize()

        # Tokenization and positional encoding
        context_encoding, target_encoding = self.tokenizer(context_encoding, target_encoding)
        context_encoding, target_encoding = self.positional_encoding(context_encoding, target_encoding, target_bbox)

        # Incorporation of context information using transformer decoder
        target_encoding, attention_map = self.decoder(target_encoding, context_encoding)

        # Classification
        main_prediction = self.classifier(target_encoding.squeeze(0))
        weighted_prediction = uncertainty * main_prediction.detach() + (1-uncertainty) * uncertainty_gate_prediction.detach() # detached from main branch and uncertainty gate classifier

        if self.weighted_prediction:
            main_prediction = uncertainty.detach() * main_prediction + (1-uncertainty.detach()) * uncertainty_gate_prediction.detach() # detached from uncertainty branch

        # Return accoring to model state
        if self.training:
            return uncertainty_gate_prediction, main_prediction, weighted_prediction, uncertainty
        elif self.extended_output:
            return uncertainty_gate_prediction, main_prediction, uncertainty, attention_map
        else:
            return uncertainty_gate_prediction, main_prediction, weighted_prediction, uncertainty

    @torch.no_grad()
    def initialize_weights(self):
        for p in self.decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for p in self.classifier.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def freeze_target_encoder(self):
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def unfreeze_target_encoder(self):
        for param in self.target_encoder.parameters():
            param.requires_grad = True


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.encoder = torchvision.models.densenet169(pretrained=True).features
        self.IMAGE_SIZE = (224, 224)
        self.NUM_FEATURES = 1664

    def forward(self, image):
        return self.encoder(image)


class Tokenizer(nn.Module):
    def __init__(self):
        super(Tokenizer, self).__init__()

        self.NUM_CONTEXT_TOKENS = 49
        self.NUM_TARGET_TOKENS = 1

    def forward(self, context_encoding, target_encoding):
        """
        Creates tokens from the encoded context and target.
        The token shapes are (NUM_CONTEXT_TOKENS, BATCH_SIZE, NUM_TOKEN_FEATURES)
        and (NUM_TARGET_TOKENS, BATCH_SIZE, NUM_TOKEN_FEATURES) respectively.
        """

        # one target token
        target_encoding = F.relu(target_encoding)
        target_encoding = F.adaptive_avg_pool2d(target_encoding, (1, 1))
        target_encoding = torch.flatten(target_encoding, 1) # output dimension: (Batchsize, NUM_ENCODER_FEATURES)
        target_encoding = torch.unsqueeze(target_encoding, 0) # output dimension: (NUM_TARGET_TOKENS=1, BATCH_SIZE, NUM_TOKEN_FEATURES=NUM_ENCODER_FEATURES)

        # 49 context tokens
        context_encoding = F.relu(context_encoding)
        context_encoding = torch.flatten(context_encoding, 2, 3)
        context_encoding = context_encoding.permute(2, 0, 1) # output dimension: (NUM_CONTEXT_TOKENS=49, BATCH_SIZE, NUM_TOKEN_FEATURES=NUM_ENCODER_FEATURES)

        return context_encoding, target_encoding


class PositionalEncoding(nn.Module):

    def __init__(self, num_context_tokens, num_token_features):
        super(PositionalEncoding, self).__init__()

        self.NUM_CONTEXT_TOKENS = num_context_tokens
        self.tokens_per_dim = int(math.sqrt(self.NUM_CONTEXT_TOKENS))
        self.positional_encoding = nn.Parameter(torch.zeros(num_context_tokens, 1, num_token_features))
        self.initialize_weights()

    def forward(self, context_tokens, target_tokens, target_bbox):
        context_tokens = context_tokens + self.positional_encoding
        target_tokens = target_tokens + torch.index_select(self.positional_encoding, 0, self.bbox2token(target_bbox)).permute(1,0,2)

        return context_tokens, target_tokens

    def bbox2token(self, bbox):
        """
        Maps relative bbox coordinates to the corresponding token ids (e.g., 0 for the token in the top left).

        Arguments:
            bbox: Tensor of dim (batch_size, 4) where a row corresponds to relative coordinates
                  in the form [xmin, ymin, w, h] (e.g., [0.1, 0.3, 0.2, 0.2]).
        """
        token_ids = ((torch.ceil((bbox[:, 0] + bbox[:, 2]/2) * self.tokens_per_dim) - 1) +
                     (torch.ceil((bbox[:, 1] + bbox[:, 3]/2) * self.tokens_per_dim) - 1) * self.tokens_per_dim).long()

        return token_ids

    @torch.no_grad()
    def initialize_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


def build_uncertainty_gate(num_encoder_features, num_classes):
    return LearnedUncertaintyGate(num_encoder_features, num_classes)


class UncertaintyGate(nn.Module):
    def __init__(self, num_features, num_classes):
        super(UncertaintyGate, self).__init__()
        self.target_classifier = nn.Linear(num_features, num_classes)
        self.initialize_weights()

    def forward(self, input_features):
        # flatten featuremap out
        input_features = F.relu(input_features)
        input_features = F.adaptive_avg_pool2d(input_features, (1, 1))
        input_features = torch.flatten(input_features, 1) # output dimension: (Batchsize, NUM_ENCODER_FEATURES)

        predictions = self.target_classifier(input_features) # predictions dimension: (Batchsize, NUM_CLASSES)
        uncertainty = self.compute_uncertainty(predictions.detach())

        return predictions, uncertainty

    @staticmethod
    def compute_uncertainty(predictions):
        return -1 * torch.sum(F.softmax(predictions, dim=1) * F.log_softmax(predictions, dim=1), dim=1, keepdim=True) # entropy as metric for uncertainty

    @torch.no_grad()
    def initialize_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class LearnedUncertaintyGate(UncertaintyGate):
    def __init__(self, num_features, num_classes):
        super(LearnedUncertaintyGate, self).__init__(num_features, num_classes)
        self.uncertainty_estimator = nn.Sequential(nn.Linear(num_features, num_features // 2),
                                                   nn.ReLU(),
                                                   nn.Linear(num_features // 2, 1),
                                                   nn.Sigmoid())
        self.initialize_weights()

    def forward(self, input_features):
        # flatten featuremap out
        input_features = F.relu(input_features)
        input_features = F.adaptive_avg_pool2d(input_features, (1, 1))
        input_features = torch.flatten(input_features, 1) # output dimension: (Batchsize, NUM_ENCODER_FEATURES)

        predictions = self.target_classifier(input_features) # predictions dimension: (Batchsize, NUM_CLASSES)
        uncertainty = self.uncertainty_estimator(input_features)

        return predictions, uncertainty


class TransformerDecoderLayerWithMap(torch.nn.TransformerDecoderLayer):
    """
    Adapted version of torch.nn.TransformerDecoderLayer without the self-attention stage. In addition, the attention map is returned.
    """

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
        """
        tgt2, attention_map = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attention_map


class TransformerDecoderWithMap(torch.nn.TransformerDecoder):
    """
    Provides the same functionality as torch.nn.TransformerDecoder but returns the attention maps in addition.
    """

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
        """
        output = tgt
        attention_maps = []

        for mod in self.layers:
            output, attention_map = mod(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
            attention_maps.append(attention_map)

        if self.norm is not None:
            output = self.norm(output)

        return output, torch.stack(attention_maps).permute(1,0,2,3) # attention map shape: (batchsize, num_decoder_layers, num_target_tokens, num_context_tokens)
