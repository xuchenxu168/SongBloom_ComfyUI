from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import chain
import logging
import typing as tp
import einops

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from dataclasses import dataclass, field, fields, replace

from ..modules.streaming import StreamingModule
from ...base.utils import length_to_mask, create_sin_embedding


def collate(tensors: tp.List[torch.Tensor], dim: int = 0) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    """Get a list of tensors and collate them to a single tensor. according to the following logic:
    - `dim` specifies the time dimension which will be stacked and padded.
    - The output will contain 1 new dimension (dimension index 0) which will be the size of
    of the original list.

    Args:
        tensors (tp.List[torch.Tensor]): List of tensors to collate.
        dim (int): Dimension which will be stacked and padded.
    Returns:
        tp.Tuple[torch.Tensor, torch.Tensor]:
            torch.Tensor: Stacked and padded tensor. The output will contain 1 new dimension
                (dimension index 0) which will be the size of the original list.
            torch.Tensor: Tensor containing length of original tensor sizes (without padding).
    """
    tensors = [x.transpose(0, dim) for x in tensors]
    lens = torch.LongTensor([len(x) for x in tensors])
    padded_tensors = pad_sequence(tensors)
    padded_tensors = padded_tensors.transpose(0, 1)
    padded_tensors = padded_tensors.transpose(1, dim + 1)
    return padded_tensors, lens



@dataclass(order=True)
class PathInZip:
    """Hold a path of file within a zip file.

    Args:
        path (str): The convention is <path_to_zip>:<relative_path_inside_zip>.
            Let's assume there is a zip file /some/location/foo.zip
            and inside of it is a json file located at /data/file1.json,
            Then we expect path = "/some/location/foo.zip:/data/file1.json".
    """

    INFO_PATH_SEP = ':'
    zip_path: str
    file_path: str

    def __init__(self, path: str) -> None:
        split_path = path.split(self.INFO_PATH_SEP)
        assert len(split_path) == 2
        self.zip_path, self.file_path = split_path

    @classmethod
    def from_paths(cls, zip_path: str, file_path: str):
        return cls(zip_path + cls.INFO_PATH_SEP + file_path)

    def __str__(self) -> str:
        return self.zip_path + self.INFO_PATH_SEP + self.file_path


@dataclass(order=True)
class BaseInfo:

    @classmethod
    def _dict2fields(cls, dictionary: dict):
        return {
                field.name: dictionary[field.name]
                for field in fields(cls) if field.name in dictionary
            }
        # try:
        #     return {
        #         field.name: dictionary[field.name]
        #         for field in fields(cls) if field.name in dictionary
        #     }
        # except:
        #     print(dictionary)

    @classmethod
    def from_dict(cls, dictionary: dict):
        _dictionary = cls._dict2fields(dictionary)
        return cls(**_dictionary)

    def to_dict(self):
        return {
            field.name: self.__getattribute__(field.name)
            for field in fields(self)
            }


@dataclass(order=True)
class AudioMeta(BaseInfo):
    path: str
    duration: float
    sample_rate: int
    amplitude: tp.Optional[float] = None
    weight: tp.Optional[float] = None
    # info_path is used to load additional information about the audio file that is stored in zip files.
    info_path: tp.Optional[PathInZip] = None

    @classmethod
    def from_dict(cls, dictionary: dict):
        base = cls._dict2fields(dictionary)
        if 'info_path' in base and base['info_path'] is not None:
            base['info_path'] = PathInZip(base['info_path'])
        return cls(**base)

    def to_dict(self):
        d = super().to_dict()
        if d['info_path'] is not None:
            d['info_path'] = str(d['info_path'])
        return d


@dataclass(order=True)
class SegmentInfo(BaseInfo):
    meta: AudioMeta
    seek_time: float
    # The following values are given once the audio is processed, e.g.
    # at the target sample rate and target number of channels.
    n_frames: int      # actual number of frames without padding
    total_frames: int  # total number of frames, padding included
    sample_rate: int   # actual sample rate
    channels: int      # number of audio channels.


logger = logging.getLogger(__name__)
TextCondition = tp.Optional[str]  # a text condition can be a string or None (if doesn't exist)
ConditionType = tp.Tuple[torch.Tensor, torch.Tensor]  # condition, mask


class WavCondition(tp.NamedTuple):
    wav: torch.Tensor
    length: torch.Tensor
    sample_rate: tp.List[int]
    path: tp.List[tp.Optional[str]] = []
    seek_time: tp.List[tp.Optional[float]] = []


class JointEmbedCondition(tp.NamedTuple):
    wav: torch.Tensor
    text: tp.List[tp.Optional[str]]
    length: torch.Tensor
    sample_rate: tp.List[int]
    path: tp.List[tp.Optional[str]] = []
    seek_time: tp.List[tp.Optional[float]] = []


@dataclass
class ConditioningAttributes:
    text: tp.Dict[str, tp.Optional[str]] = field(default_factory=dict)
    wav: tp.Dict[str, WavCondition] = field(default_factory=dict)
    joint_embed: tp.Dict[str, JointEmbedCondition] = field(default_factory=dict)

    def __getitem__(self, item):
        return getattr(self, item)

    @property
    def text_attributes(self):
        return self.text.keys()

    @property
    def wav_attributes(self):
        return self.wav.keys()

    @property
    def joint_embed_attributes(self):
        return self.joint_embed.keys()

    @property
    def attributes(self):
        return {
            "text": self.text_attributes,
            "wav": self.wav_attributes,
            "joint_embed": self.joint_embed_attributes,
        }

    def to_flat_dict(self):
        return {
            **{f"text.{k}": v for k, v in self.text.items()},
            **{f"wav.{k}": v for k, v in self.wav.items()},
            **{f"joint_embed.{k}": v for k, v in self.joint_embed.items()}
        }

    @classmethod
    def from_flat_dict(cls, x):
        out = cls()
        for k, v in x.items():
            kind, att = k.split(".")
            out[kind][att] = v
        return out



# class SegmentWithAttributes(SegmentInfo):
#     """Base class for all dataclasses that are used for conditioning.
#     All child classes should implement `to_condition_attributes` that converts
#     the existing attributes to a dataclass of type ConditioningAttributes.
#     """
#     def to_condition_attributes(self) -> ConditioningAttributes:
#         raise NotImplementedError()



def nullify_condition(condition: ConditionType, dim: int = 1):
    """Transform an input condition to a null condition.
    The way it is done by converting it to a single zero vector similarly
    to how it is done inside WhiteSpaceTokenizer and NoopTokenizer.

    Args:
        condition (ConditionType): A tuple of condition and mask (tuple[torch.Tensor, torch.Tensor])
        dim (int): The dimension that will be truncated (should be the time dimension)
        WARNING!: dim should not be the batch dimension!
    Returns:
        ConditionType: A tuple of null condition and mask
    """
    assert dim != 0, "dim cannot be the batch dimension!"
    assert isinstance(condition, tuple) and \
        isinstance(condition[0], torch.Tensor) and \
        isinstance(condition[1], torch.Tensor), "'nullify_condition' got an unexpected input type!"
    cond, mask = condition
    B = cond.shape[0]
    last_dim = cond.dim() - 1
    out = cond.transpose(dim, last_dim)
    out = 0. * out[..., :1]
    out = out.transpose(dim, last_dim)
    mask = torch.zeros((B, 1), device=out.device).int()
    assert cond.dim() == out.dim()
    return out, mask


def nullify_wav(cond: WavCondition) -> WavCondition:
    """Transform a WavCondition to a nullified WavCondition.
    It replaces the wav by a null tensor, forces its length to 0, and replaces metadata by dummy attributes.

    Args:
        cond (WavCondition): Wav condition with wav, tensor of shape [B, T].
    Returns:
        WavCondition: Nullified wav condition.
    """
    #TODO by YCY, fix this to support zero-length input (as None)
    null_wav, _ = nullify_condition((cond.wav, torch.zeros_like(cond.wav)), dim=cond.wav.dim() - 1) # B,1 all-zero
    return WavCondition(
        wav=null_wav,
        length=torch.tensor([0] * cond.wav.shape[0], device=cond.wav.device),
        sample_rate=cond.sample_rate,
        path=[None] * cond.wav.shape[0],
        seek_time=[None] * cond.wav.shape[0],
    )


def nullify_joint_embed(embed: JointEmbedCondition) -> JointEmbedCondition:
    """Nullify the joint embedding condition by replacing it by a null tensor, forcing its length to 0,
    and replacing metadata by dummy attributes.

    Args:
        cond (JointEmbedCondition): Joint embedding condition with wav and text, wav tensor of shape [B, C, T].
    """
    null_wav, _ = nullify_condition((embed.wav, torch.zeros_like(embed.wav)), dim=embed.wav.dim() - 1)
    return JointEmbedCondition(
        wav=null_wav, text=[None] * len(embed.text),
        length=torch.LongTensor([0]).to(embed.wav.device),
        sample_rate=embed.sample_rate,
        path=[None] * embed.wav.shape[0],
        seek_time=[0] * embed.wav.shape[0],
    )



class BaseConditioner(nn.Module):
    """Base model for all conditioner modules.
    We allow the output dim to be different than the hidden dim for two reasons:
    1) keep our LUTs small when the vocab is large;
    2) make all condition dims consistent.

    Args:
        dim (int): Hidden dim of the model.
        output_dim (int): Output dim of the conditioner.
    """
    def __init__(self, dim: int, output_dim: int, input_token = False, padding_idx=None):
        super().__init__()
        self.dim = dim
        self.output_dim = output_dim
        if input_token:
            self.output_proj = nn.Embedding(dim, output_dim, padding_idx)
        else:
            self.output_proj = nn.Linear(dim, output_dim)

    def tokenize(self, *args, **kwargs) -> tp.Any:
        """Should be any part of the processing that will lead to a synchronization
        point, e.g. BPE tokenization with transfer to the GPU.

        The returned value will be saved and return later when calling forward().
        """
        raise NotImplementedError()

    def forward(self, inputs: tp.Any) -> ConditionType:
        """Gets input that should be used as conditioning (e.g, genre, description or a waveform).
        Outputs a ConditionType, after the input data was embedded as a dense vector.

        Returns:
            ConditionType:
                - A tensor of size [B, T, D] where B is the batch size, T is the length of the
                  output embedding and D is the dimension of the embedding.
                - And a mask indicating where the padding tokens.
        """
        raise NotImplementedError()



def dropout_condition(sample: ConditioningAttributes, condition_type: str, condition: str) -> ConditioningAttributes:
    """Utility function for nullifying an attribute inside an ConditioningAttributes object.
    If the condition is of type "wav", then nullify it using `nullify_condition` function.
    If the condition is of any other type, set its value to None.
    Works in-place.
    """
    if condition_type not in ['text', 'wav', 'joint_embed']:
        raise ValueError(
            "dropout_condition got an unexpected condition type!"
            f" expected 'text', 'wav' or 'joint_embed' but got '{condition_type}'"
        )

    if condition not in getattr(sample, condition_type):
        raise ValueError(
            "dropout_condition received an unexpected condition!"
            f" expected wav={sample.wav.keys()} and text={sample.text.keys()}"
            f" but got '{condition}' of type '{condition_type}'!"
        )

    if condition_type == 'wav':
        wav_cond = sample.wav[condition]
        sample.wav[condition] = nullify_wav(wav_cond)
    elif condition_type == 'joint_embed':
        embed = sample.joint_embed[condition]
        sample.joint_embed[condition] = nullify_joint_embed(embed)
    else:
        sample.text[condition] = None

    return sample


class DropoutModule(nn.Module):
    """Base module for all dropout modules."""
    def __init__(self, seed: int = 1234):
        super().__init__()
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)


class AttributeDropout(DropoutModule):
    """Dropout with a given probability per attribute.
    This is different from the behavior of ClassifierFreeGuidanceDropout as this allows for attributes
    to be dropped out separately. For example, "artist" can be dropped while "genre" remains.
    This is in contrast to ClassifierFreeGuidanceDropout where if "artist" is dropped "genre"
    must also be dropped.

    Args:
        p (tp.Dict[str, float]): A dict mapping between attributes and dropout probability. For example:
            ...
            "genre": 0.1,
            "artist": 0.5,
            "wav": 0.25,
            ...
        active_on_eval (bool, optional): Whether the dropout is active at eval. Default to False.
        seed (int, optional): Random seed.
    """
    def __init__(self, p: tp.Dict[str, tp.Dict[str, float]], active_on_eval: bool = False, seed: int = 1234):
        super().__init__(seed=seed)
        self.active_on_eval = active_on_eval
        # construct dict that return the values from p otherwise 0
        self.p = {}
        for condition_type, probs in p.items():
            self.p[condition_type] = defaultdict(lambda: 0, probs)

    def forward(self, samples: tp.List[ConditioningAttributes]) -> tp.List[ConditioningAttributes]:
        """
        Args:
            samples (list[ConditioningAttributes]): List of conditions.
        Returns:
            list[ConditioningAttributes]: List of conditions after certain attributes were set to None.
        """
        if not self.training and not self.active_on_eval:
            return samples

        samples = deepcopy(samples)
        for condition_type, ps in self.p.items():  # for condition types [text, wav]
            for condition, p in ps.items():  # for attributes of each type (e.g., [artist, genre])
                # import pdb; pdb.set_trace()
                # print(condition, p)
                if torch.rand(1, generator=self.rng).item() < p:
                    for sample in samples:
                        dropout_condition(sample, condition_type, condition)
        return samples

    def __repr__(self):
        return f"AttributeDropout({dict(self.p)})"


class ClassifierFreeGuidanceDropout(DropoutModule):
    """Classifier Free Guidance dropout.
    All attributes are dropped with the same probability.

    Args:
        p (float): Probability to apply condition dropout during training.
        seed (int): Random seed.
    """
    def __init__(self, p: float, seed: int = 1234):
        super().__init__(seed=seed)
        self.p = p

    def forward(self, samples: tp.List[ConditioningAttributes]) -> tp.List[ConditioningAttributes]:
        """
        Args:
            samples (list[ConditioningAttributes]): List of conditions.
        Returns:
            list[ConditioningAttributes]: List of conditions after all attributes were set to None.
        """

        if not self.training:
            return samples
        # import pdb; pdb.set_trace()
        # decide on which attributes to drop in a batched fashion
        drop = torch.rand(1, generator=self.rng).item() < self.p
        if not drop:
            return samples

        # nullify conditions of all attributes
        samples = deepcopy(samples)
        for condition_type in ["text", "wav","joint_embed"]:
            for sample in samples:
                for condition in sample.attributes[condition_type]:
                    dropout_condition(sample, condition_type, condition)
        return samples

    def __repr__(self):
        return f"ClassifierFreeGuidanceDropout(p={self.p})"


class TextConditioner(BaseConditioner):
    ...


class WaveformConditioner(BaseConditioner):
    """Base class for all conditioners that take a waveform as input.
    Classes that inherit must implement `_get_wav_embedding` that outputs
    a continuous tensor, and `_downsampling_factor` that returns the down-sampling
    factor of the embedding model.

    Args:
        dim (int): The internal representation dimension.
        output_dim (int): Output dimension.
    """
    def __init__(self, dim: int, output_dim: int, input_token = False, padding_idx=None):
        super().__init__(dim, output_dim, input_token, padding_idx)

    def tokenize(self, x: WavCondition) -> WavCondition:
        wav, length, sample_rate, path, seek_time = x
        assert length is not None
        return WavCondition(wav, length, sample_rate, path, seek_time)

    def _get_wav_embedding(self, x: WavCondition) -> torch.Tensor:
        """Gets as input a WavCondition and returns a dense embedding."""
        raise NotImplementedError()

    def _downsampling_factor(self):
        """Returns the downsampling factor of the embedding model."""
        raise NotImplementedError()

    def forward(self, x: WavCondition) -> ConditionType:
        """Extract condition embedding and mask from a waveform and its metadata.
        Args:
            x (WavCondition): Waveform condition containing raw waveform and metadata.
        Returns:
            ConditionType: a dense vector representing the conditioning along with its mask
        """

        wav, lengths, *_ = x
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            embeds = self._get_wav_embedding(x)
        embeds = embeds.to(self.output_proj.weight)
        embeds = self.output_proj(embeds)
        # import pdb; pdb.set_trace()
        if lengths is not None:
            lengths = lengths / self._downsampling_factor()
            mask = length_to_mask(lengths, max_len=embeds.shape[1]).int()  # type: ignore
        else:
            mask = torch.ones_like(embeds)
        embeds = (embeds * mask.unsqueeze(2))

        return embeds, mask


class JointEmbeddingConditioner(BaseConditioner):
    """Joint embedding conditioning supporting both audio or text conditioning.

    Args:
        dim (int): Dimension.
        output_dim (int): Output dimension.
        autocast_dtype (str): Autocast for the conditioner.
        quantize (bool): Whether to quantize the CLAP embedding.
        n_q (int): Number of residual quantizers (used if quantize is true).
        bins (int): Quantizers' codebooks size (used if quantize is true).
        kwargs: Additional parameters for residual vector quantizer.
    """
    def __init__(self, dim: int, output_dim: int, 
                 autocast_dtype: tp.Optional[str] = 'float32', #quantize: bool = False,
                 **kwargs):
        super().__init__(dim=dim, output_dim=output_dim)
        self.autocast_dtype = getattr(torch, autocast_dtype) if autocast_dtype is not None \
                                else None
        if self.autocast_dtype is None:
            logger.warning("JointEmbeddingConditioner has no autocast, this might lead to NaN.")

        # # residual vector quantizer to discretize the conditioned embedding
        # self.quantizer  = None
        # if quantize:
        #     from ..modules.quantization import ResidualVectorQuantizer
        #     self.quantizer = ResidualVectorQuantizer(dim, n_q=n_q, bins=bins, **kwargs)

    def _get_embed(self, x: JointEmbedCondition) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """Get joint embedding in latent space from the inputs.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tensor for the latent embedding
                and corresponding empty indexes.
        """
        raise NotImplementedError()

    def forward(self, x: JointEmbedCondition) -> ConditionType:
        with torch.cuda.amp.autocast(dtype=self.autocast_dtype):
            embed, empty_idx = self._get_embed(x)
            if self.quantizer is not None:
                embed = embed.view(-1, self.dim, 1)
                q_res = self.quantizer(embed, frame_rate=1)
                out_embed = q_res.x.view(-1, self.dim)
            else:
                out_embed = embed
            out_embed = self.output_proj(out_embed).view(-1, 1, self.output_dim)
            mask = torch.ones(*out_embed.shape[:2], device=out_embed.device)
            mask[empty_idx, :] = 0  # zero-out index where the input is non-existant
            out_embed = (out_embed * mask.unsqueeze(-1))
            return out_embed, mask

    def tokenize(self, x: JointEmbedCondition) -> JointEmbedCondition:
        return x


class ConditioningProvider(nn.Module):
    """Prepare and provide conditions given all the supported conditioners.

    Args:
        conditioners (dict): Dictionary of conditioners.
    """
    def __init__(self, conditioners: tp.Dict[str, BaseConditioner]):
        super().__init__()
        self.conditioners = nn.ModuleDict(conditioners)
        def _check_conditioner_type(c):
            if isinstance(c, WaveformConditioner):
                return "wav"
            elif isinstance(c, TextConditioner):
                return "text"
            elif isinstance(c, JointEmbeddingConditioner):
                return "joint_embed"
            else:
                raise NotImplementedError(f"{type(c)} are not Implemented!")
        self.conditioner_type = {k: _check_conditioner_type(self.conditioners[k]) for k in self.conditioners}


    @property
    def joint_embed_conditions(self):
        return [k for k, v in self.conditioners.items() if isinstance(v, JointEmbeddingConditioner)]

    @property
    def has_joint_embed_conditions(self):
        return len(self.joint_embed_conditions) > 0

    @property
    def text_conditions(self):
        return [k for k, v in self.conditioners.items() if isinstance(v, TextConditioner)]

    @property
    def wav_conditions(self):
        return [k for k, v in self.conditioners.items() if isinstance(v, WaveformConditioner)]

    @property
    def has_wav_condition(self):
        return len(self.wav_conditions) > 0

    def tokenize(self, inputs: tp.List[ConditioningAttributes]) -> tp.Dict[str, tp.Any]:
        """Match attributes/wavs with existing conditioners in self, and compute tokenize them accordingly.
        This should be called before starting any real GPU work to avoid synchronization points.
        This will return a dict matching conditioner names to their arbitrary tokenized representations.

        Args:
            inputs (list[ConditioningAttributes]): List of ConditioningAttributes objects containing
                text and wav conditions.
        """
        assert all([isinstance(x, ConditioningAttributes) for x in inputs]), (
            "Got unexpected types input for conditioner! should be tp.List[ConditioningAttributes]",
            f" but types were {set([type(x) for x in inputs])}"
        )

        # import pdb; pdb.set_trace()
        output = {}
        text = self._collate_text(inputs)
        wavs = self._collate_wavs(inputs)
        joint_embeds = self._collate_joint_embeds(inputs)

        assert set(text.keys() | wavs.keys() | joint_embeds.keys()).issubset(set(self.conditioners.keys())), (
            f"Got an unexpected attribute! Expected {self.conditioners.keys()}, ",
            f"got {text.keys(), wavs.keys(), joint_embeds.keys()}"
        )

        for attribute, batch in chain(text.items(), wavs.items(), joint_embeds.items()):
            output[attribute] = self.conditioners[attribute].tokenize(batch)
        return output

    def forward(self, tokenized: tp.Dict[str, tp.Any], texts = None) -> tp.Dict[str, ConditionType]:
        """Compute pairs of `(embedding, mask)` using the configured conditioners and the tokenized representations.
        The output is for example:
        {
            "genre": (torch.Tensor([B, 1, D_genre]), torch.Tensor([B, 1])),
            "description": (torch.Tensor([B, T_desc, D_desc]), torch.Tensor([B, T_desc])),
            ...
        }

        Args:
            tokenized (dict): Dict of tokenized representations as returned by `tokenize()`.
        """
        # import pdb; pdb.set_trace()
        output = {}
        for attribute, inputs in tokenized.items():
            if attribute == 'self_wav' and texts is not None:
                condition, mask = self.conditioners[attribute](inputs, texts = texts)
            else:
                condition, mask = self.conditioners[attribute](inputs)
            output[attribute] = (condition, mask)
        return output

    def _collate_text(self, samples: tp.List[ConditioningAttributes]) -> tp.Dict[str, tp.List[tp.Optional[str]]]:
        """Given a list of ConditioningAttributes objects, compile a dictionary where the keys
        are the attributes and the values are the aggregated input per attribute.
        For example:
        Input:
        [
            ConditioningAttributes(text={"genre": "Rock", "description": "A rock song with a guitar solo"}, wav=...),
            ConditioningAttributes(text={"genre": "Hip-hop", "description": "A hip-hop verse"}, wav=...),
        ]
        Output:
        {
            "genre": ["Rock", "Hip-hop"],
            "description": ["A rock song with a guitar solo", "A hip-hop verse"]
        }

        Args:
            samples (list of ConditioningAttributes): List of ConditioningAttributes samples.
        Returns:
            dict[str, list[str, optional]]: A dictionary mapping an attribute name to text batch.
        """
        out: tp.Dict[str, tp.List[tp.Optional[str]]] = defaultdict(list)
        texts = [x.text for x in samples]
        for text in texts:
            for condition in self.text_conditions:
                out[condition].append(text[condition])
        return out

    def _collate_wavs(self, samples: tp.List[ConditioningAttributes]) -> tp.Dict[str, WavCondition]:
        """Generate a dict where the keys are attributes by which we fetch similar wavs,
        and the values are Tensors of wavs according to said attributes.

        *Note*: by the time the samples reach this function, each sample should have some waveform
        inside the "wav" attribute. It should be either:
        1. A real waveform
        2. A null waveform due to the sample having no similar waveforms (nullified by the dataset)
        3. A null waveform due to it being dropped in a dropout module (nullified by dropout)

        Args:
            samples (list of ConditioningAttributes): List of ConditioningAttributes samples.
        Returns:
            dict[str, WavCondition]: A dictionary mapping an attribute name to wavs.
        """
        # import pdb; pdb.set_trace()
        wavs = defaultdict(list)
        lengths = defaultdict(list)
        sample_rates = defaultdict(list)
        paths = defaultdict(list)
        seek_times = defaultdict(list)
        out: tp.Dict[str, WavCondition] = {}

        for sample in samples:
            for attribute in self.wav_conditions:
                wav, length, sample_rate, path, seek_time = sample.wav[attribute]
                assert wav.dim() == 3, f"Got wav with dim={wav.dim()}, but expected 3 [1, C, T]"
                assert wav.size(0) == 1, f"Got wav [B, C, T] with shape={wav.shape}, but expected B == 1"
                # mono-channel conditioning
                # wav = wav.mean(1, keepdim=True)  # [1, 1, T] # by cyy, 为了实现后续功能注释掉了，请手动确保channel=1，or 输入channel 符合预期
                wavs[attribute].append(wav.flatten())  # [C*T]
                lengths[attribute].append(length)
                sample_rates[attribute].extend(sample_rate)
                paths[attribute].extend(path)
                seek_times[attribute].extend(seek_time)

        # stack all wavs to a single tensor
        for attribute in self.wav_conditions:
            stacked_wav, _ = collate(wavs[attribute], dim=0)
            out[attribute] = WavCondition(
                stacked_wav.unsqueeze(1), torch.cat(lengths[attribute]), sample_rates[attribute],
                paths[attribute], seek_times[attribute])

        return out

    def _collate_joint_embeds(self, samples: tp.List[ConditioningAttributes]) -> tp.Dict[str, JointEmbedCondition]:
        """Generate a dict where the keys are attributes by which we compute joint embeddings,
        and the values are Tensors of pre-computed embeddings and the corresponding text attributes.

        Args:
            samples (list[ConditioningAttributes]): List of ConditioningAttributes samples.
        Returns:
            A dictionary mapping an attribute name to joint embeddings.
        """
        texts = defaultdict(list)
        wavs = defaultdict(list)
        lengths = defaultdict(list)
        sample_rates = defaultdict(list)
        paths = defaultdict(list)
        seek_times = defaultdict(list)
        channels: int = 0
        
        out = {}
        for sample in samples:
            for attribute in self.joint_embed_conditions:
                wav, text, length, sample_rate, path, seek_time = sample.joint_embed[attribute]
                assert wav.dim() == 3
                if channels == 0:
                    channels = wav.size(1)
                else:
                    assert channels == wav.size(1), "not all audio has same number of channels in batch"
                assert wav.size(0) == 1, "Expecting single-wav batch in the collate method"
                wav = einops.rearrange(wav, "b c t -> (b c t)")  # [1, C, T] => [C * T] 
                wavs[attribute].append(wav)
                texts[attribute].extend(text)
                lengths[attribute].append(length)
                sample_rates[attribute].extend(sample_rate)
                paths[attribute].extend(path)
                seek_times[attribute].extend(seek_time)

        for attribute in self.joint_embed_conditions:
            stacked_texts = texts[attribute]
            stacked_paths = paths[attribute]
            stacked_seek_times = seek_times[attribute]
            stacked_wavs = pad_sequence(wavs[attribute])
            stacked_wavs = einops.rearrange(stacked_wavs, "(c t) b -> b c t", c=channels)
            stacked_sample_rates = sample_rates[attribute]
            stacked_lengths = torch.cat(lengths[attribute])

            assert stacked_lengths.size(0) == stacked_wavs.size(0)
            assert len(stacked_sample_rates) == stacked_wavs.size(0)
            assert len(stacked_texts) == stacked_wavs.size(0)
            out[attribute] = JointEmbedCondition(
                text=stacked_texts, wav=stacked_wavs,
                length=stacked_lengths, sample_rate=stacked_sample_rates,
                path=stacked_paths, seek_time=stacked_seek_times)

        return out


class ConditionFuser(StreamingModule):
    """Condition fuser handles the logic to combine the different conditions
    to the actual model input.

    Args:
        fuse2cond (tp.Dict[str, str]): A dictionary that says how to fuse
            each condition. For example:
            {
                "prepend": ["description"],
                "sum": ["genre", "bpm"],
                "cross": ["description"],
            }
        cross_attention_pos_emb (bool, optional): Use positional embeddings in cross attention.
        cross_attention_pos_emb_scale (int): Scale for positional embeddings in cross attention if used.
    """
    FUSING_METHODS = ["sum", "prepend", "cross", "input_interpolate"]

    def __init__(self, fuse2cond: tp.Dict[str, tp.List[str]], cross_attention_pos_emb: bool = False,
                 cross_attention_pos_emb_scale: float = 1.0):
        super().__init__()
        assert all(
            [k in self.FUSING_METHODS for k in fuse2cond.keys()]
        ), f"Got invalid fuse method, allowed methods: {self.FUSING_METHODS}"
        self.cross_attention_pos_emb = cross_attention_pos_emb
        self.cross_attention_pos_emb_scale = cross_attention_pos_emb_scale
        self.fuse2cond: tp.Dict[str, tp.List[str]] = fuse2cond
        self.cond2fuse: tp.Dict[str, str] = {}
        for fuse_method, conditions in fuse2cond.items():
            for condition in conditions:
                self.cond2fuse[condition] = fuse_method

    def forward(
        self,
        input: torch.Tensor,
        conditions: tp.Dict[str, ConditionType]
    ) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        """Fuse the conditions to the provided model input.

        Args:
            input (torch.Tensor): Transformer input.
            conditions (dict[str, ConditionType]): Dict of conditions.
        Returns:
            tuple[torch.Tensor, torch.Tensor]: The first tensor is the transformer input
                after the conditions have been fused. The second output tensor is the tensor
                used for cross-attention or None if no cross attention inputs exist.
        """
        # import pdb; pdb.set_trace()
        B, T, _ = input.shape

        if 'offsets' in self._streaming_state:
            first_step = False
            offsets = self._streaming_state['offsets']
        else:
            first_step = True
            offsets = torch.zeros(input.shape[0], dtype=torch.long, device=input.device)

        assert set(conditions.keys()).issubset(set(self.cond2fuse.keys())), \
            f"given conditions contain unknown attributes for fuser, " \
            f"expected {self.cond2fuse.keys()}, got {conditions.keys()}"
        cross_attention_output = None
        prepend_input = input[:, :0]
        for cond_type, (cond, cond_mask) in conditions.items():
            op = self.cond2fuse[cond_type]
            if op == 'sum': 
                input += cond
            elif op == 'input_interpolate':
                cond = einops.rearrange(cond, "b t d -> b d t")
                cond = F.interpolate(cond, size=input.shape[1])
                input += einops.rearrange(cond, "b d t -> b t d")
            elif op == 'prepend':
                prepend_input = torch.cat([cond.to(input.dtype), prepend_input], dim=1) 
                # NOTE 这里cond应该在后,这样顺序才符合配置文件,否则为逆序
                # 但是之前实验是这样的为了保持一致就没改
            elif op == 'cross':
                if cross_attention_output is not None:
                    cross_attention_output = torch.cat([cross_attention_output, cond], dim=1)
                else:
                    cross_attention_output = cond
            else:
                raise ValueError(f"unknown op ({op})")

        if self.cross_attention_pos_emb and cross_attention_output is not None:
            positions = torch.arange(
                cross_attention_output.shape[1],
                device=cross_attention_output.device
            ).view(1, -1, 1)
            pos_emb = create_sin_embedding(positions, cross_attention_output.shape[-1])
            cross_attention_output = cross_attention_output + self.cross_attention_pos_emb_scale * pos_emb

        if first_step:
            input = torch.cat([prepend_input, input], dim=1)
        if self._is_streaming:
            self._streaming_state['offsets'] = offsets + T

        return input, cross_attention_output
