from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional, Type, TypeVar, Union, get_args, get_origin
from ast import literal_eval

import yaml

T = TypeVar("T")


@dataclass
class BaseConfig:
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def merge_from_dict(self, values: dict[str, Any]) -> None:
        for key, value in values.items():
            if not hasattr(self, key):
                raise KeyError(f"{self.__class__.__name__} has no field {key!r}")
            setattr(self, key, value)


@dataclass
class SystemConfig(BaseConfig):
    seed: int
    work_dir: str


@dataclass
class WandbConfig(BaseConfig):
    enabled: bool
    project: str
    entity: Optional[str]
    name: Optional[str]
    tags: Optional[list[str]]
    group: Optional[str]
    notes: Optional[str]
    mode: Optional[str]


@dataclass
class LRScheduleConfig(BaseConfig):
    name: str
    warmup_iters: int
    max_iters: int
    min_lr: float


@dataclass
class TrainerConfig(BaseConfig):
    device: str
    num_workers: int
    max_iters: Optional[int]
    batch_size: int
    learning_rate: float
    betas: tuple[float, float]
    weight_decay: float
    grad_norm_clip: float
    eval_interval: Optional[int]
    eval_batches: Optional[int]
    eval_batch_size: Optional[int]
    auto_batch_size: bool
    auto_batch_size_start: Optional[int]
    auto_batch_size_factor: int
    auto_batch_size_max: int
    lr_schedule: LRScheduleConfig
    wandb: WandbConfig


@dataclass
class ModelConfig(BaseConfig):
    model_type: Optional[str]
    n_layer: Optional[int]
    n_head: Optional[int]
    n_embd: Optional[int]
    vocab_size: Optional[int]
    block_size: Optional[int]
    embd_pdrop: float
    resid_pdrop: float
    attn_pdrop: float


@dataclass
class AdderDataConfig(BaseConfig):
    ndigit: int


@dataclass
class CharDataConfig(BaseConfig):
    block_size: int
    input_path: str
    train_split: float


@dataclass
class AdderConfig(BaseConfig):
    system: SystemConfig
    data: AdderDataConfig
    model: ModelConfig
    trainer: TrainerConfig


@dataclass
class CharGPTConfig(BaseConfig):
    system: SystemConfig
    data: CharDataConfig
    model: ModelConfig
    trainer: TrainerConfig


def _convert_value(value: Any, target_type: Type[Any]) -> Any:
    origin = get_origin(target_type)

    if origin is None:
        if is_dataclass(target_type):
            if not isinstance(value, dict):
                raise TypeError(f"expected dict for {target_type.__name__}, got {type(value)}")
            return _from_dict(target_type, value)
        if target_type is Any:
            return value
        return target_type(value)

    if origin is list:
        (inner_type,) = get_args(target_type)
        return [_convert_value(v, inner_type) for v in value]

    if origin is tuple:
        inner_types = get_args(target_type)
        if len(inner_types) == 2 and inner_types[1] is Ellipsis:
            return tuple(_convert_value(v, inner_types[0]) for v in value)
        if len(inner_types) != len(value):
            raise TypeError(f"expected {len(inner_types)} values, got {len(value)}")
        return tuple(_convert_value(v, t) for v, t in zip(value, inner_types))

    if origin is dict:
        key_type, val_type = get_args(target_type)
        return {
            _convert_value(k, key_type): _convert_value(v, val_type)
            for k, v in value.items()
        }

    if origin is Union:
        if value is None and type(None) in get_args(target_type):
            return None
        for t in get_args(target_type):
            if t is type(None):
                continue
            try:
                return _convert_value(value, t)
            except Exception:
                continue
        raise TypeError(f"cannot convert {value!r} to {target_type}")

    raise TypeError(f"unsupported type {target_type}")


def _from_dict(cls: Type[T], data: dict[str, Any]) -> T:
    if not is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")
    field_names = {f.name for f in fields(cls)}
    missing = [f.name for f in fields(cls) if f.name not in data]
    if missing:
        raise KeyError(f"missing config fields for {cls.__name__}: {missing}")
    extra = set(data.keys()) - field_names
    if extra:
        raise KeyError(f"unexpected config fields for {cls.__name__}: {sorted(extra)}")

    kwargs = {}
    for f in fields(cls):
        kwargs[f.name] = _convert_value(data[f.name], f.type)
    return cls(**kwargs)


def _apply_overrides(config: dict[str, Any], overrides: list[str]) -> None:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        key, raw = item.split("=", 1)
        try:
            value = literal_eval(raw)
        except (ValueError, SyntaxError):
            value = raw
        path = key.split(".")
        cur = config
        for p in path[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[path[-1]] = value


def load_config(path: str | Path, cls: Type[T], overrides: Optional[list[str]] = None) -> T:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"config at {path} must be a mapping")
    if overrides:
        _apply_overrides(data, overrides)
    return _from_dict(cls, data)
