# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any, Optional
import logging
from os import path

# Third Party
# pylint: disable=ungrouped-imports
from instructlab.training import (
    DistributedBackend,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    StrictStr,
    field_validator,
    model_validator,
)

# Local
from ..defaults import (
    CONFIG_VERSION,
    DEFAULTS,
    LOG_FORMAT,
)

class _general(BaseModel):
    """Class describing various top-level configuration options for all commands."""

    # model configuration
    model_config = ConfigDict(extra="ignore")

    # additional fields with defaults
    log_level: StrictStr = Field(default="INFO", description="Log level for logging.")
    debug_level: int = Field(default=0, description="Debug level for logging.")
    log_format: StrictStr = Field(
        default=LOG_FORMAT,
        description="Log format. https://docs.python.org/3/library/logging.html#logrecord-attributes",
        validate_default=True,
    )

    @field_validator("log_level")
    def validate_log_level(cls, v):
        # TODO: remove 'valid_levels' once we switch to support Python 3.11+ and call
        # "logging.getLevelNamesMapping()" instead
        valid_levels = [
            "DEBUG",
            "INFO",
            "WARNING",
            "WARN",
            "FATAL",
            "CRITICAL",
            "ERROR",
            "NOTSET",
        ]
        if v.upper() not in valid_levels:
            raise ValueError(
                f"'{v}' is not a valid log level name. valid levels: {valid_levels}"
            )
        return v.upper()

    @field_validator("log_format")
    @classmethod
    def validate_log_format(cls, log_format):
        try:
            logging.PercentStyle(log_format).validate()
            return log_format
        except ValueError as e:
            raise ValueError(
                f"\nFailed to configure log format: {e}\n"
                "Have you specified a valid log format?\n"
                "Consider reading: https://docs.python.org/3/library/logging.html#logrecord-attributes"
            ) from e

    @model_validator(mode="after")
    def after_debug_level(self):
        # set debug level when log level is DEBUG
        if self.log_level == "DEBUG" and self.debug_level == 0:
            self.debug_level = 1
        return self


class _chat(BaseModel):
    """Class describing configuration of the 'chat' sub-command."""

    # model configuration
    model_config = ConfigDict(extra="ignore")
    model: str = Field(
        default_factory=lambda: DEFAULTS.DEFAULT_CHAT_MODEL,
        description="Model to be used for chatting with.",
    )
    # additional fields with defaults
    vi_mode: bool = Field(default=False, description="Enable vim keybindings for chat.")
    visible_overflow: bool = Field(
        default=True,
        description="Renders vertical overflow if enabled, displays ellipses otherwise.",
    )
    context: str = Field(
        default="default",
        description="Predefined setting or environment that influences the behavior and responses of the chat assistant. Each context is associated with a specific prompt that guides the assistant on how to respond to user inputs. Available contexts: default, cli_helper.",
    )
    session: Optional[str] = Field(
        default=None, description="Filepath of a dialog session file."
    )
    # use a lambda to avoid caching
    logs_dir: str = Field(
        default_factory=lambda: DEFAULTS.CHATLOGS_DIR,
        description="Directory where chat logs are stored.",
    )
    temperature: float = Field(
        default=1.0,
        description="Controls the randomness of the model's responses. Lower values make the output more deterministic, while higher values produce more random results.",
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description="The maximum number of tokens that can be generated in the chat completion. Be aware that larger values use more memory.",
    )

    def __init__(self, **data):
        super().__init__(**data)
        if 'model' in data:
            if DEFAULTS.MODELS_DIR not in data['model']:
                self.model = path.join(DEFAULTS.MODELS_DIR, data['model'])


class _serve_vllm(BaseModel):
    """Class describing configuration of vLLM serving backend."""

    llm_family: str = Field(
        default="",  # TODO: convert to None and use a pattern to validate
        description="Large Language Model Family",
        examples=["merlinite", "granite"],
    )
    max_startup_attempts: int | None = Field(
        default=120,
        description="Maximum number of attempts to start the vLLM server.",
    )
    gpus: Optional[int] = Field(default=None, description="Number of GPUs to use.")
    # arguments to pass into vLLM process
    vllm_args: list[str] | None = Field(
        default_factory=list,
        description="vLLM specific arguments. All settings can be passed as a list of strings, see: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html",
        examples=[
            ["--dtype", "auto"],
            ["--lora-alpha", "32"],
        ],
    )


class _serve_llama_cpp(BaseModel):
    """Class describing configuration of llama-cpp serving backend."""

    gpu_layers: int = Field(
        default=-1,
        description="Number of model layers to offload to GPU. -1 means all layers.",
    )
    max_ctx_size: PositiveInt = Field(
        default=4096,
        description="Maximum number of tokens that can be processed by the model.",
    )
    llm_family: str = Field(
        default="",  # TODO: convert to None and use a pattern to validate
        description="Large Language Model Family",
        examples=["merlinite", "granite"],
    )


class _serve(BaseModel):
    """Class describing configuration of the 'serve' sub-command."""

    # model configuration
    model_config = ConfigDict(extra="forbid", protected_namespaces=())
    # vLLM configuration
    vllm: _serve_vllm = Field(
        default_factory=_serve_vllm,
        description="vLLM serving settings.",
    )
    # llama-cpp configuration
    llama_cpp: _serve_llama_cpp = Field(
        default_factory=_serve_llama_cpp,
        description="llama-cpp serving settings.",
    )
    model_path: StrictStr = Field(
        default_factory=lambda: DEFAULTS.DEFAULT_CHAT_MODEL,
        description="Directory where model to be served is stored.",
    )
    # additional fields with defaults
    host_port: StrictStr = Field(
        default="127.0.0.1:8000", description="Host and port to serve on."
    )
    chat_template: Optional[str] = Field(
        default=None,
        description="Chat template to supply to the model. Possible values: 'auto'(default), 'tokenizer', a path to a jinja2 file.",
        examples=[
            "auto",
            "tokenizer",
            "A filesystem path expressing the location of a custom template",
        ],
    )
    # we don't set a default value here since it's auto-detected
    backend: Optional[str] = Field(
        default=None,
        description="Serving backend to use to host the model.",
        examples=["vllm", "llama-cpp"],
        pattern="vllm|llama-cpp",
    )

    def api_base(self):
        """Returns server API URL, based on the configured host and port"""
        return f"http://{self.host_port}/v1"

    def __init__(self, **data):
        super().__init__(**data)
        if 'model_path' in data:
            if DEFAULTS.MODELS_DIR not in data['model_path']:
                self.model_path = path.join(DEFAULTS.MODELS_DIR, data['model_path'])

class _generate(BaseModel):
    """Class describing configuration of the 'generate' sub-command."""

    # model configuration
    model_config = ConfigDict(extra="ignore")
    pipeline: Optional[str] = Field(
        default=DEFAULTS.SDG_PIPELINE,
        description="Data generation pipeline to use. Available: 'simple', 'full', or a valid path to a directory of pipeline workflow YAML files. Note that 'full' requires a larger teacher model, Mixtral-8x7b.",
    )
    model: StrictStr = Field(
        default_factory=lambda: DEFAULTS.DEFAULT_TEACHER_MODEL,
        description="Teacher model that will be used to synthetically generate training data.",
    )
    taxonomy_path: StrictStr = Field(
        default_factory=lambda: DEFAULTS.TAXONOMY_DIR,
        description="Directory where taxonomy is stored and accessed from.",
    )
    taxonomy_base: StrictStr = Field(
        default=DEFAULTS.TAXONOMY_BASE,
        description="Branch of taxonomy used to calculate diff against.",
    )
    # additional fields with defaults
    teacher: _serve = Field(
        default_factory=lambda: _serve(model_path=DEFAULTS.DEFAULT_TEACHER_MODEL),
        description="Teacher configuration",
    )
    num_cpus: PositiveInt = Field(
        default=DEFAULTS.NUM_CPUS,
        description="Number of CPU cores to use for generation.",
    )
    chunk_word_count: PositiveInt = Field(
        default=DEFAULTS.CHUNK_WORD_COUNT,
        description="Maximum number of words per chunk.",
    )
    # DEPRECATED: see sdg_scale_factor instead
    # Left in place so that we can still detect and give a warning if its
    # specified in an old configuration file.
    num_instructions: Optional[int] = Field(
        default=-1,
        description="Number of instructions to use",
        deprecated="see 'sdg_scale_factor' instead",
        exclude=True,
    )
    sdg_scale_factor: Optional[PositiveInt] = Field(
        default=DEFAULTS.SDG_SCALE_FACTOR,
        description="The total number of instructions to be generated.",
    )
    output_dir: StrictStr = Field(
        default_factory=lambda: DEFAULTS.DATASETS_DIR,
        description="Directory where generated datasets are stored.",
    )
    # TODO: remove this? It's not used anywhere, was removed by 19b9f4794f79ef81578c00c901bac3ee9db8c046
    # related issue: https://github.com/instructlab/instructlab/issues/2261
    seed_file: StrictStr = Field(
        description="Path to seed file to be used for generation.",
        default_factory=lambda: DEFAULTS.SEED_FILE,
        deprecated=True,
    )


class _mmlu(BaseModel):
    """Class describing configuration of MMLU evaluation benchmark."""

    few_shots: int = Field(
        default=5,
        description="Number of question-answer pairs provided in the context preceding the question used for evaluation.",
    )
    batch_size: str | int = Field(
        default="auto",
        description="Batch size for evaluation. Valid values are a positive integer or 'auto' to select the largest batch size that will fit in memory.",
    )


class _mtbench(BaseModel):
    """Class describing configuration of MTBench evaluation benchmark."""

    judge_model: str = Field(
        default_factory=lambda: DEFAULTS.JUDGE_MODEL_MT,
        description="Judge model for mt_bench and mt_bench_branch.",
    )
    output_dir: str = Field(
        default_factory=lambda: DEFAULTS.EVAL_DATA_DIR,
        description="Directory where evaluation results are stored.",
    )
    max_workers: str | int = Field(
        default="auto",
        description="Number of workers to use for evaluation with mt_bench or mt_bench_branch. Must be a positive integer or 'auto'.",
    )


class _mtbenchbranch(BaseModel):
    """Class describing configuration of MTBenchBranch evaluation benchmark."""

    taxonomy_path: str = Field(
        default_factory=lambda: DEFAULTS.TAXONOMY_DIR,
        description="Path to where base taxonomy is stored.",
    )


class _mmlubranch(BaseModel):
    """Class describing configuration of MMLUBranch evaluation benchmark."""

    tasks_dir: str = Field(
        default_factory=lambda: DEFAULTS.DATASETS_DIR,
        description="Directory where custom MMLU tasks are stored.",
    )


class _evaluate(BaseModel):
    """Class describing configuration of the 'evaluate' sub-command."""

    # model configuration
    model_config = ConfigDict(extra="ignore", protected_namespaces=())
    model: Optional[str] = Field(
        default=None,
        description="Model to be evaluated",
    )
    base_model: str = Field(
        default=DEFAULTS.MODEL_REPO,
        description="Base model to compare with 'model' for mt_bench_branch and mmlu_branch.",
    )
    branch: Optional[str] = Field(
        default=None,
        description="Taxonomy branch containing custom skills/knowledge that should be used for evaluation runs.",
    )
    base_branch: Optional[str] = Field(default=None, description="Base taxonomy branch")
    gpus: Optional[int] = Field(
        default=None, description="Number of GPUs to use for running evaluation."
    )
    mmlu: _mmlu = Field(
        default_factory=_mmlu,
        description="MMLU benchmarking settings",
    )
    mmlu_branch: _mmlubranch = Field(
        default_factory=_mmlubranch,
        description="Settings to run MMLU against a branch of taxonomy containing custom skills/knowledge used for training.",
    )
    mt_bench: _mtbench = Field(
        default_factory=_mtbench,
        description="Multi-turn benchmarking settings for skills.",
    )
    mt_bench_branch: _mtbenchbranch = Field(
        default_factory=_mtbenchbranch,
        description="Settings to run MT-Bench against a branch of taxonomy containing custom skills/knowledge used for training",
    )


class _train(BaseModel):
    """Class describing configuration of the 'train' sub-command."""

    # model configuration
    model_config = ConfigDict(
        extra="ignore",
        protected_namespaces=(),
        use_enum_values=True,  # populate models with the value property of enums, rather than the raw enum.
    )
    pipeline: str = Field(
        default="full",
        description="Training pipeline to use. Simple is for systems with limited resources, full is for more capable consumer systems (64 GB of RAM), and accelerated is for systems with a dedicated GPU.",
        examples=["simple", "full", "accelerated"],
        pattern="simple|full|accelerated",
    )
    model_path: str = Field(
        default=DEFAULTS.MODEL_REPO,
        description="Directory where the model to be trained is stored.",
    )
    device: str = Field(
        default="cpu",
        description="PyTorch device to use. Use 'cpu' for 'simple' and 'full' training on Linux. Use 'mps' for 'full' training on MacOS Metal Performance Shader. Use 'cuda' for Nvidia CUDA / AMD ROCm GPUs. Use 'hpu' for Intel Gaudi GPUs.",
        examples=["cpu", "mps", "cuda", "hpu"],
        pattern="cpu|mps|cuda|hpu",
    )
    data_path: str = Field(
        default_factory=lambda: DEFAULTS.DATASETS_DIR,
        description="For the training library (primary training method), this specifies the path to the dataset file. For legacy training (MacOS/Linux), this specifies the path to the directory.",
    )
    ckpt_output_dir: str = Field(
        default_factory=lambda: DEFAULTS.CHECKPOINTS_DIR,
        description="Directory where periodic training checkpoints are stored.",
    )
    data_output_dir: str = Field(
        default_factory=lambda: DEFAULTS.INTERNAL_DIR,
        description="Directory where the processed training data is stored (post filtering/tokenization/masking).",
    )
    max_seq_len: int = Field(
        default=4096,
        description="Maximum sequence length to be included in the training set. Samples exceeding this length will be dropped.",
    )
    max_batch_len: int = Field(
        default=5000,
        description="Maximum tokens per gpu for each batch that will be handled in a single step. If running into out-of-memory errors, this value can be lowered but not below the `max_seq_len`.",
    )
    num_epochs: int = Field(
        default=10, description="Number of epochs to run training for."
    )
    effective_batch_size: int = Field(
        default=64,
        description="The number of samples in a batch that the model should see before its parameters are updated.",
    )
    save_samples: int = Field(
        default=250000,
        description="Number of samples the model should see before saving a checkpoint.",
    )
    checkpoint_at_epoch: bool = Field(
        default=True, description="Save a checkpoint at the end of each epoch."
    )
    deepspeed_cpu_offload_optimizer: bool = Field(
        default=False, description="Allow CPU offload for deepspeed optimizer."
    )
    fsdp_cpu_offload_optimizer: bool = Field(
        default=False, description="Allow CPU offload for FSDP optimizer."
    )
    distributed_backend: DistributedBackend = Field(
        default=DistributedBackend.DEEPSPEED,
        description="Pick a distributed training backend framework for GPU accelerated full fine-tuning.",
        validate_default=True,  # ensures that the 'use_enum_values' flag takes effect on the default value
    )
    lora_rank: int | None = Field(
        default=0,
        description="Rank of low rank matrices to be used during training.",
    )
    lora_quantize_dtype: str | None = Field(
        default="nf4",
        description="The data type for quantization in LoRA training. Valid options are 'None' and 'nf4'.",
        examples=["nf4"],
    )
    is_padding_free: bool = Field(
        default=False,
        description="Boolean to indicate if the model being trained is a padding-free transformer model such as Granite.",
    )
    nproc_per_node: int = Field(
        default=1,
        description="Number of GPUs to use for training. This value is not supported in legacy training or MacOS.",
    )
    disable_flash_attn: Optional[bool] = Field(
        default=False,
        description="Whether or not we should disable the use of flash-attention during training. This is useful when using older GPUs.",
    )
    additional_args: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional arguments to pass to the training script. These arguments are passed as key-value pairs to the training script.",
    )
    # additional training configuration for
    # lab-multiphase training.
    # TODO: could move into its own object.
    # Not strictly necessary for a correct training object.
    phased_phase1_num_epochs: int | None = Field(
        default=7,
        gt=0,
        description="Number of epochs to run training for during phase1 (experimentally optimal number is 7).",
    )
    # anything greater than 0 enables samples_per_save for the phase.
    phased_phase1_samples_per_save: int = Field(
        default=0,
        ge=0,
        description="Number of samples the model should see before saving a checkpoint during phase1. Disabled when set to 0.",
    )
    phased_phase1_effective_batch_size: int | None = Field(
        default=128,
        description="Phased phase1 effective batch size.",
    )
    phased_phase2_num_epochs: int | None = Field(
        default=10,
        gt=0,
        description="Number of epochs to run training for during phase2.",
    )
    # phased_phase2_samples_per_save is disabled when the value is 0.
    # anything greater than 0 enables samples_per_save for the phase.
    phased_phase2_samples_per_save: int = Field(
        default=0,
        ge=0,
        description="Number of samples the model should see before saving a checkpoint during phase2. Disabled when set to 0.",
    )
    phased_phase2_effective_batch_size: int | None = Field(
        default=3840, description="Phased phase2 effective batch size."
    )
    phased_mt_bench_judge: str | None = Field(
        default_factory=lambda: DEFAULTS.DEFAULT_JUDGE_MODEL,
        description="Judge model path for phased MT-Bench evaluation.",
    )
    phased_base_dir: str | None = Field(
        default_factory=lambda: DEFAULTS.PHASED_DIR,
        description="Base directory for organization of end-to-end intermediate outputs.",
    )
    training_journal: str | None = Field(
        default=None,
        description="Optional path to a yaml file that tracks the progress of multiphase training.",
    )

    deepspeed_cpu_offload_optimizer_pin_memory: bool = Field(
        default=False,
        description="TODO",
    )

    deepspeed_cpu_offload_optimizer_ratio: int = Field(
        default=1,
        description="TODO",
    )

    learning_rate: float = Field(
        default=2.0e-05,
        description="TODO",
    )

    lora_alpha: int = Field(
        default=32,
        description="TODO",
    )

    lora_dropout: float = Field(
        default=0.1,
        description="TODO",
    )

    lora_target_modules: list[str] = Field(
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
        description="TODO",
    )

    nnodes: int = Field(
        default=1,
        description="TODO",
    )

    node_rank: int = Field(
        default=0,
        description="TODO",
    )

    random_seed: int = Field(
        default=42,
        description="TODO",
    )

    rdzv_endpoint: str = Field(
        default="127.0.0.1:12222",
        description="TODO",
    )

    rdzv_id: int = Field(
        default=123,
        description="TODO",
    )

    warmup_steps: int = Field(
        default=25,
        description="TODO",
    )


class _model(BaseModel):
    """Class describing configuration of the 'model' commands."""

    # pydantic configuration
    model_config = ConfigDict(extra="forbid")

    # chat configuration
    chat: _chat = Field(
        default_factory=_chat, description="Chat configuration section."
    )

    # serve configuration (includes both llama-cpp and vLLM configuration)
    serve: _serve = Field(
        default_factory=_serve, description="Serve configuration section."
    )

class _info(BaseModel):
    """Class describing configuration of the 'system info' sub-command."""

    # model configuration
    model_config = ConfigDict(extra="forbid")
    new_var: str = Field(
        default="foo",
        description="throw away variable for illustrative pusposes",
    )

class _system(BaseModel):
    """Class describing configuration of the 'system' commands."""

    # pydantic configuration
    model_config = ConfigDict(extra="forbid")

    # chat configuration
    info: _info = Field(
        default_factory=_info, description="System info configuration section."
    )

class DefaultConfig(BaseModel):
    """Configuration for the InstructLab CLI.
    Config options are defined by the respective subclasses and are loaded into a single 'Config' object here
    Instantation of this object should be done via 'get_default_config()'
    Note that values here can be overriden by a users 'config.yaml' or command line overrides in some cases
    """

    # model commands configuration
    model: _model = Field(
        default_factory=_model, description="Model commands configuration section."
    )

    # system commands configuration
    system: _system = Field(
        default_factory=_system, description="System commands configuration section."
    )
    # generate configuration
    generate: _generate = Field(
        default_factory=_generate, description="Generate configuration section."
    )
    # train configuration
    train: _train = Field(
        default_factory=_train, description="Train configuration section."
    )
    # evaluate configuration
    evaluate: _evaluate = Field(
        default_factory=_evaluate, description="Evaluate configuration section."
    )
    # additional fields with defaults
    general: _general = Field(
        default_factory=_general, description="General configuration section."
    )
    # model configuration
    model_config = ConfigDict(extra="forbid")
    # config file versioning
    version: str = Field(
        default=CONFIG_VERSION,
        description="Configuration file structure version.",
        frozen=True,  # don't allow this to be changed anywhere in the code
    )
