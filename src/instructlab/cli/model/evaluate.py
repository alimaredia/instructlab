# SPDX-License-Identifier: Apache-2.0

# pylint: disable=ungrouped-imports
# Standard
import logging
import pathlib

# Third Party
import click
import instructlab.eval.ragas as ragas_eval

# First Party
from instructlab import clickext
from instructlab.model.backends import backends
from instructlab.model.evaluate import Benchmark, evaluate_model

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--model",
    type=click.STRING,
    cls=clickext.ConfigOption,
)
@click.option(
    "--base-model",
    type=click.STRING,
    cls=clickext.ConfigOption,
    required=True,  # default from config
)
@click.option(
    "--benchmark",
    type=click.Choice([m.value for m in Benchmark.__members__.values()]),
    required=True,
    help="Benchmarks to run during evaluation",
)
@click.option(
    "--judge-model",
    type=click.STRING,
    cls=clickext.ConfigOption,
    config_sections="mt_bench",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    cls=clickext.ConfigOption,
    config_sections="mt_bench",
)
@click.option(
    "--max-workers",
    type=click.STRING,
    cls=clickext.ConfigOption,
    config_sections="mt_bench",
)
@click.option(
    "--taxonomy-path",
    type=click.Path(),
    cls=clickext.ConfigOption,
    config_sections="mt_bench_branch",
)
@click.option(
    "--branch",
    type=click.STRING,
    cls=clickext.ConfigOption,
)
@click.option(
    "--base-branch",
    type=click.STRING,
    cls=clickext.ConfigOption,
)
@click.option(
    "--few-shots",
    type=click.INT,
    cls=clickext.ConfigOption,
    config_sections="mmlu",
)
@click.option(
    "--batch-size",
    type=click.STRING,
    cls=clickext.ConfigOption,
    config_sections="mmlu",
)
@click.option(
    "--tasks-dir",
    type=click.Path(),
    cls=clickext.ConfigOption,
    config_sections="mmlu_branch",
)
@click.option(
    "--gpus",
    type=click.IntRange(min=0),
    help="Number of GPUs to utilize for evaluation (not applicable to llama-cpp)",
)
@click.option(
    "--merge-system-user-message",
    is_flag=True,
    help="Indicates whether to merge system and user message for mt_bench and mt_bench_branch (required for Mistral based judges)",
)
@click.option(
    "--backend",
    type=click.Choice(tuple(backends.SUPPORTED_BACKENDS)),
    help="Serving backend to use for the model and base model (if applicable) during evaluation. Options are vllm and llama-cpp.",
)
@click.option(
    "--judge-backend",
    type=click.Choice(tuple(backends.SUPPORTED_BACKENDS)),
    help="Serving backend to use for the judge model for during mt_bench or mt_bench_branch evaluation. Options are vllm and llama-cpp.",
)
@click.option(
    "--tls-insecure",
    is_flag=True,
    help="Disable TLS verification for model serving.",
)
@click.option(
    "--tls-client-cert",
    type=click.Path(),
    default="",
    show_default=True,
    help="Path to the TLS client certificate to use for model serving.",
)
@click.option(
    "--tls-client-key",
    type=click.Path(),
    default="",
    show_default=True,
    help="Path to the TLS client key to use for model serving.",
)
@click.option(
    "--tls-client-passwd",
    type=click.STRING,
    default="",
    help="TLS client certificate password for model serving.",
)
@click.option(
    "--enable-serving-output",
    is_flag=True,
    help="Print serving engine logs.",
)
@click.option(
    "--input-questions",
    type=click.Path(exists=True, path_type=pathlib.Path),
    default=None,
    help="Path to the questions and reference answers the model will be evaluating in the DK-Bench evaluation",
)
@click.option(
    "--output-file-formats",
    type=click.STRING,
    default="jsonl",
    show_default=True,
    help="Comma-separated list of file formats for results of the DK-Bench evaluation. Valid options are csv, jsonl, and xlsx. If this option is not provided the results are written as a .jsonl file",
)
@click.option(
    "--system-prompt",
    type=click.STRING,
    default=ragas_eval._DEFAULT_SYSTEM_PROMPT,
    help="Prompt for the model when getting responses in the DK-Bench evaluation",
)
@click.option(
    "--temperature",
    type=click.FLOAT,
    default=0.0,
    help="Temperature for the model when getting responses in the DK-Bench evaluation",
)
@click.option(
    "--model-name",
    type=click.STRING,
    default=None,
    help="Model name of model getting responses and being evaluated in the DK-Bench evaluation",
)
@click.option(
    "--judge-model-name",
    type=click.STRING,
    default="gpt-4o",
    help="Name of judge model doing evaluation in the DK-Bench evaluation. Judge model name must be an OpenAI model",
)
@click.pass_context
@clickext.display_params
def evaluate(
    ctx,
    model,
    base_model,
    benchmark,
    judge_model,
    output_dir,
    max_workers: str | int,
    taxonomy_path,
    branch,
    base_branch,
    few_shots,
    batch_size: str | int,
    tasks_dir,
    gpus,
    merge_system_user_message,
    backend,
    judge_backend,
    tls_insecure,  # pylint: disable=unused-argument
    tls_client_cert,  # pylint: disable=unused-argument
    tls_client_key,  # pylint: disable=unused-argument
    tls_client_passwd,  # pylint: disable=unused-argument
    enable_serving_output,
    input_questions,
    output_file_formats,
    system_prompt,
    temperature,
    model_name,
    judge_model_name,
) -> None:
    """Evaluates a trained model"""
    try:
        evaluate_model(
            ctx,
            model,
            base_model,
            benchmark,
            judge_model,
            output_dir,
            max_workers,
            taxonomy_path,
            branch,
            base_branch,
            few_shots,
            batch_size,
            tasks_dir,
            gpus,
            merge_system_user_message,
            backend,
            judge_backend,
            tls_insecure,
            tls_client_cert,
            tls_client_key,
            tls_client_passwd,
            enable_serving_output,
            input_questions,
            output_file_formats,
            system_prompt,
            temperature,
            model_name,
            judge_model_name,
        )
    except Exception as e:
        logger.error(f"An error occurred during evaluation: {str(e)}")
        raise click.exceptions.Exit(1) from e
