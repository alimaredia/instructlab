# SPDX-License-Identifier: Apache-2.0

# pylint: disable=ungrouped-imports
# Standard
from datetime import datetime
from typing import List, Tuple
import enum
import logging
import os
import pathlib

# Third Party
from instructlab.eval.mt_bench_common import (
    get_openai_client as get_local_openai_client,
)
from instructlab.eval.ragas import ModelConfig, RagasEvaluator
from openai import OpenAI, OpenAIError
from ragas.evaluation import EvaluationResult  # type: ignore
import click
import pandas as pd

# Local
from ..client_utils import ClientException, http_client
from ..configuration import DEFAULTS
from ..utils import is_model_gguf, is_model_safetensors
from .evaluate import get_model_name as get_local_model_name
from .evaluate import launch_server

logger = logging.getLogger(__name__)


class IOFileType(enum.Enum):
    CSV = "csv"
    JSONL = "jsonl"
    XLSX = "xlsx"


def validate_output_file_formats(file_formats: List[str]):
    for file_format in file_formats:
        valid_formats = {item.value for item in IOFileType}
        if file_format not in valid_formats:
            raise ValueError(
                f"File format {file_format} is not a valid output format. Format must be one of {valid_formats}"
            )


def model_is_local_and_valid(model: str) -> bool:
    valid_model = False
    if model is not None and os.path.exists(model):
        model_path = pathlib.Path(model)
        if model_path.is_dir():
            valid_model = is_model_safetensors(model_path)
        elif model_path.is_file():
            valid_model = is_model_gguf(model_path)
    return valid_model


def get_endpoint_model_name(openai_client: OpenAI, model_name: str | None) -> str:
    try:
        models = openai_client.models.list()
    except OpenAIError as exc:
        raise ClientException(
            f"Connection Error {exc}. Unable to list models at"
        ) from exc

    endpoint_model_names = [model.id for model in models.data]

    if model_name is not None:
        if model_name not in endpoint_model_names:
            raise ValueError(
                f"Model named {model_name} is not listed at endpoint. Models listed at endpoint are: {endpoint_model_names}"
            )
    # No model name input by user
    else:
        if len(endpoint_model_names) == 1:
            model_name = endpoint_model_names[0]
        else:
            raise ValueError(
                f"More than one model listed at endpoint {openai_client.base_url}. To evaluate a model at this endpoint pass in one of following models listed: {endpoint_model_names}"
            )

    logger.debug("Model %s found at endpoint %s", model_name, openai_client.base_url)
    return model_name


def create_results_file_name(
    file_format: str, output_dir: str, timestamp: str, model_name: str
) -> str:
    if IOFileType.CSV.value == file_format:
        file_type = "csv"
    elif IOFileType.XLSX.value == file_format:
        file_type = "xlsx"
    elif IOFileType.JSONL.value == file_format:
        file_type = "jsonl"
    else:
        raise ValueError("File format is not one of: csv, xlsx, jsonl")

    output_path = pathlib.Path(output_dir)
    if not output_path.exists() or not output_path.is_dir():
        output_dir = DEFAULTS.EVAL_DATA_DIR
        logger.debug(
            "Output dir provided is not a directory or does not exist, writing results to default evaluation output directory %s",
            output_dir,
        )

    return f"{output_dir}/responses-{model_name}-{timestamp}.{file_type}"


def make_run_dir(output_dir: str) -> str:
    now = datetime.now()
    timestamp = now.isoformat()
    # remove any training slashes
    output_dir = os.path.normpath(output_dir)
    run_dir = f"{output_dir}/job-{timestamp}"

    path_does_not_exist = not os.path.exists(output_dir)
    path_is_not_dir = not os.path.isdir(output_dir)
    if path_does_not_exist or path_is_not_dir:
        run_dir = f"{DEFAULTS.EVAL_DATA_DIR}/job-{timestamp}"
        logger.debug(
            "User provided output directory path %s does not exist or is not a directory. Creating directory for results in %s",
            output_dir,
            DEFAULTS.EVAL_DATA_DIR,
        )

    os.makedirs(run_dir, exist_ok=True)
    logger.debug("Created directory for results at %s", run_dir)
    return run_dir


def print_results(result: EvaluationResult, results_files: List[str], model_name: str):
    print("\n")
    print("# DK-BENCH REPORT")
    print(f"\n## MODEL: {model_name}\n")
    total_score = 0
    for i, score in enumerate(result.scores):
        print(f"Question #{i+1}:     {score['domain_specific_rubrics']}/5")
        total_score += score["domain_specific_rubrics"]

    average = total_score / len(result.scores)
    print("----------------------------")
    print(f"Average Score:   {average:.2f}/5")
    print(f"Total Score:     {total_score}/{len(result.scores)*5}\n")

    print("Responses and scores written to:")
    for file in results_files:
        print(f"{file}")


def create_excel_results_file(excel_file: str, result: EvaluationResult):
    summary_df = pd.DataFrame()
    scores = [score["domain_specific_rubrics"] for score in result.scores]
    summary_df["scores"] = scores

    question_indices = [f"Q{i + 1}" for i in range(len(summary_df))]
    question_indices.append("Average")
    question_indices.append("Total Score")
    question_indices.append("Median")

    average = summary_df.mean(axis=0, numeric_only=True)
    total_score = summary_df.sum(axis=0, numeric_only=True)
    median = summary_df.median(axis=0, numeric_only=True)

    summary_df.loc[len(summary_df)] = average
    summary_df.loc[len(summary_df)] = total_score
    summary_df.loc[len(summary_df)] = median

    summary_df.insert(0, "question index", question_indices)

    response_df = result.dataset.to_pandas()
    response_df["scores"] = scores

    with pd.ExcelWriter(excel_file) as writer:
        response_df.to_excel(writer, sheet_name="dataset", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
    return summary_df


def write_results(
    result: EvaluationResult, file_formats: List[str], output_dir: str, model_name: str
) -> List[str]:
    validate_output_file_formats(file_formats)
    response_df = result.dataset.to_pandas()
    response_df["model_name"] = model_name

    scores = [score["domain_specific_rubrics"] for score in result.scores]
    response_df["scores"] = scores

    now = datetime.now()
    timestamp = now.isoformat()
    response_df["evaluation_run"] = f"run-{timestamp}"

    results_files = []
    for fmt in file_formats:
        if IOFileType.JSONL.value == fmt:
            results_file = create_results_file_name(
                IOFileType.JSONL.value, output_dir, timestamp, model_name
            )
            response_df.to_json(f"{results_file}", orient="records", lines=True)
            results_files.append(results_file)
        elif IOFileType.CSV.value == fmt:
            results_file = create_results_file_name(
                IOFileType.CSV.value, output_dir, timestamp, model_name
            )
            response_df.to_csv(f"{results_file}", index=False)
            results_files.append(results_file)
        elif IOFileType.XLSX.value == fmt:
            results_file = create_results_file_name(
                IOFileType.XLSX.value, output_dir, timestamp, model_name
            )
            create_excel_results_file(results_file, result)
            results_files.append(results_file)

    return results_files


def run_local_model_evaluation(
    ctx,
    evaluator,
    model,
    model_name: str | None,
    temperature,
    system_prompt,
    max_workers,
    gpus,
    backend,
    enable_serving_output,
    input_questions,
    judge_model_name,
    judge_openai_api_key,
):
    server = None
    try:
        logger.debug("Model being evaluated in DK-Bench is local")
        if model_name is None:
            model_name = get_local_model_name(model)
        server, api_base, _ = launch_server(
            eval_serve=ctx.obj.config.serve,
            tls_client_cert=ctx.params["tls_client_cert"],
            tls_client_key=ctx.params["tls_client_key"],
            tls_client_passwd=ctx.params["tls_client_passwd"],
            tls_insecure=ctx.params["tls_insecure"],
            model=model,
            model_name=model_name,
            max_workers=max_workers,
            gpus=gpus,
            backend=backend,
            enable_serving_output=enable_serving_output,
        )
        openai_client = get_local_openai_client(model_api_base=api_base, api_key=None)
        model_config = ModelConfig(
            model_name=model_name, temperature=temperature, system_prompt=system_prompt
        )
        result = evaluator.run(
            dataset=input_questions,
            student_model=model_config,
            student_openai_client=openai_client,
            judge_model_name=judge_model_name,
            judge_openai_api_key=judge_openai_api_key,
        )

    finally:
        if server is not None:
            server.shutdown()

    return result, model_config.model_name


def run_endpoint_model_evaluation(
    ctx: click.Context,
    evaluator: RagasEvaluator,
    model: str,
    model_name: str | None,
    temperature: float,
    system_prompt: str,
    input_questions: pathlib.Path,
    judge_model_name: str,
    judge_openai_api_key: str,
) -> Tuple[EvaluationResult, str]:
    logger.debug("Model being evaluated in DK-Bench is an endpoint")
    http_params = http_client(
        {
            "tls_client_cert": ctx.params["tls_client_cert"],
            "tls_client_key": ctx.params["tls_client_key"],
            "tls_client_passwd": ctx.params["tls_client_passwd"],
            "tls_insecure": ctx.params["tls_insecure"],
        }
    )

    api_key = os.environ.get("STUDENT_ENDPOINT_API_KEY", None)
    if api_key is None:
        logger.debug(
            "API_KEY for model at endpoint %s is not set. To set it set the environment variable $STUDENT_ENDPOINT_API_KEY",
            model,
        )
        api_key = "NO_API_KEY"

    openai_client = OpenAI(
        base_url=model,
        api_key=api_key,
        timeout=DEFAULTS.CONNECTION_TIMEOUT,
        http_client=http_params,
    )

    model_name = get_endpoint_model_name(openai_client, model_name)
    model_config = ModelConfig(
        model_name=model_name, temperature=temperature, system_prompt=system_prompt
    )

    result = evaluator.run(
        dataset=input_questions,
        student_model=model_config,
        student_openai_client=openai_client,
        judge_model_name=judge_model_name,
        judge_openai_api_key=judge_openai_api_key,
    )
    model_name = model_config.model_name

    return result, model_config.model_name


def run_dk_bench(
    ctx: click.Context,
    model: str,
    model_name: str | None,
    max_workers: str | int | None,
    gpus: int | None,
    backend: str | None,
    enable_serving_output: bool,
    input_questions: pathlib.Path,
    system_prompt: str,
    temperature: float,
    judge_model_name: str,
) -> tuple[EvaluationResult, str | None]:
    logging.info("Running DK-Bench with settings: %s", locals())

    if "OPENAI_API_KEY" not in os.environ:
        raise EnvironmentError(
            "Environment variable 'OPENAI_API_KEY' must be set to run the Judge model in DK-Bench."
        )
    judge_openai_api_key = os.environ.get("OPENAI_API_KEY", "NO_API_KEY")

    # need to make sure input questions is a jsonl file
    if input_questions.suffix.lstrip(".") != IOFileType.JSONL.value:
        raise ValueError(
            f"Invalid file type: {input_questions}. Expected a '.jsonl' file."
        )

    get_responses_from_model = True
    try:
        test_df = pd.read_json(input_questions, orient="records", lines=True)
        if "response" in test_df.columns:
            logger.info(
                "Input file %s already contains responses for evaluation. Responses from %s will not be collected for this file.",
                input_questions,
                model,
            )
            get_responses_from_model = False

    except BaseException as exc:
        raise ValueError(
            f"Contents of {input_questions} cannot be loaded as JSON. Please ensure it is a valid '.jsonl' file."
        ) from exc

    evaluator = RagasEvaluator()
    if get_responses_from_model:
        # local model evaluation
        if model_is_local_and_valid(model):
            result, model_name = run_local_model_evaluation(
                ctx,
                evaluator,
                model,
                model_name,
                temperature,
                system_prompt,
                max_workers,
                gpus,
                backend,
                enable_serving_output,
                input_questions,
                judge_model_name,
                judge_openai_api_key,
            )
        # endpoint model evaluation
        else:
            result, model_name = run_endpoint_model_evaluation(
                ctx,
                evaluator,
                model,
                model_name,
                temperature,
                system_prompt,
                input_questions,
                judge_model_name,
                judge_openai_api_key,
            )
    # evaluation on just a dataset with responses already provided
    else:
        result = evaluator.run(
            dataset=input_questions,
            judge_model_name=judge_model_name,
            judge_openai_api_key=judge_openai_api_key,
        )
        model_name = "no-model-provided"

    return result, model_name
