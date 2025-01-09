# SPDX-License-Identifier: Apache-2.0

# pylint: disable=ungrouped-imports
# Standard
from datetime import datetime
import enum
import logging
import pathlib
import os

# Third Party
import httpx
import click
from openai import OpenAI, OpenAIError
import pandas as pd
from ragas.dataset_schema import EvaluationDataset, EvaluationResult

# First Party
from instructlab.eval.ragas import ModelConfig, RagasEvaluator
from instructlab.eval.mt_bench_common import get_openai_client

# Local
from ..utils import is_model_gguf, is_model_safetensors, validate_taxonomy_file
from ..client_utils import http_client, ClientException
from .evaluate import launch_server, get_model_name
from ..configuration import DEFAULTS

logger = logging.getLogger(__name__)

class IOFileType(enum.Enum):
    CSV: str = "csv"
    JSONL: str = "jsonl"
    XLSX: str = "xlsx"
    QNA_YAML: str = "qna.yaml"

def model_is_local(model: str) -> bool:
    if model is not None and os.path.exists(model):
        model_path = pathlib.Path(model)
        valid_model = False
        if model_path.is_dir():
            valid_model = is_model_safetensors(model_path)
        elif model_path.is_file():
            valid_model = is_model_gguf(model_path)
        return valid_model
    else:
        return False

def get_endpoint_model_name(endpoint: str, api_key: str, http_client: httpx.Client) -> str:
    try:
        client = OpenAI(
            base_url=endpoint,
            api_key=api_key,
            timeout=DEFAULTS.CONNECTION_TIMEOUT,
            http_client=http_client,
        )
        models = client.models.list()
    except OpenAIError as exc:
        raise ClientException(f"Connection Error {exc}") from exc

    if len(models.data) != 1:
        raise ClientException(f"More than one model at endpoint, pass in model name manually")

    return models.data[0].id

def create_results_file_name(file_format: IOFileType, output_dir: str, timestamp: str, model_name: str) -> str:
    valid_result_formats = [IOFileType.CSV.value, IOFileType.XLSX.value, IOFileType.JSONL.value]
    if file_format not in valid_result_formats:
        raise ValueError("File format is not one of: csv, xlsx, jsonl")
    
    if IOFileType.CSV.value == file_format:
        file_type = "csv"
    if IOFileType.XLSX.value == file_format:
        file_type = "xlsx"
    if IOFileType.JSONL.value == file_format:
        file_type = "jsonl"

    output_path = pathlib.Path(output_dir)
    if not output_path.exists() or not output_path.is_dir():
        output_dir = DEFAULTS.EVAL_DATA_DIR
        logger.debug("Output dir provided is not a directory or does not exist, writing results to default evaluation output directory %s", output_dir)

    return f"{output_dir}/responses-{model_name}-{timestamp}.{file_type}"

def read_qna_yaml(yaml_file: pathlib.Path) -> pd.DataFrame:
    qna_list = []

    _, errors = validate_taxonomy_file(yaml_file)
    if errors:
        logger.debug("Skipping %s due to errors. Run `ilab taxonomy diff` on a taxonomy with this qna.yaml file to get errors", yaml_file)
        return pd.DataFrame()

    with yaml_file.open("r") as file:
        qna = yaml.safe_load(file)
        for seed_example in qna["seed_examples"]:
            for questions_and_answers in seed_example["questions_and_answers"]:
                qna_list.append({
                    "user_input": questions_and_answers["question"].strip(),
                    "reference": questions_and_answers["answer"].strip()
                })
    return pd.DataFrame(qna_list)

def get_input_df_from_file(file: pathlib.Path, evaluator: RagasEvaluator) -> pd.DataFrame:
    input_df = pd.DataFrame()

    # skips file's that are just the directory when traversing subdirectories
    if file.is_dir():
        return input_df

    try:
        if IOFileType.CSV.value == file.suffix.strip('.'):
            input_df = pd.read_csv(file)
        elif IOFileType.JSONL.value == file.suffix.strip('.'):
            input_df = pd.read_json(file, orient="records", lines=True)
        elif IOFileType.XLSX.value == file.suffix.strip('.'):
            input_df = pd.read_excel(file)
        elif IOFileType.QNA_YAML.value == file.name:
            input_df = read_qna_yaml(file)
        else:
            logger.debug("Ignoring reading %s. Invalid file format. File extension must be .csv, .jsonl, or .xlsx", file)
            return input_df

        evaluator.validate_dataset(input_df)
    except ValueError as exc:
        print(f"Error in {file}. {exc}")
        return pd.DataFrame()

    logger.debug("Added %s to the evaluation dataset", file)
    return input_df

def get_input_df(input_questions: str, evaluator: RagasEvaluator) -> pd.DataFrame:
    input_questions_path = pathlib.Path(input_questions)
    input_df = pd.DataFrame()

    if input_questions_path.exists():
        if input_questions_path.is_dir():
            for file in input_questions_path.rglob("*"):
                file_df = get_input_df_from_file(file,evaluator)
                input_df = pd.concat([input_df, file_df], ignore_index=True)
            # final validation of entire file after all concats
            evaluator.validate_dataset(input_df)
        if input_questions_path.is_file():
            input_df = get_input_df_from_file(input_questions_path,evaluator)

    input_df = input_df.drop_duplicates(subset=["user_input"])

    return input_df

def get_responses_from_model(
    ctx: click.Context,
    evaluator: RagasEvaluator,
    input_df: pd.DataFrame,
    model: str,
    model_prompt: str,
    temperature: float,
    max_workers: str | int | None,
    gpus: int | None,
    backend: str | None, 
    enable_serving_output: bool,
) -> pd.DataFrame:
    server = None
    if model_is_local(model):
        logger.debug("DK-Bench model is local")
        model_name = get_model_name(model)
        server, api_base, effective_gpus = launch_server(
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
    else:
        logger.debug("DK-Bench model is an endpoint")
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
            logger.debug("API_KEY for model at endpoint %s is not set. To set it set the environment variable $STUDENT_ENDPOINT_API_KEY", model)

        model_name = get_endpoint_model_name(model, api_key, http_params)
        api_base = model

    openai_client = get_openai_client(model_api_base=api_base, api_key=api_key)
    model_config = ModelConfig(model_name=model_name, temperature=temperature, system_prompt=model_prompt)
    input_df = evaluator.generate_answers_from_model(input_df, model_config, openai_client)

    return input_df, server, model_name

def make_run_dir(output_dir: str) -> str:
    now = datetime.now()
    timestamp = now.strftime("%m-%d-%Y_%H-%M-%S-%f")
    if output_dir.endswith("/"):
        output_dir = output_dir[:-1]
    run_dir = f"{output_dir}/job-{timestamp}"

    path_does_not_exist = not os.path.exists(output_dir) 
    path_is_not_dir =  not os.path.isdir(output_dir)
    if path_does_not_exist or path_is_not_dir:
        run_dir = f"{DEFAULTS.EVAL_DATA_DIR}/job-{timestamp}",

    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def print_header():
    print("\n")
    print("# DK-BENCH REPORT")

def print_results(result: EvaluationResult, model_name: str):
    print(f"\n## MODEL: {model_name}\n")
    total_score = 0
    question_num = 0
    for score in result.scores:
        question_num += 1
        print(f"Question #{question_num}:     {score['domain_specific_rubrics']}/5")
        total_score += score['domain_specific_rubrics']

    average = total_score/len(result.scores)
    average = round(average, 2)
    print(f"----------------------------")
    print(f"Average Score:   {average}/5")
    print(f"Total Score:     {total_score}/{question_num*5}\n")

def create_excel_results_file(excel_file: str, result: EvaluationResult, model_name: str):
    summary_df = pd.DataFrame()
    scores = [score["domain_specific_rubrics"] for score in result.scores]
    summary_df['scores'] = scores

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

    summary_df.insert(0, 'question index', question_indices)

    response_df = result.dataset.to_pandas()
    response_df['scores'] = scores

    # Append the new sheet
    with pd.ExcelWriter(excel_file) as writer:
        response_df.to_excel(writer, sheet_name="dataset",index=False) 
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
    return summary_df

def write_results(result: EvaluationResult, file_formats: str, output_dir: str, model_name: str):
        response_df = result.dataset.to_pandas()
        response_df["model_name"] = model_name

        scores = [score["domain_specific_rubrics"] for score in result.scores]
        response_df['scores'] = scores

        now = datetime.now() 
        timestamp = now.strftime("%m-%d-%Y_%H-%M-%S-%f")
        response_df["evaluation_run"] = f"run-{timestamp}"

        print(f"Responses and scores written to:")
        for fmt in file_formats:
            if IOFileType.JSONL.value == fmt:
                results_file = create_results_file_name(IOFileType.JSONL.value, output_dir, timestamp, model_name)
                response_df.to_json(f"{results_file}", orient="records", lines=True)
            elif IOFileType.CSV.value == fmt:
                results_file = create_results_file_name(IOFileType.CSV.value, output_dir, timestamp, model_name)
                response_df.to_csv(f"{results_file}", index=False)
            elif IOFileType.XLSX.value == fmt:
                results_file = create_results_file_name(IOFileType.XLSX.value, output_dir, timestamp, model_name)
                create_excel_results_file(results_file, result, model_name)
            else:
                logger.debug("Output format %s is not valid", fmt)
                continue

            print(f"{results_file}")
        
def run_dk_bench(ctx: click.Context,
                 model: str,
                 max_workers: str | int | None,
                 gpus: int | None,
                 backend: str | None,
                 enable_serving_output: bool,
                 input_questions: pathlib.Path,
                 output_dir: str,
                 model_prompt: str,
                 temperature: float,
                 judge_model_name: str,
) -> tuple[EvaluationResult, str | None]:

    logging.info(f"Running DK-Bench with settings: %s", locals())

    if "OPENAI_API_KEY" not in os.environ:
        raise EnvironmentError("Environment variable 'OPENAI_API_KEY' must be set to run the Judge model in DK-Bench.")
    judge_openai_api_key = os.environ.get("OPENAI_API_KEY", None)

    evaluator = RagasEvaluator(judge_model_name=judge_model_name, judge_openai_api_key=judge_openai_api_key)

    input_df = get_input_df(input_questions, evaluator)

    server = None
    model_name = None
    try:
        if model is not None:
            input_df, server, model_name = get_responses_from_model(ctx, evaluator, input_df, model, model_prompt, temperature, max_workers, gpus, backend, enable_serving_output)

        result = evaluator.run(dataset=input_df)

    finally:
        if model_is_local(model):
            if server is not None:
                server.shutdown()

    if not model_name:
        model_name = "No Model Provided"

    return result, model_name
