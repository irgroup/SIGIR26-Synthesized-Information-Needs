from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from topic_gen import logger

from src.data import MODELS_DIR


def get_llm_qwen3_14B_MT100_no_think(connection: str):
    return ChatOpenAI(
        openai_api_base=connection,
        openai_api_key="not-needed",
        model_name="Qwen/Qwen3-14B",
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
            "max_tokens": 100,
        },
    )


def get_llm_qwen3_14B_no_think(connection: str):
    return ChatOpenAI(
        openai_api_base=connection,
        openai_api_key="not-needed",
        model_name="Qwen/Qwen3-14B",
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
            "max_tokens": 10000,
        },
    )


def get_llm_qwen3_30B_MT100_no_think(connection: str):
    return ChatOpenAI(
        openai_api_base=connection,
        openai_api_key="not-needed",
        model_name="Qwen/Qwen3-30B-A3B-Instruct-2507",
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
            "max_tokens": 100,
        },
    )


def get_llm_qwen3_30B_no_think(connection: str):
    return ChatOpenAI(
        openai_api_base=connection,
        openai_api_key="not-needed",
        model_name="Qwen/Qwen3-30B-A3B-Instruct-2507",
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
            "max_tokens": 10000,
        },
    )


def get_llm_qwen3_next_80B_fp8_no_think(connection: str):
    return ChatOpenAI(
        openai_api_base=connection,
        openai_api_key="not-needed",
        model_name="Qwen/Qwen3-Next-80B-A3B-Instruct-FP8",
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
            "max_tokens": 10000,
        },
    )


def get_llm_gpt_oss_20B(connection: str):
    return ChatOpenAI(
        openai_api_base=connection,
        openai_api_key="not-needed",
        model_name="openai/gpt-oss-20b",
    )


def get_llm_gpt_oss_120B(connection: str):
    return ChatOpenAI(
        openai_api_base=connection,
        openai_api_key="not-needed",
        model_name="openai/gpt-oss-120b",
        # extra_body={"reasoning_effort": "low"},
    )


def get_llm_gpt_oss_120B_MT1000(connection: str):
    return ChatOpenAI(
        openai_api_base=connection,
        openai_api_key="not-needed",
        model_name="openai/gpt-oss-120b",
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
            "max_tokens": 1000,
        },
    )

def get_llm_nemotron3_30B(connection: str):
    # from langchain_mistralai import ChatMistralAI

    return ChatOpenAI(
        openai_api_base=connection,
        openai_api_key="not-needed",
        model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    )


def get_llm_mistral3_14B_MT100(connection: str):
    # from langchain_mistralai import ChatMistralAI

    return ChatOpenAI(
        openai_api_base=connection,
        # base_url=connection,
        openai_api_key="not-needed",
        model="mistralai/Ministral-3-14B-Instruct-2512",
        max_tokens=100,
        # model_kwargs={"chat_template_kwargs": {"enable_thinking": False}},
    )


def get_llm_gpt_oss_120B_MT1000_ollama(connection: str):
    from langchain_ollama import ChatOllama

    return ChatOllama(base_url=connection, model="gpt-oss:120b", num_predict=1000)


def get_llm_gpt_oss_120B_ollama(connection: str):
    from langchain_ollama import ChatOllama

    return ChatOllama(base_url=connection, model="gpt-oss:120b")


def get_llm_llama3_1_70b_instruct_q8_0_MT1000_ollama(connection: str):
    from langchain_ollama import ChatOllama

    return ChatOllama(
        base_url=connection, model="llama3.1:70b-instruct-q8_0", num_predict=1000
    )


def get_llm_llama3_1_70b_instruct_q8_0_ollama(connection: str):
    from langchain_ollama import ChatOllama

    return ChatOllama(base_url=connection, model="llama3.1:70b-instruct-q8_0")


def get_llm_llama3_3_70b_instruct_q8(connection: str):
    return ChatOpenAI(
        openai_api_base=connection,
        openai_api_key="not-needed",
        model_name="RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic",
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
            "max_tokens": 10000,
        },
    )


def get_llm_llama3_1_8B_instruct(connection: str):
    return ChatOpenAI(
        openai_api_base=connection,
        openai_api_key="not-needed",
        model_name="meta-llama/Llama-3.1-8B-Instruct",
    )


def get_llm_gemeni_flash(connection=None):
    return init_chat_model(
        model="gemini-2.5-flash",
        model_provider="google_genai",
        temperature=0,
    )


def get_llm_deepseek(connection=None):
    return init_chat_model(
        model="deepseek-chat",
        model_provider="deepseek",
        temperature=0,
    )


def get_llm_qwen3_30B_A3B_Instruct_2507_FP8(connection=None):
    import torch
    from langchain_community.llms import VLLM

    # setup gpus
    logger.info(f"Cuda: {torch.cuda.is_available()}")
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    # os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    logger.info(f"Number of GPUs: {torch.cuda.device_count()}")

    # load model
    llm = VLLM(
        model="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
        download_dir=MODELS_DIR,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0,
        vllm_kwargs={
            "max_model_len": 32768,
            "max_num_batched_tokens": 32768,
            "gpu_memory_utilization": 0.7,
        },
    )
    return llm


# register LLM connections
def get_llm(llm_name: str, connection: str):
    llm_connections = {
        "qwen3-14B-MT100-no-think": get_llm_qwen3_14B_MT100_no_think,
        "qwen3-14B-no-think": get_llm_qwen3_14B_no_think,
        "qwen3-30B-MT100-no-think": get_llm_qwen3_30B_MT100_no_think,
        "qwen3-30B-no-think": get_llm_qwen3_30B_no_think,
        "gemini-2.5-flash": get_llm_gemeni_flash,
        "Qwen3-30B-A3B-Instruct-2507-FP8": get_llm_qwen3_30B_A3B_Instruct_2507_FP8,
        "gpt-oss-20B": get_llm_gpt_oss_20B,
        "gpt-oss-120B": get_llm_gpt_oss_120B,
        "gpt-oss-120B-MT1000": get_llm_gpt_oss_120B_MT1000,
        "gpt-oss-120B-ollama": get_llm_gpt_oss_120B_ollama,
        "gpt-oss-120B-MT1000-ollama": get_llm_gpt_oss_120B_MT1000_ollama,
        "deepseek-V3.2": get_llm_deepseek,
        "llama3-1-8B-instruct": get_llm_llama3_1_8B_instruct,
        "llama3-1-70B_instruct_q8_0_MT1000_ollama": get_llm_llama3_1_70b_instruct_q8_0_MT1000_ollama,
        "llama3-1-70B_instruct_q8_0_ollama": get_llm_llama3_1_70b_instruct_q8_0_ollama,
        "llama3-3-70b_instruct_q8": get_llm_llama3_3_70b_instruct_q8,
        "mistral3-14B-MT100": get_llm_mistral3_14B_MT100,
        "qwen3-80B-next-no-think": get_llm_qwen3_next_80B_fp8_no_think,
        "get_llm_nemotron3_30B": get_llm_nemotron3_30B,
    }
    llm = llm_connections.get(llm_name)
    if llm is None:
        raise ValueError(
            f"LLM {llm_name} is not supported. Available options: {list(llm_connections.keys())}"
        )
    return llm(connection=connection)


LLM_NAMES = {
    "llama3-1-8B-instruct": "Llama3.1-8B",
    "qwen3-14B-MT100-no-think": "Qwen3-14B",
    "qwen3-14B-no-think": "Qwen3-14B",
    "mistral3-14B-MT100": "Mistral3-14B",
    "gpt-oss-20B": "GPT-OSS-20B",
    "qwen3-30B-MT100-no-think": "Qwen3-30B",
    "qwen3-30B-no-think": "Qwen3-30B",
    "Qwen3-30B-A3B-Instruct-2507-FP8": "Qwen3-30B-A3B-Instruct-2507-FP8",
    "gpt-oss-120B": "GPT-OSS-120B",
    "gpt-oss-120B-MT1000": "GPT-OSS-120B",
    "gpt-oss-120B-ollama": "GPT-OSS-120B-O",
    "gpt-oss-120B-MT1000-ollama": "GPT-OSS-120B-O",
    "llama3-1-70B_instruct_q8_0_MT1000_ollama": "Llama3.1-70B",
    "llama3-1-70B_instruct_q8_0_ollama": "Llama3.1-70B",
    "llama3-3-70b_instruct_q8": "Llama3.3-70B",
    "qwen3-80B-next-no-think": "Qwen3-Next-80B",
    "deepseek-V3.2": "Deepseek-V3.2",
    "gemini-2.5-flash": "Gemini-2.5-Flash",
    "get_llm_nemotron3_30B": "NemoTron3-30B",
}


MODEL_SORTER = [
    "gpt-4.1",
    "Llama3.1-8B",
    "Qwen3-14B",
    "Mistral3-14B",
    "GPT-OSS-20B",
    "Qwen3-30B-A3B-Instruct-2507-FP8",
    "Qwen3-30B",
    "Llama3.1-70B",
    "Qwen3-Next-80B",
    "GPT-OSS-120B",
    "GPT-OSS-120B-O",
    "Deepseek-V3.2",
    "Gemini-2.5-Flash",
    "NemoTron3-30B",
]
