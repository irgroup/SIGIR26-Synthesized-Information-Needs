from langchain.chat_models import init_chat_model


def get_llm_qwen():
    return init_chat_model(
        model="qwen/qwen3-14b",
        temperature=0.0,
        base_url="http://127.0.0.1:1234/v1",
        api_key="not-needed",
        model_provider="openai",
    )


def get_llm_qwen_8b():
    return init_chat_model(
        model="qwen3:8b",
        temperature=0.0,
        base_url="http://139.6.160.39:6543/v1",
        api_key="not-needed",
        model_provider="openai",
    )


def get_llm_gemeni_flash():
    return init_chat_model(
        model="gemini-2.5-flash",
        model_provider="google_genai",
        temperature=0,
    )


def get_llm(llm_name):
    llm_connections = {
        "qwen": get_llm_qwen,
        "qwen-8b": get_llm_qwen_8b,
        "gemini-flash": get_llm_gemeni_flash,
    }
    llm = llm_connections.get(llm_name)
    if llm is None:
        raise ValueError(
            f"LLM {llm_name} is not supported. Available options: {list(llm_connections.keys())}"
        )
    return llm()
