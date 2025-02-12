from dotenv import load_dotenv
import os

def get_model(model_name: str, temperature: float = 0, **kwargs):
    """
    Get a model from the environment variables.
    """
    # Load the environment variables
    load_dotenv()

    # Get the model from the environment variables
    if "claude" in model_name:
        from langchain_anthropic import ChatAnthropic
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables.")
        os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")
        return ChatAnthropic(model=model_name, temperature=temperature, **kwargs)
    elif "gpt" in model_name:
        from langchain_openai import ChatOpenAI
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        return ChatOpenAI(model=model_name, temperature=temperature, **kwargs)
    elif 'llama' in model_name:
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model_name, temperature=temperature, **kwargs)
    else:
        raise ValueError(f"Model {model_name} not supported.")
