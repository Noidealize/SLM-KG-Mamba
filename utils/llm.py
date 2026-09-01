from pathlib import Path


def resolve_llm_ckp_dir(path_or_repo_id):
    text = str(path_or_repo_id)
    looks_like_path = (
        "\\" in text
        or "/" in text
        or text.startswith(".")
        or ":" in text
    )

    if looks_like_path:
        path = Path(text).expanduser()
        if path.exists():
            return str(path.resolve())
        raise FileNotFoundError(
            f"LLM path does not exist: {path}. "
            "Check --llm_ckp_dir, or download the model to that directory first."
        )

    return text
