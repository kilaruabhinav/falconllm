"""Check setup with the executing interpreter, without revealing credentials."""
import importlib
from pathlib import Path
import subprocess
import sys


def main() -> int:
    # Running a file under scripts/ puts scripts/ (not the repo) on sys.path.
    # A root-cwd `python -c` child checks real imports without mutating sys.path.
    root = Path(__file__).resolve().parents[1]
    if Path.cwd() != root or sys.path[0] != "":
        result = subprocess.run(
            [sys.executable, "-c", "import runpy; runpy.run_path('scripts/check_setup.py', run_name='__main__')"],
            cwd=root,
        )
        return result.returncode

    failed = sys.version_info < (3, 11)
    print(f"{'FAIL' if failed else 'PASS'} Python {sys.version.split()[0]} (requires 3.11+)")
    for label, module in [
        ("python-dotenv", "dotenv"), ("pydantic", "pydantic"),
        ("google-genai", "google.genai"), ("backend config", "backend.agent.config"),
        ("backend engine", "backend.agent.engine"), ("backend schemas", "backend.agent.schemas"),
        ("LLM factory", "backend.agent.llm_factory"), ("tool registry", "backend.tools.registry"),
        ("state layer", "backend.core.state"), ("trace layer", "backend.core.trace"),
        ("persistence", "backend.core.persistence"), ("FastAPI app", "backend.api.app"),
    ]:
        try:
            importlib.import_module(module)
            print(f"PASS {label} import")
        except Exception as exc:
            failed = True
            print(f"FAIL {label} import ({type(exc).__name__}); run {sys.executable} -m pip install -r requirements.txt")
    try:
        from backend.agent.config import AgentConfig
        config = AgentConfig()
        print("PASS configuration values")
        print(f"PASS Gemini key configured: {'YES' if config.GEMINI_API_KEY else 'NO'}")
        print(f"PASS OpenAI key configured: {'YES' if config.OPENAI_API_KEY else 'NO'}")
        print(f"PASS Search key configured: {'YES' if config.TAVILY_API_KEY else 'NO'}")
        print(f"PASS Configured LLM mode: {config.LLM_PROVIDER_MODE}")
        chain = []
        if config.LLM_PROVIDER_MODE in {"gemini", "fallback"} and config.GEMINI_API_KEY:
            chain.extend([
                f"gemini/{config.GEMINI_PRIMARY_MODEL}",
            ])
            if config.LLM_PROVIDER_MODE == "fallback":
                chain.append(f"gemini/{config.GEMINI_FALLBACK_MODEL}")
        if config.LLM_PROVIDER_MODE in {"openai", "fallback"} and config.OPENAI_API_KEY:
            chain.append(f"openai/{config.OPENAI_MODEL}")
        print("PASS Configured chain: " + (" -> ".join(chain) if chain else "none"))
        from backend.tools.factory import create_tool_registry
        registry = create_tool_registry(config)
        names = registry.list_tools()
        expected = {"calculator", "search", "file_reader"}
        if set(names) != expected:
            raise RuntimeError(f"unexpected tools: {names}")
        print(f"PASS tool registry ({', '.join(names)})")
        from backend.core.persistence import SQLiteTraceStore
        store = SQLiteTraceStore(":memory:")
        store.close()
        print("PASS SQLite schema initialization")
    except Exception:
        failed = True
        print("FAIL configuration; check .env against .env.example")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
