# Phoenix Project Guidelines

## Setup & Commands
- Install dependencies: `pip install -r requirements.txt`
- Run indexer: `python phoenix_indexer.py --input path/to/conversations.json --db phoenix_memory.db`
- Run chat interface: `python phoenix_chat.py --db phoenix_memory.db`
- Install spaCy model: `python -m spacy download en_core_web_sm`

## Code Style Guidelines
- Use Python type hints consistently (see example: `def generate_embedding(text: str) -> list[float] | None:`)
- Organize imports: standard library first, then third-party packages, then local modules
- Follow PEP 8 naming conventions: snake_case for functions/variables, PascalCase for classes
- Comprehensive error handling with specific exception types and descriptive messages
- Include docstrings for all functions and classes
- Use consistent indentation (4 spaces)
- Prefer explicit error handling over silent failure
- Rate-limit API calls properly (sleep between calls)
- Use retry mechanisms for external API calls

## Project Structure
- Separate concerns between indexing (phoenix_indexer.py) and chat interface (phoenix_chat.py)
- Keep constants and configurations at the top of files
- Use classes for stateful components (e.g., GeminiClient)