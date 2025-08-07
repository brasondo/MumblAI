# mumble_extender.py
import subprocess
import json

class MumbleExtender:
    def __init__(self, model="llama3"):
        self.model = model

    def rewrite_mumbles(self, mumbles: list[str], style: str = "Yeat") -> list[str]:
        prompt = self._build_prompt(mumbles, style)
        response = self._query_ollama(prompt)
        return self._extract_lines(response, len(mumbles))

    def _build_prompt(self, mumbles: list[str], style: str) -> str:
        lines = "\n".join([f"{i+1}. {line}" for i, line in enumerate(mumbles)])
        return f"""
You're an AI ghostwriter for a rapper named {style}. 
Your job is to turn vague, unclear, or weak bars into hard, stylistically on-brand lines. 
Rewrite the lines below in {style}'s voice. Keep the count the same. Stick to one or two bars per line.

Lines to improve:
{lines}

Respond only with the new lines numbered 1 to {len(mumbles)}.
"""

    def _query_ollama(self, prompt: str) -> str:
        result = subprocess.run(
            ["ollama", "run", self.model],
            input=prompt,
            text=True,
            capture_output=True
        )
        return result.stdout

    def _extract_lines(self, raw_output: str, expected: int) -> list[str]:
        lines = []
        for line in raw_output.strip().split("\n"):
            if "." in line:
                try:
                    _, content = line.split(".", 1)
                    lines.append(content.strip())
                except ValueError:
                    continue
        return lines[:expected]
