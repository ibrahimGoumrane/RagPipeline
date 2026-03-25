from __future__ import annotations

from typing import Any

import requests


class LLMClient:
    """Qwen/OpenAI-compatible HTTP client for text generation APIs."""

    def __init__(self, api_url: str, api_key: str, model: str, timeout: int = 60):
        if not api_url:
            raise ValueError("Qwen API URL is required")
        if not model:
            raise ValueError("Qwen model name is required")
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def summarize_table(self, html_table: str) -> str:
        prompt = (
            "Tu es un assistant d'analyse financiere. "
            "Reponds toujours en francais, avec un ton professionnel et concis. "
            "A partir du tableau HTML ci-dessous, redige un resume court (2-3 phrases) "
            "des tendances principales, des valeurs notables et des comparaisons importantes. "
            "Retourne uniquement le texte final du resume, sans preambule ni commentaire hors sujet.\\n\\n"
            f"Table:\\n{html_table}"
        )
        return self.generate_text(prompt)

    def generate_text(self, prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Reponds toujours en francais, de maniere concise et professionnelle. "
                        "Donne uniquement le contenu final utile. "
                        "N'ajoute pas d'analyse interne, de justification, de meta-commentaire, "
                        "ni d'information non liee au contenu."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
        }

        response = requests.post(
            self.api_url,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()

        text = self._parse_openai_style(data)
        if text:
            return text

        raise ValueError("Unsupported Qwen/OpenAI response payload format")

    @staticmethod
    def _parse_openai_style(data: dict[str, Any]) -> str:
        choices = data.get("choices") or []
        if not choices:
            return ""

        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(parts).strip()

        return ""
