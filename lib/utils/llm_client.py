from __future__ import annotations

from typing import Any
import base64
from pathlib import Path

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
            "Tu es un extracteur de données pour un système RAG financier. "
            "Extrais uniquement les faits chiffrés clés de ce tableau HTML : valeurs importantes, totaux, comparaisons entre colonnes. "
            "Format attendu : une liste de faits courts et précis. "
            "Interdit : descriptions du tableau, recommandations, conclusions, phrases d'introduction. "
            "Langue : français. Longueur maximale : 10 lignes. /no_think"
            f"\n\nTableau:\n{html_table}"
        )
        return self.generate_text(prompt)

    def describe_image(
        self,
        image_input: str,
        additional_headers: dict[str, str] | None = None,
        max_words: int = 512,
        image_media_type: str = "image/png",
    ) -> str:

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if additional_headers:
            headers.update(additional_headers)

        system_prompt = (
            "Tu es un extracteur de données pour un système RAG financier. "
            "Extrais uniquement les données factuelles visibles : titres, valeurs numériques, labels et catégories. "
            "Format attendu : une liste de faits courts et précis. "
            "Interdit : descriptions visuelles (couleurs, styles, mise en page), recommandations, conclusions, phrases d'introduction. "
            f"Langue : français. Longueur max {max_words} tokens. /no_think"
        )

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{image_media_type};base64,{image_input}",
                            },
                        },
                    ],
                },
            ],
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
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

        raise ValueError("Unsupported OpenAI/Qwen vision response payload format")
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
            "chat_template_kwargs": {"enable_thinking": False}
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

    def generate_embeddings(self, inputs: str | list[str]) -> list[list[float]]:
        """Generate embeddings from an OpenAI-compatible embeddings endpoint."""
        normalized_inputs = [inputs] if isinstance(inputs, str) else list(inputs)
        if not normalized_inputs:
            return []

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "input": normalized_inputs,
        }

        response = requests.post(
            self.api_url,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()

        rows = data.get("data")
        if not isinstance(rows, list):
            raise ValueError("Invalid embeddings payload: missing data list")

        sorted_rows = sorted(
            rows,
            key=lambda row: row.get("index", 0) if isinstance(row, dict) else 0,
        )

        embeddings: list[list[float]] = []
        for row in sorted_rows:
            if not isinstance(row, dict) or not isinstance(row.get("embedding"), list):
                raise ValueError("Invalid embeddings payload: each data item must contain embedding list")

            embeddings.append([float(value) for value in row["embedding"]])

        if len(embeddings) != len(normalized_inputs):
            raise ValueError(
                "Invalid embeddings payload: vector count does not match input count"
            )

        return embeddings

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
