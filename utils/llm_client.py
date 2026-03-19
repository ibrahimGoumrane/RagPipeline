from __future__ import annotations

from typing import Any

import requests


class LLMClient:
    """Small reusable HTTP client for text generation APIs."""

    def __init__(self, api_url: str, api_key: str, model: str, timeout: int = 60):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def summarize_table(self, html_table: str) -> str:
        prompt = (
            "You are a financial analysis assistant. "
            "Given the HTML table below, write a short 2-3 sentence summary of the main trends, "
            "notable values, and important comparisons. "
            "Return only the summary text.\\n\\n"
            f"Table:\\n{html_table}"
        )
        return self.generate_text(prompt)

    def generate_text(self, prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 300,
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

        text = self._parse_gemini_native(data)
        if text:
            return text

        raise ValueError("Unsupported response payload format from description API")

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
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    parts.append(block["text"])
            return "\n".join(parts).strip()

        return ""

    @staticmethod
    def _parse_gemini_native(data: dict[str, Any]) -> str:
        candidates = data.get("candidates") or []
        if not candidates:
            return ""

        content = candidates[0].get("content") or {}
        parts = content.get("parts") or []

        texts: list[str] = []
        for part in parts:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                texts.append(part["text"])
        return "\n".join(texts).strip()
