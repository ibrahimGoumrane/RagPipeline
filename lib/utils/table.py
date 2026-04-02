from __future__ import annotations

import uuid
from typing import Any

from bs4 import BeautifulSoup


def extract_table_row_chunks(
    combined_html: str,
    heading: str,
    page_ref: int | None,
    doc_id: str,
    logger: Any | None = None,
) -> list[dict[str, Any]]:
    """Parse HTML tables and emit one chunk per row-header/value pair set."""
    try:
        soup = BeautifulSoup(combined_html, "html.parser")
        tables = soup.find_all("table")
        if not tables:
            return []

        all_row_chunks: list[dict[str, Any]] = []

        for table in tables:
            column_names = _resolve_column_names(table)
            if not column_names:
                continue

            data_rows = _get_data_rows(table)
            for row in data_rows:
                cells = row.find_all(["td", "th"])
                if not cells:
                    continue

                if cells[0].name == "th":
                    row_label = cells[0].get_text(strip=True)
                    data_cells = cells[1:]
                    col_names_for_row = column_names[1:] if len(column_names) > 1 else column_names
                else:
                    row_label = None
                    data_cells = cells
                    col_names_for_row = column_names

                pairs: list[str] = []
                for col_name, cell in zip(col_names_for_row, data_cells):
                    val = cell.get_text(strip=True)
                    if not val:
                        continue
                    label = f"{row_label} - {col_name}" if row_label else col_name
                    pairs.append(f"{label}: {val}")

                if not pairs:
                    continue

                content = " | ".join(pairs)
                all_row_chunks.append(
                    {
                        "chunk_id": str(uuid.uuid4()),
                        "_heading": heading,
                        "element_type": "table_row",
                        "page_ref": page_ref,
                        "doc_id": doc_id,
                        "content": content,
                        "content_for_embedding": content,
                        "token_estimate": len(content.split()),
                    }
                )

        return all_row_chunks
    except Exception as exc:
        if logger:
            logger.warning("Row extraction failed: %s", exc)
        return []


def _resolve_column_names(table: Any) -> list[str]:
    """Build flat column names from thead (colspan-aware) or fallback header rows."""
    thead = table.find("thead")

    if thead:
        header_rows = thead.find_all("tr")
    else:
        all_rows = table.find_all("tr")
        header_rows = []
        for row in all_rows:
            cells = row.find_all(["th", "td"])
            if any(c.name == "th" for c in cells):
                header_rows.append(row)
            else:
                break

    if not header_rows:
        return []

    layers: list[list[str]] = []
    for row in header_rows:
        layer: list[str] = []
        for cell in row.find_all(["th", "td"]):
            colspan = cell.get("colspan")
            text = cell.get_text(strip=True)
            layer.extend([text] * (int(colspan) if colspan else 1))
        layers.append(layer)

    n_cols = max(len(layer) for layer in layers)
    for layer in layers:
        if len(layer) < n_cols:
            layer.extend([""] * (n_cols - len(layer)))

    column_names: list[str] = []
    for col_i in range(n_cols):
        parts: list[str] = []
        for layer in layers:
            val = layer[col_i]
            if val and (not parts or val != parts[-1]):
                parts.append(val)
        column_names.append(" / ".join(parts) if parts else f"col_{col_i}")

    return column_names


def _get_data_rows(table: Any) -> list[Any]:
    """Return table data rows, skipping header rows when no tbody is present."""
    tbody = table.find("tbody")
    if tbody:
        return tbody.find_all("tr")

    all_rows = table.find_all("tr")
    data_rows: list[Any] = []
    header_done = False

    for row in all_rows:
        cells = row.find_all(["th", "td"])
        if not header_done:
            if any(c.name == "th" for c in cells):
                continue
            header_done = True
        data_rows.append(row)

    return data_rows
