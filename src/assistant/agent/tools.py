import re
from collections.abc import Callable

from assistant.graphrag.search import global_search, local_search
from assistant.infra.erp_client import (
    ErpClient,
    ErpNotFound,
    ErpRejected,
)
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import KnowledgeBase

WAREHOUSES = [
    "Москва",
    "Санкт-Петербург",
    "Казань",
    "Новосибирск",
]

PART_PATTERN = re.compile(r"^P-\d{4}[A-Z]?$")
MODEL_PATTERN = re.compile(r"^(AR|BZ|GS)-\d{3}$")


def _tool(name: str, description: str, properties: dict, required: list[str]):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


TOOL_SCHEMAS = [
    _tool(
        "search_knowledge",
        "Найти конкретные факты в графе знаний.",
        {
            "question": {"type": "string"},
        },
        ["question"],
    ),
    _tool(
        "summarize_topic",
        "Ответить на обзорный вопрос по сводкам графа.",
        {
            "question": {"type": "string"},
        },
        ["question"],
    ),
    _tool(
        "check_stock",
        "Проверить остаток детали на складе.",
        {
            "part_id": {"type": "string"},
            "warehouse": {
                "type": "string",
                "enum": WAREHOUSES,
            },
        },
        ["part_id", "warehouse"],
    ),
    _tool(
        "get_price",
        "Получить цену детали в рублях.",
        {
            "part_id": {"type": "string"},
        },
        ["part_id"],
    ),
    _tool(
        "create_service_order",
        "Создать заказ детали для модели оборудования.",
        {
            "part_id": {"type": "string"},
            "warehouse": {
                "type": "string",
                "enum": WAREHOUSES,
            },
            "qty": {"type": "integer", "minimum": 1},
            "model": {"type": "string"},
        },
        ["part_id", "warehouse", "qty", "model"],
    ),
]


def build_tools(
    kb: KnowledgeBase,
    llm: LLMClient,
    erp: ErpClient,
) -> dict[str, Callable[..., dict]]:
    """Создаёт функции инструментов агента."""

    def search_knowledge(question: str) -> dict:
        result = local_search(question, kb, llm)
        return {
            "answer": result.text,
            "facts": result.facts,
            "sources": result.sources,
            "refused": result.refused,
        }

    def summarize_topic(question: str) -> dict:
        result = global_search(question, kb, llm)
        return {
            "answer": result.text,
            "sources": result.sources,
            "refused": result.refused,
        }

    def check_stock(part_id: str, warehouse: str) -> dict:
        if not isinstance(part_id, str) or not PART_PATTERN.fullmatch(part_id):
            return {"error": "Некорректный номер детали."}

        if warehouse not in WAREHOUSES:
            return {"error": "Неизвестный склад."}

        try:
            return erp.get_stock(part_id, warehouse)
        except ErpNotFound:
            return {"error": "Данные об остатке детали отсутствуют в ERP."}
        except ErpRejected:
            return {"error": "ERP отклонила запрос остатка."}

    def get_price(part_id: str) -> dict:
        if not isinstance(part_id, str) or not PART_PATTERN.fullmatch(part_id):
            return {"error": "Некорректный номер детали."}

        try:
            return erp.get_price(part_id)
        except ErpNotFound:
            return {"error": "Данные о цене детали отсутствуют в ERP."}
        except ErpRejected:
            return {"error": "ERP отклонила запрос цены."}

    def create_service_order(
        part_id: str,
        warehouse: str,
        qty: int,
        model: str,
    ) -> dict:
        if not isinstance(part_id, str) or not PART_PATTERN.fullmatch(part_id):
            return {"error": "Некорректный номер детали."}

        if warehouse not in WAREHOUSES:
            return {"error": "Неизвестный склад."}

        if isinstance(qty, bool) or not isinstance(qty, int) or qty <= 0:
            return {"error": "Количество должно быть положительным целым числом."}

        if not isinstance(model, str) or not MODEL_PATTERN.fullmatch(model):
            return {"error": "Некорректный код модели."}

        try:
            return erp.create_order(part_id, warehouse, qty, model)
        except ErpNotFound:
            return {"error": "Необходимая запись не найдена в ERP."}
        except ErpRejected as exc:
            return {
                "error": str(exc) or "ERP отклонила создание заказа."
            }

    return {
        "search_knowledge": search_knowledge,
        "summarize_topic": summarize_topic,
        "check_stock": check_stock,
        "get_price": get_price,
        "create_service_order": create_service_order,
    }