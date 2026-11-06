"""Цикл агента. Требования FR-7, FR-8, FR-9 и FR-10."""

import json
import re
from collections.abc import Callable
from typing import Any

from assistant.agent.tools import TOOL_SCHEMAS
from assistant.infra.erp_client import ErpUnavailable
from assistant.infra.grounding import code_type, find_codes
from assistant.infra.llm import LLMClient, tool_message
from assistant.infra.schemas import AgentResult, ToolStep

PART_PATTERN = re.compile(r"^P-\d{4}[A-Z]?$")

ERP_TOOLS = {"check_stock", "get_price", "create_service_order"}

QUANTITY_PATTERN = re.compile(
    r"(?<![\w-])(\d+)\s*(?:шт(?:\.|ук(?:а|и)?)?|штук(?:а|и)?|"
    r"единиц(?:а|ы)?|экземпляр(?:а|ов)?)",
    re.IGNORECASE,
)

CURRENCY_PATTERN = re.compile(
    r"(?<![\w-])(\d[\d\s\u00a0.,]*\d|\d)\s*"
    r"(?:₽|руб(?:\.|лей|ля|ль)?)(?!\w)",
    re.IGNORECASE,
)

PRICE_LABEL_PATTERN = re.compile(
    r"\b(?:цена|стоимость|стоит|обойд[её]тся)\b\s*"
    r"(?:составляет|равна|:|—|-)?\s*(\d[\d\s\u00a0]*)",
    re.IGNORECASE,
)

NUMBER_PATTERN = re.compile(
    r"(?<![\w-])(\d[\d\s\u00a0]*\d|\d)(?!\w)"
)

STOCK_CONTEXT_PATTERN = re.compile(
    r"\b(?:остат(?:ок|ка|ки)?|на\s+складе|в\s+наличии|"
    r"доступно|имеется|числится)\b",
    re.IGNORECASE,
)

VALUE_CONTEXT_BARRIER_PATTERN = re.compile(
    r"\b(?:цена|стоимость|стоит|закажи|заказать|заказ|"
    r"количество|артикул|код|модель)\b",
    re.IGNORECASE,
)

SENTENCE_BREAK_PATTERN = re.compile(r"[.!?;\n]")



AGENT_HISTORY_CHAR_LIMIT = 3500

AGENT_SYSTEM_PROMPT = (
    "Ты — ассистент сервисной службы. Используй инструменты для получения фактов. "
    "Не выдумывай коды деталей, цены, остатки, статусы и результаты заказов. "
    "Код детали должен быть указан пользователем или подтверждён результатом инструмента. "
    "Для конкретных фактов о модели, ошибке, регламенте, детали, узле или поставщике "
    "используй search_knowledge ровно один раз. Для обзорных вопросов, статистики "
    "отказов и перечней снятых с производства деталей используй summarize_topic "
    "ровно один раз. Не повторяй поиск с тем же вопросом после получения результата. "
    "Для цены используй get_price, для остатков на конкретном складе — check_stock. "
    "При оформлении заказа сначала найди актуальную деталь и проверь её остаток "
    "и статус производства. Количество заказа должно быть явно указано пользователем. "
    "Не создавай заказ без явного подтверждения пользователя. "
    "Поля qty и discontinued независимы: положительный остаток не отменяет "
    "снятие детали с производства. Если discontinued=true или остатка недостаточно, "
    "не оформляй заказ. Не утверждай, что заказ создан, пока create_service_order "
    "не вернул успешный результат с номером заказа. "
    "В результатах search_knowledge используй прежде всего структурированные facts, "
    "а не пересказ инструмента. Значения цены и остатка бери только из ERP."
)


def _available_tool_schemas(
    question: str,
    steps: list[ToolStep],
) -> list[dict]:
    """Возвращает только инструменты, подходящие для вопроса и текущего шага."""
    lowered = question.casefold()
    asks_global = bool(
        re.search(
            r"чаще\s+всего|наиболее\s+част|больше\s+всего|"
            r"наибольш\w*|максимальн\w*|статистик\w*|частот\w*|"
            r"снят\w*\s+с\s+производств",
            lowered,
        )
    )
    asks_price = bool(
        re.search(r"сколько\s+стоит|стоимост\w*|цен\w*", lowered)
    )
    asks_stock = bool(
        re.search(
            r"остат\w*|на\s+складе|в\s+наличии|сколько\s+(?:штук|шт)|есть\s+ли",
            lowered,
        )
    )
    asks_order = _asks_for_order(question)

    used = {step.tool for step in steps}
    available = []

    for schema in TOOL_SCHEMAS:
        function = schema.get("function", {})
        name = function.get("name")

        if name == "search_knowledge":
            if asks_global or name in used:
                continue
        elif name == "summarize_topic":
            if not asks_global or name in used:
                continue
        elif name == "check_stock" and not (asks_stock or asks_order):
            continue
        elif name == "get_price" and not asks_price:
            continue
        elif name == "create_service_order" and not asks_order:
            continue

        available.append(schema)

    return available



def _compact_tool_history(steps: list[ToolStep]) -> str:
    """Формирует компактную историю результатов инструментов для LLM.

    Полные результаты остаются в ToolStep. Для контекста модели сохраняются
    только факты, необходимые для продолжения работы.
    Источники документов не передаются повторно.
    """
    facts: list[str] = []
    summaries: list[str] = []
    erp_results: list[str] = []
    errors: list[str] = []

    seen_facts: set[str] = set()

    erp_fields = (
        "part_id",
        "warehouse",
        "qty",
        "price_rub",
        "discontinued",
        "order_id",
        "status",
    )

    argument_fields = (
        "part_id",
        "warehouse",
        "qty",
        "model",
    )

    for step in steps:
        result = step.result if isinstance(step.result, dict) else {}

        error = step.error or result.get("error")
        if error:
            errors.append(f"{step.tool}: {str(error)[:200]}")

        if step.tool == "search_knowledge":
            # Факты предпочтительнее сгенерированного ответа:
            # ответ поиска может ошибочно утверждать, что данных нет,
            # хотя необходимые отношения присутствуют в facts.
            for fact in result.get("facts", []):
                if not isinstance(fact, str) or fact in seen_facts:
                    continue

                seen_facts.add(fact)
                facts.append(fact[:220])

            # Если фактов нет, сохраняем хотя бы текст ответа.
            if not result.get("facts") and isinstance(result.get("answer"), str):
                summaries.append(result["answer"][:500])

        elif step.tool == "summarize_topic":
            answer = result.get("answer")
            if isinstance(answer, str) and answer.strip():
                summaries.append(answer[:900])

        elif step.tool in ERP_TOOLS:
            values = {
                key: result[key]
                for key in erp_fields
                if key in result
            }

            args = {
                key: step.args[key]
                for key in argument_fields
                if key in step.args
            }

            if values:
                erp_results.append(
                    f"{step.tool}: "
                    f"{json.dumps(values, ensure_ascii=False)}"
                )
            elif args and not error:
                erp_results.append(
                    f"{step.tool}: "
                    f"{json.dumps(args, ensure_ascii=False)}"
                )

    # Цепочки замен важнее остальных дополнительных фактов:
    # они необходимы для определения актуальной детали.
    priority_facts = [
        fact
        for fact in facts
        if "Цепочка замен:" in fact
        or "Последняя деталь в однозначной цепочке" in fact
    ]

    ordinary_facts = [
        fact
        for fact in facts
        if fact not in priority_facts
    ]

    prioritized_facts = priority_facts + ordinary_facts

    sections: list[str] = []

    if errors:
        sections.append(
            "Ошибки инструментов:\n" + "\n".join(errors[-4:])
        )

    if erp_results:
        sections.append(
            "Результаты ERP:\n" + "\n".join(erp_results[-4:])
        )

    if summaries:
        sections.append(
            "Ответы инструментов:\n" + "\n".join(summaries[-2:])
        )

    base = "\n\n".join(sections)

    header = "Подтверждённые факты графа:\n"
    included_facts: list[str] = []

    # Добавляем факты, пока не достигнут лимит контекста.
    current_length = len(base) + len(header)

    for fact in prioritized_facts[:32]:
        line = f"- {fact}\n"

        if current_length + len(line) > AGENT_HISTORY_CHAR_LIMIT:
            break

        included_facts.append(line)
        current_length += len(line)

    if included_facts:
        sections.append(header + "".join(included_facts).rstrip())

    return "\n\n".join(sections)


def _build_agent_messages(
    question: str,
    steps: list[ToolStep],
) -> list[dict]:
    """Строит новый компактный контекст для каждого обращения к LLM."""
    user_content = f"Вопрос пользователя:\n{question}"

    history = _compact_tool_history(steps)

    if history:
        user_content += (
            "\n\nРезультаты предыдущих вызовов инструментов:\n"
            + history
        )

    return [
        {
            "role": "system",
            "content": AGENT_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]



def _as_dict(value: Any) -> dict:
    """Приводит результат инструмента к словарю."""
    if isinstance(value, dict):
        return value
    return {"value": str(value)}


def _number_token(value: str | int | float) -> str | None:
    """Нормализует целое числовое значение, включая запись с пробелами."""
    if isinstance(value, bool):
        return None

    if isinstance(value, float) and not value.is_integer():
        return None
    if isinstance(value, int | float):
        return str(int(value))

    digits = re.sub(r"\D", "", str(value))
    return digits or None


def _extract_explicit_quantity(question: str) -> int | None:
    """Возвращает количество, явно названное пользователем.

    Если в вопросе указано несколько разных количеств, результат считается
    неоднозначным и заказ без уточнения не допускается.
    """
    quantities = {
        int(match.group(1))
        for match in QUANTITY_PATTERN.finditer(question)
        if int(match.group(1)) > 0
    }
    if len(quantities) != 1:
        return None
    return next(iter(quantities))


def _extract_price_values(text: str) -> set[str]:
    """Извлекает цены из текста по валютным обозначениям и словам-контекстам."""
    values = {
        token
        for match in CURRENCY_PATTERN.finditer(text)
        if (token := _number_token(match.group(1))) is not None
    }
    values.update(
        token
        for match in PRICE_LABEL_PATTERN.finditer(text)
        if (token := _number_token(match.group(1))) is not None
    )
    return values


def _extract_stock_values(text: str) -> set[str]:
    """Извлекает числа, обозначенные в тексте именно как складские остатки."""
    values: set[str] = set()

    for match in NUMBER_PATTERN.finditer(text):
        previous_breaks = list(SENTENCE_BREAK_PATTERN.finditer(text, 0, match.start()))
        sentence_start = previous_breaks[-1].end() if previous_breaks else 0
        before = text[sentence_start:match.start()]
        after = text[match.end():match.end() + 45]

        # Число должно следовать за складским контекстом либо непосредственно
        # перед ним (например, «3 шт. в наличии»). Точка после «шт.» считается
        # частью сокращения, а не границей предложения.
        recent_context = before[-80:]
        stock_contexts = list(STOCK_CONTEXT_PATTERN.finditer(recent_context))
        barriers = list(VALUE_CONTEXT_BARRIER_PATTERN.finditer(recent_context))
        stock_context_is_nearest = bool(stock_contexts) and (
            not barriers or stock_contexts[-1].start() > barriers[-1].start()
        )
        has_stock_context = bool(
            stock_context_is_nearest
            or re.match(
                r"\s*(?:(?:шт(?:\.|ук(?:а|и)?)?|штук(?:а|и)?|"
                r"единиц(?:а|ы)?)\s*)?\.?\s*"
                r"(?:в\s+наличии|на\s+складе)",
                after,
                re.IGNORECASE,
            )
        )
        if has_stock_context:
            token = _number_token(match.group(1))
            if token is not None:
                values.add(token)

    return values


def _tool_result_codes(result: dict) -> set[str]:
    """Находит коды, появившиеся в любом поле ответа инструмента."""
    serialized = json.dumps(result, ensure_ascii=False, default=str)
    return set(find_codes(serialized))


def _validate_answer(
    answer: str,
    *,
    known_codes: set[str],
    known_prices: set[str],
    known_stocks: set[str],
) -> dict[str, list[str]]:
    """Ищет в ответе коды, цены и остатки без подтверждения."""
    unsupported_codes = sorted(set(find_codes(answer)) - known_codes)
    unsupported_prices = sorted(_extract_price_values(answer) - known_prices)
    unsupported_stocks = sorted(_extract_stock_values(answer) - known_stocks)

    return {
        "codes": unsupported_codes,
        "prices": unsupported_prices,
        "stocks": unsupported_stocks,
    }


def _has_unsupported_facts(violations: dict[str, list[str]]) -> bool:
    return any(violations.values())


def _interrupted_result(
    steps: list[ToolStep],
    sources: set[str],
) -> AgentResult:
    details = []
    for step in steps:
        if step.error or not isinstance(step.result, dict):
            continue

        if isinstance(step.result.get("answer"), str):
            details.append(f"{step.tool}: {step.result['answer']}")
            continue

        useful_fields = {
            key: step.result[key]
            for key in (
                "part_id",
                "warehouse",
                "qty",
                "price_rub",
                "discontinued",
                "order_id",
                "status",
            )
            if key in step.result
        }
        if useful_fields:
            details.append(
                f"{step.tool}: "
                f"{json.dumps(useful_fields, ensure_ascii=False, default=str)}"
            )

    answer = (
        "Работа прервана: достигнут лимит вызовов инструментов. "
        "Полученные промежуточные результаты сохранены."
    )
    if details:
        answer += " Уже полученные данные: " + "; ".join(details)

    return AgentResult(
        answer=answer,
        steps=steps,
        sources=sorted(sources),
    )



def _warehouse_from_question(question: str) -> str | None:
    """Извлекает явно названный склад."""
    warehouses = ("Москва", "Санкт-Петербург", "Казань", "Новосибирск")
    lowered = question.casefold()
    return next((item for item in warehouses if item.casefold() in lowered), None)


def _resolve_part_from_knowledge(
    question: str,
    steps: list[ToolStep],
) -> str | None:
    """Находит деталь по цепочке модель/ошибка -> компонент -> деталь -> замена."""
    codes = find_codes(question)
    explicit_parts = {code for code in codes if code_type(code) == "Part"}
    if len(explicit_parts) == 1:
        return next(iter(explicit_parts))
    if explicit_parts:
        return None

    models = {code for code in codes if code_type(code) == "Model"}
    errors = {code for code in codes if code_type(code) == "ErrorCode"}
    facts = [
        fact
        for step in steps
        if step.tool == "search_knowledge" and isinstance(step.result, dict)
        for fact in step.result.get("facts", [])
        if isinstance(fact, str)
    ]

    model_parts: set[str] = set()
    target_components: set[str] = set()
    part_components: dict[str, set[str]] = {}
    replacements: dict[str, set[str]] = {}
    relation_pattern = re.compile(r"^(.+?)\s+-([A-Z_]+)->\s+(.+?)$")

    for fact in facts:
        match = relation_pattern.match(fact.strip())
        if not match:
            continue
        subject, relation, object_name = match.groups()
        subject_codes = set(find_codes(subject))
        object_codes = set(find_codes(object_name))

        if relation == "HAS_PART" and subject_codes.intersection(models):
            model_parts.update(
                code for code in object_codes if code_type(code) == "Part"
            )
        elif relation == "INDICATES" and subject_codes.intersection(errors):
            target_components.add(object_name)
        elif relation == "FAILED" and subject_codes.intersection(models):
            target_components.add(object_name)
        elif relation == "IS_A":
            for code in subject_codes:
                if code_type(code) == "Part":
                    part_components.setdefault(code, set()).add(object_name)
        elif relation == "REPLACED_BY":
            for code in subject_codes:
                if code_type(code) == "Part":
                    replacements.setdefault(code, set()).update(
                        item for item in object_codes if code_type(item) == "Part"
                    )

    if not target_components:
        lowered = question.casefold()
        target_components = {
            component
            for components in part_components.values()
            for component in components
            if component.casefold() in lowered
        }

    candidates = {
        part_id
        for part_id, components in part_components.items()
        if target_components.intersection(components)
        and (not models or part_id in model_parts)
    }
    if len(candidates) != 1:
        return None

    current = next(iter(candidates))
    visited = {current}
    while True:
        next_parts = replacements.get(current, set())
        if not next_parts:
            return current
        if len(next_parts) != 1:
            return None
        current = next(iter(next_parts))
        if current in visited:
            return None
        visited.add(current)


def _latest_stock_result(
    steps: list[ToolStep],
    part_id: str,
    warehouse: str,
) -> dict | None:
    for step in reversed(steps):
        if (
            step.tool == "check_stock"
            and isinstance(step.args, dict)
            and step.args.get("part_id") == part_id
            and step.args.get("warehouse") == warehouse
            and isinstance(step.result, dict)
        ):
            return step.result
    return None


def _blocked_order_answer(
    question: str,
    steps: list[ToolStep],
    part_id: str,
    warehouse: str,
) -> str | None:
    """Проверяет, не снята ли деталь с производства и хватает ли остатка."""
    stock = _latest_stock_result(steps, part_id, warehouse)
    if stock is None:
        return None

    quantity = stock.get("qty")
    if stock.get("discontinued") is True:
        if isinstance(quantity, int) and not isinstance(quantity, bool):
            return (
                f"Заказ не оформлен: деталь {part_id} снята с производства. "
                f"При этом на складе «{warehouse}» числится {quantity} шт. "
                "Наличие остатка не означает, что деталь доступна для заказа."
            )
        return f"Заказ не оформлен: деталь {part_id} снята с производства."

    requested = _extract_explicit_quantity(question)
    if (
        requested is not None
        and isinstance(quantity, int)
        and not isinstance(quantity, bool)
        and quantity < requested
    ):
        return (
            f"Заказ не оформлен: запрошено {requested} шт. детали {part_id}, "
            f"но на складе «{warehouse}» числится только {quantity} шт."
        )
    return None


def _fallback_is_allowed(steps: list[ToolStep]) -> bool:
    """Разрешает резервный вызов только до ERP-действий или после поиска знаний.

    Ошибочные/неизвестные вызовы и уже выполненные ERP-инструменты не должны
    запускать второй, неожиданный сценарий после текстового ответа модели.
    Успешный check_stock можно продолжить оформлением заказа.
    """
    if not steps:
        return True

    allowed_tools = {"search_knowledge", "check_stock"}
    return all(
        step.tool in allowed_tools and not step.error
        for step in steps
    )


def _fallback_tool_request(
    question: str,
    steps: list[ToolStep],
) -> tuple[str, dict] | None:
    """Выбирает обязательный инструмент, если модель собирается ответить без него."""
    lowered = question.casefold()
    codes = find_codes(question)
    part_codes = [code for code in codes if code_type(code) == "Part"]
    part_id = part_codes[0] if len(part_codes) == 1 else None
    if part_id is None:
        part_id = _resolve_part_from_knowledge(question, steps)

    warehouse = _warehouse_from_question(question)
    asks_stock = bool(
        re.search(
            r"на\s+складе|в\s+наличии|остат\w*|сколько\s+штук|есть\s+ли",
            lowered,
        )
    )
    asks_price = bool(
        re.search(r"сколько\s+стоит|стоимость|цена|по\s+какой\s+цене", lowered)
    )
    asks_order = bool(
        re.search(
            r"\b(?:оформи|создай|сформируй)\s+заказ\b|\bзакажи\s+\d+\b",
            lowered,
        )
    )

    if re.search(r"снят\w*\s+с\s+производств", lowered) or re.search(
        r"чаще\s+всего|наиболее\s+част|по\s+статистик", lowered
    ):
        if not any(step.tool == "summarize_topic" for step in steps):
            return "summarize_topic", {"question": question}

    if part_id:
        if asks_order and warehouse:
            stock = _latest_stock_result(steps, part_id, warehouse)
            if stock is None:
                return "check_stock", {"part_id": part_id, "warehouse": warehouse}
            if _blocked_order_answer(question, steps, part_id, warehouse):
                return None
            requested = _extract_explicit_quantity(question)
            if requested is not None and not stock.get("error"):
                return "create_service_order", {
                    "part_id": part_id,
                    "warehouse": warehouse,
                    "qty": requested,
                }

        if asks_stock and warehouse:
            already_checked = any(
                step.tool == "check_stock"
                and isinstance(step.args, dict)
                and step.args.get("part_id") == part_id
                and step.args.get("warehouse") == warehouse
                for step in steps
            )
            if not already_checked:
                return "check_stock", {"part_id": part_id, "warehouse": warehouse}

        if asks_price and not any(
            step.tool == "get_price"
            and isinstance(step.args, dict)
            and step.args.get("part_id") == part_id
            for step in steps
        ):
            return "get_price", {"part_id": part_id}

    if any(step.tool == "search_knowledge" for step in steps):
        return None

    needs_knowledge = any(
        code_type(code) in {"Model", "ErrorCode", "Procedure"} for code in codes
    ) or bool(
        re.search(
            r"ошибк\w*|регламент\w*|модел\w*|узел\w*|компонент\w*|"
            r"детал\w*|поставщик\w*|поставля\w*|в\s+каких\s+моделях",
            lowered,
        )
    )
    if needs_knowledge:
        return "search_knowledge", {"question": question}

    # Не повторяем запрос к ERP, если нужный инструмент уже вызывался.
    # В частности, get_price и check_stock выше проверяются с учётом аргументов.
    return None


def _extract_warehouse(question: str) -> str | None:
    """Совместимое имя для извлечения склада в цикле агента."""
    return _warehouse_from_question(question)


def _asks_for_order(question: str) -> bool:
    """Проверяет, просит ли пользователь именно оформить заказ."""
    return bool(
        re.search(
            r"\b(?:оформи|создай|сформируй)\s+заказ\b|\bзакажи\s+\d+\b",
            question,
            re.IGNORECASE,
        )
    )


def _record_tool_result(
    name: str,
    args: dict,
    result: dict,
    steps: list[ToolStep],
    sources: set[str],
) -> None:
    """Сохраняет результат инструмента и его источники."""
    error = result.get("error")
    steps.append(
        ToolStep(
            tool=name,
            args=dict(args),
            result=result,
            error=str(error) if error is not None else None,
        )
    )
    if name in {"search_knowledge", "summarize_topic"}:
        tool_sources = result.get("sources", [])
        if isinstance(tool_sources, list):
            sources.update(str(source) for source in tool_sources)

def run_agent(
    question: str,
    llm: LLMClient,
    tools: dict[str, Callable[..., dict]],
    *,
    max_steps: int = 6,
    confirm: bool = False,
) -> AgentResult:
    """Ведёт цикл агента с проверкой фактов и безопасным созданием заказов."""
    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    steps: list[ToolStep] = []
    sources: set[str] = set()
    known_codes = set(find_codes(question))
    known_prices = _extract_price_values(question)
    known_stocks = _extract_stock_values(question)
    explicitly_requested_qty = _extract_explicit_quantity(question)

    # Кеш действует в пределах одного запуска агента.
    cache: dict[str, dict] = {}
    tool_calls_count = 0
    knowledge_refused = False

    if max_steps < 1:
        return AgentResult(
            answer="Работа не выполнена: лимит вызовов инструментов равен нулю.",
            steps=steps,
            sources=[],
        )

    while True:
        if tool_calls_count >= max_steps:
            return _interrupted_result(steps, sources)

        if _asks_for_order(question):
            pending_part_id = _resolve_part_from_knowledge(question, steps)
            pending_warehouse = _extract_warehouse(question)
            if pending_part_id and pending_warehouse:
                blocked_answer = _blocked_order_answer(
                    question,
                    steps,
                    pending_part_id,
                    pending_warehouse,
                )
                if blocked_answer:
                    return AgentResult(
                        answer=blocked_answer,
                        steps=steps,
                        sources=sorted(sources),
                    )

        response = llm.chat(
            messages,
            tools=_available_tool_schemas(question, steps),
        )

        if not response.tool_calls:
            fallback = (
                _fallback_tool_request(question, steps)
                if _fallback_is_allowed(steps)
                else None
            )

            if fallback is not None:
                fallback_name, fallback_args = fallback

                if fallback_name not in tools:
                    return AgentResult(
                        answer=(
                            f"Не удалось выполнить запрос: инструмент "
                            f"{fallback_name} недоступен."
                        ),
                        steps=steps,
                        sources=sorted(sources),
                        refused=True,
                    )

                if tool_calls_count >= max_steps:
                    return _interrupted_result(steps, sources)

                if fallback_name == "create_service_order":
                    requested_qty = _extract_explicit_quantity(question)
                    if requested_qty is None:
                        return AgentResult(
                            answer="Уточните количество деталей перед созданием заказа.",
                            steps=steps,
                            sources=sorted(sources),
                        )

                    if fallback_args.get("qty") != requested_qty:
                        return AgentResult(
                            answer=(
                                "Количество в заказе не совпадает с количеством, "
                                "указанным пользователем. Уточните количество."
                            ),
                            steps=steps,
                            sources=sorted(sources),
                        )

                    if not confirm:
                        pending_result = {"status": "pending_confirmation"}
                        steps.append(
                            ToolStep(
                                tool=fallback_name,
                                args=dict(fallback_args),
                                result=pending_result,
                            )
                        )
                        return AgentResult(
                            answer="Заказ подготовлен и ожидает подтверждения пользователя.",
                            steps=steps,
                            sources=sorted(sources),
                            needs_confirmation=True,
                            pending_action={
                                "tool": fallback_name,
                                "args": dict(fallback_args),
                            },
                        )

                tool_calls_count += 1
                fallback_result = None
                last_unavailable_error = None
                retries = 3 if fallback_name in ERP_TOOLS else 1

                for attempt in range(retries):
                    try:
                        fallback_result = _as_dict(
                            tools[fallback_name](**fallback_args)
                        )
                        break
                    except ErpUnavailable as exc:
                        last_unavailable_error = str(exc)
                        if attempt + 1 == retries:
                            break
                    except Exception as exc:
                        fallback_result = {
                            "error": f"Ошибка инструмента {fallback_name}: {exc}"
                        }
                        break

                if fallback_result is None:
                    fallback_result = {
                        "error": (
                            "Данные ERP получить не удалось после трёх попыток. "
                            f"Последняя ошибка: {last_unavailable_error or 'неизвестная ошибка'}"
                        )
                    }

                _record_tool_result(
                    fallback_name,
                    fallback_args,
                    fallback_result,
                    steps,
                    sources,
                )

                known_codes.update(_tool_result_codes(fallback_result))
                serialized_result = json.dumps(
                    fallback_result,
                    ensure_ascii=False,
                    default=str,
                )
                known_prices.update(_extract_price_values(serialized_result))
                known_stocks.update(_extract_stock_values(serialized_result))

                if fallback_name == "get_price":
                    price = fallback_result.get("price_rub")
                    if isinstance(price, int | float | str) and not isinstance(price, bool):
                        token = _number_token(price)
                        if token is not None:
                            known_prices.add(token)

                if fallback_name == "check_stock":
                    quantity = fallback_result.get("qty")
                    if isinstance(quantity, int | float | str) and not isinstance(quantity, bool):
                        token = _number_token(quantity)
                        if token is not None:
                            known_stocks.add(token)

                if fallback_name in {"search_knowledge", "summarize_topic"}:
                    knowledge_refused = (
                        knowledge_refused or bool(fallback_result.get("refused"))
                    )
                    if fallback_result.get("refused") or fallback_result.get("error"):
                        return AgentResult(
                            answer=fallback_result.get(
                                "answer",
                                fallback_result.get(
                                    "error",
                                    "В базе знаний не найдено достаточно информации.",
                                ),
                            ),
                            steps=steps,
                            sources=sorted(sources),
                            refused=True,
                        )

                # Если пользователь просит оформить заказ, не продолжаем
                # после обнаружения снятой с производства детали.
                if _asks_for_order(question):
                    part_id = fallback_args.get("part_id")
                    warehouse = fallback_args.get("warehouse")
                    if isinstance(part_id, str) and isinstance(warehouse, str):
                        blocked_answer = _blocked_order_answer(
                            question,
                            steps,
                            part_id,
                            warehouse,
                        )
                        if blocked_answer:
                            return AgentResult(
                                answer=blocked_answer,
                                steps=steps,
                                sources=sorted(sources),
                            )

                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Обязательный вызов инструмента {fallback_name} "
                            "выполнен. Результат:\n"
                            f"{json.dumps(fallback_result, ensure_ascii=False, default=str)}\n\n"
                            "Используй только подтверждённые значения. "
                            "Для check_stock помни: qty — количество на складе, "
                            "а discontinued — отдельный статус производства. "
                            "Не делай вывод, что остаток равен нулю, если qty больше нуля."
                        ),
                    }
                )
                continue

            # Не разрешаем модели самостоятельно придумывать объяснение
            # неизвестного кода ошибки или модели, если поиск не запускался.
            answer = response.content.strip()
            violations = _validate_answer(
                answer,
                known_codes=known_codes,
                known_prices=known_prices,
                known_stocks=known_stocks,
            )

            if _has_unsupported_facts(violations):
                history = _compact_tool_history(steps)
                correction = llm.chat(
                    [
                        {
                            "role": "system",
                            "content": (
                                "Исправь ответ, используя только вопрос и результаты "
                                "вызовов инструментов. Удали все неподтверждённые коды "
                                "деталей, цены и остатки. Не заменяй их догадками. "
                                "Если данных недостаточно, прямо сообщи об этом."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Вопрос: {question}\n"
                                "Результаты инструментов:\n"
                                f"{history}\n"
                                f"Неподтверждённые сведения: "
                                f"{json.dumps(violations, ensure_ascii=False)}\n"
                                f"Ответ, который нужно исправить: {answer}"
                            ),
                        },
                    ],
                    max_tokens=384,
                )
                answer = correction.content.strip()

                remaining_violations = _validate_answer(
                    answer,
                    known_codes=known_codes,
                    known_prices=known_prices,
                    known_stocks=known_stocks,
                )
                if _has_unsupported_facts(remaining_violations):
                    return AgentResult(
                        answer=(
                            "В базе знаний и результатах ERP не найдено "
                            "достаточного подтверждения ответа."
                        ),
                        steps=steps,
                        sources=sorted(sources),
                        refused=True,
                    )

            answer_is_refusal = bool(
                re.search(
                    r"(?:в базе знаний не найдено|недостаточно данных|"
                    r"не удалось найти|не удалось подтвердить|"
                    r"данные ERP получить не удалось|"
                    r"данные о цене.*отсутствуют в ERP|"
                    r"информация о цене.*отсутствует в ERP|"
                    r"данные о наличии.*отсутствуют в ERP|"
                    r"информация о наличии.*отсутствует в ERP|"
                    r"не могу ответить|не могу предоставить информацию)",
                    answer,
                    re.IGNORECASE,
                )
            )
            return AgentResult(
                answer=answer,
                steps=steps,
                sources=sorted(sources),
                refused=knowledge_refused or answer_is_refusal,
            )

        # Сохраняем вызовы ассистента перед добавлением ответов инструментов.
        messages.append(response.as_message())

        for call in response.tool_calls:
            if tool_calls_count >= max_steps:
                return _interrupted_result(steps, sources)

            # Считаем каждый запрос function calling, в том числе отклонённый
            # проверками и повторный запрос, для которого используется кеш.
            tool_calls_count += 1
            name = call.name
            raw_args = call.arguments
            args = raw_args if isinstance(raw_args, dict) else {"_raw": str(raw_args)}
            step_args = dict(args)

            if name in {"search_knowledge", "summarize_topic"} and any(
                previous.tool == name for previous in steps
            ):
                result = {
                    "error": (
                        f"Инструмент {name} уже выполнялся. "
                        "Используй полученный результат вместо повторного поиска."
                    )
                }
                steps.append(
                    ToolStep(tool=name, args=step_args, error=result["error"])
                )
                messages.append(tool_message(call, result))
                continue

            if name not in tools:
                result = {"error": f"Неизвестный инструмент: {name}"}
                steps.append(ToolStep(tool=name, args=step_args, error=result["error"]))
                messages.append(tool_message(call, result))
                continue

            if not isinstance(raw_args, dict) or "_raw" in raw_args:
                result = {"error": "Не удалось разобрать аргументы инструмента."}
                steps.append(ToolStep(tool=name, args=step_args, error=result["error"]))
                messages.append(tool_message(call, result))
                continue

            part_id = args.get("part_id")
            if part_id is not None:
                if not isinstance(part_id, str) or not PART_PATTERN.fullmatch(part_id):
                    result = {"error": "Некорректный номер детали."}
                    steps.append(
                        ToolStep(tool=name, args=step_args, error=result["error"])
                    )
                    messages.append(tool_message(call, result))
                    continue

                if part_id not in known_codes:
                    result = {
                        "error": (
                            "Деталь не подтверждена вопросом или результатами "
                            "предыдущих инструментов. Сначала вызови search_knowledge."
                        )
                    }
                    steps.append(
                        ToolStep(tool=name, args=step_args, error=result["error"])
                    )
                    messages.append(tool_message(call, result))
                    continue

            if name == "create_service_order":
                if explicitly_requested_qty is None:
                    error = "Уточните количество деталей перед созданием заказа."
                    result = {"error": error}
                    steps.append(ToolStep(tool=name, args=step_args, error=error))
                    messages.append(tool_message(call, result))
                    return AgentResult(
                        answer=error,
                        steps=steps,
                        sources=sorted(sources),
                    )

                requested_qty = args.get("qty")
                if (
                    isinstance(requested_qty, bool)
                    or not isinstance(requested_qty, int)
                    or requested_qty != explicitly_requested_qty
                ):
                    error = (
                        "Количество в заказе не совпадает с количеством, "
                        "указанным пользователем. Уточните количество."
                    )
                    result = {"error": error}
                    steps.append(ToolStep(tool=name, args=step_args, error=error))
                    messages.append(tool_message(call, result))
                    return AgentResult(
                        answer=error,
                        steps=steps,
                        sources=sorted(sources),
                    )

                order_part_id = args.get("part_id")
                order_warehouse = args.get("warehouse")
                if isinstance(order_part_id, str) and isinstance(order_warehouse, str):
                    blocked_answer = _blocked_order_answer(
                        question,
                        steps,
                        order_part_id,
                        order_warehouse,
                    )
                    if blocked_answer:
                        return AgentResult(
                            answer=blocked_answer,
                            steps=steps,
                            sources=sorted(sources),
                        )

                if not confirm:
                    pending_result = {"status": "pending_confirmation"}
                    steps.append(
                        ToolStep(
                            tool=name,
                            args=step_args,
                            result=pending_result,
                        )
                    )
                    return AgentResult(
                        answer=(
                            "Заказ подготовлен и ожидает подтверждения пользователя."
                        ),
                        steps=steps,
                        sources=sorted(sources),
                        needs_confirmation=True,
                        pending_action={"tool": name, "args": step_args},
                    )

            cache_key = json.dumps(
                {"tool": name, "args": args},
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )

            if cache_key in cache:
                result = cache[cache_key]
            else:
                result = None
                last_unavailable_error = None
                retries = 3 if name in ERP_TOOLS else 1

                for attempt in range(retries):
                    try:
                        result = _as_dict(tools[name](**args))
                        last_unavailable_error = None
                        break
                    except ErpUnavailable as exc:
                        last_unavailable_error = str(exc)
                        if attempt + 1 == retries:
                            break
                    except Exception as exc:
                        result = {
                            "error": f"Ошибка инструмента {name}: {exc}",
                        }
                        break

                if result is None:
                    result = {
                        "error": (
                            "Данные ERP получить не удалось после трёх попыток. "
                            "Последняя ошибка: "
                            f"{last_unavailable_error or 'неизвестная ошибка'}"
                        )
                    }

                cache[cache_key] = result

            step_error = result.get("error")
            steps.append(
                ToolStep(
                    tool=name,
                    args=step_args,
                    result=result,
                    error=str(step_error) if step_error is not None else None,
                )
            )
            messages.append(tool_message(call, result))

            # Коды подтверждаются любым предыдущим результатом инструмента.
            known_codes.update(_tool_result_codes(result))
            serialized_result = json.dumps(result, ensure_ascii=False, default=str)
            known_prices.update(_extract_price_values(serialized_result))
            known_stocks.update(_extract_stock_values(serialized_result))

            if name in {"search_knowledge", "summarize_topic"}:
                tool_sources = result.get("sources", [])
                if isinstance(tool_sources, list):
                    sources.update(str(source) for source in tool_sources)
                knowledge_refused = knowledge_refused or bool(result.get("refused"))

                # Отказ поиска — конечный результат. Повторный поиск не добавит
                # информации и раньше приводил к исчерпанию лимита шагов.
                if result.get("refused") or (
                    name == "search_knowledge"
                    and isinstance(result.get("answer"), str)
                    and re.search(
                        r"в базе знаний не найдено|недостаточно данных",
                        result["answer"],
                        re.IGNORECASE,
                    )
                ):
                    refusal_text = result.get("answer") or result.get("error") or (
                        "В базе знаний не найдено достаточно информации."
                    )
                    return AgentResult(
                        answer=str(refusal_text),
                        steps=steps,
                        sources=sorted(sources),
                        refused=True,
                    )

            if name == "get_price":
                price = result.get("price_rub")
                if isinstance(price, int | float | str) and not isinstance(price, bool):
                    token = _number_token(price)
                    if token is not None:
                        known_prices.add(token)

            if name == "check_stock":
                quantity = result.get("qty")
                if isinstance(quantity, int | float | str) and not isinstance(quantity, bool):
                    token = _number_token(quantity)
                    if token is not None:
                        known_stocks.add(token)