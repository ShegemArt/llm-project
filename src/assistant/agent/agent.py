"""Цикл агента. Требования FR-7, FR-8, FR-9 и FR-10."""

import json
import re
from collections.abc import Callable
from typing import Any

from assistant.agent.tools import TOOL_SCHEMAS
from assistant.infra.erp_client import ErpUnavailable
from assistant.infra.grounding import find_codes
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
        {
            "role": "system",
            "content": (
                "Ты ассистент сервисной службы. Используй инструменты для "
                "получения фактов. Не выдумывай коды деталей, цены, остатки "
                "и результаты заказов. Номер детали должен быть подтверждён "
                "вопросом или результатом предыдущего инструмента. "
                "Количество заказа должно быть явно указано пользователем. "
                "Не создавай заказ без явного подтверждения пользователя. "
                "Если ERP недоступна, сообщи об этом и не подменяй отсутствие "
                "данных нулевым остатком."
            ),
        },
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

        response = llm.chat(messages, tools=TOOL_SCHEMAS)

        if not response.tool_calls:
            answer = response.content.strip()
            violations = _validate_answer(
                answer,
                known_codes=known_codes,
                known_prices=known_prices,
                known_stocks=known_stocks,
            )

            if _has_unsupported_facts(violations):
                history = json.dumps(messages, ensure_ascii=False, default=str)
                correction = llm.chat(
                    [
                        {
                            "role": "system",
                            "content": (
                                "Исправь ответ, используя только вопрос и "
                                "результаты вызовов инструментов. Удали все "
                                "неподтверждённые коды деталей, цены и остатки. "
                                "Не заменяй их догадками. Если данных недостаточно, "
                                "прямо сообщи об этом."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Вопрос: {question}\n"
                                "История: "
                                f"{history}"
                                "\n"
                                f"Неподтверждённые сведения: "
                                f"{json.dumps(violations, ensure_ascii=False)}\n"
                                f"Ответ, который нужно исправить: {answer}"
                            ),
                        },
                    ]
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
                    r"данные ERP получить не удалось|не могу ответить)",
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

