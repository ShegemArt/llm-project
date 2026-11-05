"""Поиск по графу знаний: local search и global map-reduce (FR-4, FR-5, FR-10)."""

import json
import re
import networkx as nx

from assistant.infra.grounding import (
    code_type,
    find_codes,
    find_names,
)
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import (
    MAP_SCHEMA,
    Answer,
    KnowledgeBase,
)


def _sources(graph: nx.DiGraph, nodes: set[str]) -> list[str]:
    """Преобразует источники чанков в идентификаторы документов."""
    sources = set()

    for node in nodes:
        if node not in graph:
            continue

        for chunk_id in graph.nodes[node].get("sources", []):
            sources.add(chunk_id.split("#", 1)[0])

    for u, v, data in graph.edges(data=True):
        if u in nodes and v in nodes:
            for chunk_id in data.get("sources", []):
                sources.add(chunk_id.split("#", 1)[0])

    return sorted(sources)


def _format_edge(
    graph: nx.DiGraph,
    u: str,
    v: str,
    data: dict,
) -> str:
    """Форматирует направленное отношение для контекста LLM."""
    left = graph.nodes[u]["name"]
    right = graph.nodes[v]["name"]
    relation = data.get("type", "RELATED_TO")

    return f"{left} -{relation}-> {right}"


def _replacement_paths(
    graph: nx.DiGraph,
    start_nodes: set[str],
) -> dict[str, list[str]]:
    """Находит однозначные цепочки замен от исходных деталей до конечных."""
    paths = {}

    for start in sorted(start_nodes):
        if start not in graph or graph.nodes[start].get("type") != "Part":
            continue

        path = [start]
        visited = {start}
        current = start

        while True:
            candidates = [
                target
                for _, target, data in graph.out_edges(current, data=True)
                if data.get("type") == "REPLACED_BY"
            ]

            if not candidates:
                break

            # При нескольких заменах нельзя произвольно выбирать одну.
            if len(candidates) != 1:
                path = []
                break

            next_node = candidates[0]

            # Защита от циклов.
            if next_node in visited:
                path = []
                break

            path.append(next_node)
            visited.add(next_node)
            current = next_node

        if len(path) > 1:
            paths[start] = path

    return paths


def _expand_part_context(
    graph: nx.DiGraph,
    seed_nodes: set[str],
) -> tuple[set[str], set[tuple[str, str]], dict[str, list[str]]]:
    """Расширяет контекст найденных деталей связями типа, поставщиком и заменами."""
    expanded_nodes = set(seed_nodes)
    selected_edges: set[tuple[str, str]] = set()

    part_nodes = {
        node
        for node in expanded_nodes
        if graph.nodes[node].get("type") == "Part"
    }

    replacement_paths = _replacement_paths(graph, part_nodes)

    # Добавляем все рёбра однозначных цепочек замен.
    for path in replacement_paths.values():
        for source, target in zip(path[:-1], path[1:], strict=True):
            selected_edges.add((source, target))
            expanded_nodes.update((source, target))

    # После добавления замен в expanded_nodes появились новые детали.
    # Для каждой детали ищем её тип, поставщика и следующие замены.
    part_nodes = {
        node
        for node in expanded_nodes
        if graph.nodes[node].get("type") == "Part"
    }

    for part_node in part_nodes:
        for _, target, data in graph.out_edges(part_node, data=True):
            if data.get("type") not in {
                "IS_A",
                "SUPPLIED_BY",
                "REPLACED_BY",
            }:
                continue

            selected_edges.add((part_node, target))
            expanded_nodes.add(target)

    return expanded_nodes, selected_edges, replacement_paths


def _global_graph_statistics(graph: nx.DiGraph) -> dict:
    """Считает глобальные частоты отказов и снятые с производства детали.

    Вес FAILED-ребра — количество разных чанков, в которых зафиксирован
    отказ модели. Статистика рассчитывается Python, а не языковой моделью.
    """
    model_to_lines: dict[str, set[str]] = {}
    model_to_parts: dict[str, set[str]] = {}

    for source, target, data in graph.edges(data=True):
        relation_type = data.get("type")

        if relation_type == "BELONGS_TO":
            model_to_lines.setdefault(source, set()).add(
                graph.nodes[target].get("name", target)
            )

        elif relation_type == "HAS_PART":
            model_to_parts.setdefault(source, set()).add(target)

    failures_by_line_component: dict[tuple[str, str], int] = {}
    failures_by_line_model: dict[tuple[str, str], int] = {}

    for model_node, component_node, data in graph.edges(data=True):
        if data.get("type") != "FAILED":
            continue

        weight = data.get("weight", 1)

        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            weight = 1

        weight = int(weight)

        model_name = graph.nodes[model_node].get("name", model_node)
        component_name = graph.nodes[component_node].get(
            "name",
            component_node,
        )

        for line_name in model_to_lines.get(model_node, set()):
            component_key = (line_name, component_name)
            model_key = (line_name, model_name)

            failures_by_line_component[component_key] = (
                failures_by_line_component.get(component_key, 0) + weight
            )

            failures_by_line_model[model_key] = (
                failures_by_line_model.get(model_key, 0) + weight
            )

    # Учитываем только детали, которые установлены в модели линейки
    # и имеют исходящее отношение REPLACED_BY.
    discontinued_by_line: dict[str, set[str]] = {}

    for model_node, part_nodes in model_to_parts.items():
        for line_name in model_to_lines.get(model_node, set()):
            for part_node in part_nodes:
                replacements = [
                    target
                    for _, target, data in graph.out_edges(
                        part_node,
                        data=True,
                    )
                    if data.get("type") == "REPLACED_BY"
                ]

                if replacements:
                    discontinued_by_line.setdefault(line_name, set()).add(
                        graph.nodes[part_node].get("name", part_node)
                    )

    return {
        "failures_by_line_and_component": [
            {
                "line": line,
                "component": component,
                "count": count,
            }
            for (line, component), count
            in sorted(failures_by_line_component.items())
        ],
        "failures_by_line_and_model": [
            {
                "line": line,
                "model": model,
                "count": count,
            }
            for (line, model), count
            in sorted(failures_by_line_model.items())
        ],
        "discontinued_parts_by_line": [
            {
                "line": line,
                "parts": sorted(parts),
            }
            for line, parts in sorted(discontinued_by_line.items())
        ],
    }


def local_search(
    question: str,
    kb: KnowledgeBase,
    llm: LLMClient,
) -> Answer:
    """Ищет факты по опорным сущностям и отвечает только по найденным рёбрам."""
    graph = kb.graph
    anchor_nodes = set()

    codes = find_codes(question)

    for code in codes:
        kind = code_type(code)
        node_id = f"{kind}:{code}"

        if node_id in graph:
            anchor_nodes.add(node_id)

    for kind, name in find_names(question, kb.vocabulary):
        node_id = f"{kind}:{name}"

        if node_id in graph:
            anchor_nodes.add(node_id)

    for code in codes:
        node_id = f"{code_type(code)}:{code}"

        if node_id not in graph:
            return Answer(
                text="В базе знаний не найдено подтверждение для указанного кода.",
                refused=True,
            )

    if not anchor_nodes:
        return Answer(
            text="В базе знаний не найдено опорных сущностей для ответа.",
            refused=True,
        )

    selected_edges: set[tuple[str, str]] = set()
    undirected = graph.to_undirected()

    if len(anchor_nodes) >= 2:
        anchors = sorted(anchor_nodes)

        for index, source in enumerate(anchors):
            for target in anchors[index + 1:]:
                try:
                    paths = nx.all_simple_paths(
                        undirected,
                        source,
                        target,
                        cutoff=3,
                    )

                    for path in paths:
                        for u, v in zip(
                            path[:-1],
                            path[1:],
                            strict=True,
                        ):
                            # Поиск пути идёт по неориентированной версии графа,
                            # но в контекст добавляем только реальные направленные рёбра.
                            if graph.has_edge(u, v):
                                selected_edges.add((u, v))

                            if graph.has_edge(v, u):
                                selected_edges.add((v, u))

                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
    else:
        root = next(iter(anchor_nodes))
        distances = nx.single_source_shortest_path_length(
            undirected,
            root,
            cutoff=2,
        )

        neighbourhood = set(distances)

        for source, target in graph.edges():
            if source in neighbourhood and target in neighbourhood:
                selected_edges.add((source, target))

    path_nodes = set(anchor_nodes)

    for source, target in selected_edges:
        path_nodes.update((source, target))

    # Добавляем связи типа, поставщика и замены для найденных деталей.
    path_nodes, part_edges, replacement_paths = _expand_part_context(
        graph,
        path_nodes,
    )
    selected_edges.update(part_edges)

    facts = []

    for source, target in sorted(selected_edges):
        data = graph.get_edge_data(source, target)
        if data:
            facts.append(_format_edge(graph, source, target, data))

    # Дополнительные программно вычисленные факты о конечных деталях.
    # Они не добавляют новых связей в граф, а лишь явно показывают результат
    # прохождения уже существующей цепочки REPLACED_BY.
    for path in replacement_paths.values():
        names = [
            graph.nodes[node].get("name", node)
            for node in path
        ]

        facts.append(
            "Цепочка замен: " + " -> ".join(names)
        )
        facts.append(
            f"Последняя деталь в однозначной цепочке замен: {names[-1]}"
        )

    facts = list(dict.fromkeys(facts))

    if not facts:
        return Answer(
            text="В базе знаний не найдено подтверждающих фактов.",
            refused=True,
        )

    context = "\n".join(
        f"{index + 1}. {fact}"
        for index, fact in enumerate(facts)
    )[:8000]

    response = llm.chat(
        [
            {
                "role": "system",
                "content": (
                    "Отвечай только по переданным фактам. "
                    "Не добавляй отсутствующие коды, цены, остатки или связи. "
                    "Если вопрос спрашивает, какая деталь используется сейчас, "
                    "а в фактах есть однозначная цепочка REPLACED_BY, "
                    "назови последнюю деталь этой цепочки. "
                    "Не переноси поставщика со старой детали на новую, "
                    "если для новой детали нет отдельной связи SUPPLIED_BY. "
                    "Если фактов недостаточно или цепочка замен неоднозначна, "
                    "прямо сообщи об этом."
                ),
            },
            {
                "role": "user",
                "content": f"Вопрос:\n{question}\n\nФакты:\n{context}",
            },
        ]
    )

    return Answer(
        text=response.content,
        sources=_sources(graph, path_nodes),
        facts=facts,
    )

def _discontinued_parts_answer(
    question: str,
    kb: KnowledgeBase,
) -> Answer | None:
    """Отвечает на вопрос о снятых с производства деталях конкретной линейки."""
    if not re.search(
        r"снят\w*\s+с\s+производств",
        question,
        re.IGNORECASE,
    ):
        return None

    lines = [
        name
        for kind, name in find_names(question, kb.vocabulary)
        if kind == "ProductLine"
    ]

    if not lines:
        return None

    statistics = _global_graph_statistics(kb.graph)

    discontinued_by_line = {
        item["line"]: item["parts"]
        for item in statistics["discontinued_parts_by_line"]
    }

    graph = kb.graph
    selected_nodes: set[str] = set()
    facts: list[str] = []
    answer_parts = []

    for line in sorted(set(lines)):
        parts = discontinued_by_line.get(line, [])

        line_node = f"ProductLine:{line}"
        selected_nodes.add(line_node)

        if not parts:
            answer_parts.append(
                f"Для линейки «{line}» снятые с производства детали не найдены."
            )
            continue

        replacements_by_part = {}

        for source, target, data in graph.edges(data=True):
            if (
                data.get("type") != "BELONGS_TO"
                or target != line_node
            ):
                continue

            selected_nodes.add(source)

            for _, part_node, part_data in graph.out_edges(
                source,
                data=True,
            ):
                if part_data.get("type") != "HAS_PART":
                    continue

                part_name = graph.nodes[part_node].get("name", part_node)

                if part_name not in parts:
                    continue

                selected_nodes.add(part_node)

                replacements = [
                    replacement
                    for _, replacement, replacement_data in graph.out_edges(
                        part_node,
                        data=True,
                    )
                    if replacement_data.get("type") == "REPLACED_BY"
                ]

                for replacement in replacements:
                    selected_nodes.add(replacement)
                    facts.append(
                        _format_edge(
                            graph,
                            part_node,
                            replacement,
                            graph.get_edge_data(part_node, replacement),
                        )
                    )

                replacements_by_part[part_name] = [
                    graph.nodes[node].get("name", node)
                    for node in replacements
                ]

        details = []

        for part in sorted(parts):
            replacements = replacements_by_part.get(part, [])

            if replacements:
                details.append(
                    f"{part} (замена: {', '.join(sorted(replacements))})"
                )
            else:
                details.append(part)

        answer_parts.append(
            f"В линейке «{line}» сняты с производства: "
            + "; ".join(details)
            + "."
        )

    return Answer(
        text="\n".join(answer_parts),
        sources=_sources(graph, selected_nodes),
        facts=list(dict.fromkeys(facts)),
    )


def global_search(
    question: str,
    kb: KnowledgeBase,
    llm: LLMClient,
) -> Answer:
    """Объединяет сводки сообществ и точные глобальные статистики графа."""
    partial_answers = []
    sources = set()

    for community in kb.communities:
        if not community.summary.strip():
            continue

        prompt = (
            f"Вопрос: {question}\n"
            f"Сводка сообщества:\n{community.summary}\n\n"
            "Оцени, помогает ли сводка ответить на вопрос. Верни JSON с полями "
            "answer и score от 0 до 10. score — только релевантность этой сводки, "
            "а не число отказов или частота события. Если данных недостаточно, "
            "верни score=0."
        )

        try:
            response = llm.chat(
                [{"role": "user", "content": prompt}],
                json_schema=MAP_SCHEMA,
            )

            item = json.loads(response.content)
            score = item["score"]
            answer = item["answer"]

            if (
                isinstance(score, int)
                and not isinstance(score, bool)
                and 0 < score <= 10
                and isinstance(answer, str)
                and answer.strip()
            ):
                partial_answers.append(
                    {
                        "answer": answer,
                        "score": score,
                    }
                )

                sources.update(
                    _sources(kb.graph, set(community.nodes))
                )

        except (ValueError, KeyError, TypeError):
            continue

    if not partial_answers:
        return Answer(
            text="В базе знаний не найдено достаточно информации.",
            refused=True,
        )

    graph_statistics = _global_graph_statistics(kb.graph)

    response = llm.chat(
        [
            {
                "role": "system",
                "content": (
                    "Объедини частичные ответы в один точный ответ. Не добавляй "
                    "неподтверждённые факты. Если частичные ответы противоречат "
                    "друг другу, укажи неопределённость. Для вопросов о частоте "
                    "отказов используй числовые счётчики из программной статистики. "
                    "При равных максимальных значениях перечисли все варианты, "
                    "не выбирай первый произвольно. Поле score означает релевантность "
                    "сводки и никогда не является частотой отказов."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Вопрос: {question}\n\n"
                    "Частичные ответы по сводкам сообществ:\n"
                    f"{json.dumps(partial_answers, ensure_ascii=False)}\n\n"
                    "Дополнительная статистика, вычисленная Python непосредственно "
                    "по весам рёбер графа (weight FAILED = число разных чанков/отчётов):\n"
                    f"{json.dumps(graph_statistics, ensure_ascii=False)}\n\n"
                    "Используй статистику только для соответствующих вопросов. "
                    "Для вопроса о наиболее частых отказах сравни count. Для вопроса "
                    "о моделях сравни число отказов по model. Для вопроса о снятых "
                    "с производства деталях используй discontinued_parts_by_line."
                ),
            },
        ]
    )

    return Answer(
        text=response.content,
        sources=sorted(sources),
    )