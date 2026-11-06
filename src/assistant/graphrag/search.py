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


def _format_edge(graph: nx.DiGraph, u: str, v: str, data: dict) -> str:
    left = graph.nodes[u]["name"]
    right = graph.nodes[v]["name"]
    relation = data.get("type", "RELATED_TO")
    return f"{left} -{relation}-> {right}"


def _replacement_paths(
    graph: nx.DiGraph,
    start_nodes: set[str],
) -> dict[str, list[str]]:
    """Возвращает однозначные цепочки REPLACED_BY для найденных деталей."""
    paths: dict[str, list[str]] = {}

    for start in sorted(start_nodes):
        if start not in graph or graph.nodes[start].get("type") != "Part":
            continue

        path = [start]
        visited = {start}
        current = start

        while True:
            candidates = sorted(
                target
                for _, target, data in graph.out_edges(current, data=True)
                if data.get("type") == "REPLACED_BY"
            )

            # Несколько вариантов замены нельзя разрешить автоматически.
            if len(candidates) != 1:
                break

            next_node = candidates[0]
            if next_node in visited:
                # Циклическая цепочка не считается корректной.
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
) -> tuple[set[str], set[tuple[str, str]]]:
    """Добавляет для найденных деталей тип, поставщика и все последующие замены."""
    expanded_nodes = set(seed_nodes)
    selected_edges: set[tuple[str, str]] = set()
    pending_parts = [
        node
        for node in expanded_nodes
        if graph.nodes[node].get("type") == "Part"
    ]
    visited_parts: set[str] = set()

    while pending_parts:
        part_node = pending_parts.pop()
        if part_node in visited_parts:
            continue
        visited_parts.add(part_node)

        for _, target, data in graph.out_edges(part_node, data=True):
            relation = data.get("type")
            if relation not in {"IS_A", "SUPPLIED_BY", "REPLACED_BY"}:
                continue

            selected_edges.add((part_node, target))
            expanded_nodes.add(target)

            if (
                relation == "REPLACED_BY"
                and target in graph
                and graph.nodes[target].get("type") == "Part"
                and target not in visited_parts
            ):
                pending_parts.append(target)

    return expanded_nodes, selected_edges


def _global_graph_statistics(graph: nx.DiGraph) -> dict:
    """Считает глобальные частоты отказов и снятые с производства детали.

    Вес FAILED-ребра — количество разных чанков (отчётов), в которых
    зафиксирован отказ модели. Статистика считается кодом, а не LLM.
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
        component_name = graph.nodes[component_node].get("name", component_node)

        for line_name in model_to_lines.get(model_node, set()):
            component_key = (line_name, component_name)
            model_key = (line_name, model_name)
            failures_by_line_component[component_key] = (
                failures_by_line_component.get(component_key, 0) + weight
            )
            failures_by_line_model[model_key] = (
                failures_by_line_model.get(model_key, 0) + weight
            )

    # Учитываем только детали, которые действительно установлены в модели
    # соответствующей линейки и имеют исходящее ребро REPLACED_BY.
    discontinued_by_line: dict[str, set[str]] = {}
    for model_node, part_nodes in model_to_parts.items():
        for line_name in model_to_lines.get(model_node, set()):
            for part_node in part_nodes:
                replacements = [
                    target
                    for _, target, data in graph.out_edges(part_node, data=True)
                    if data.get("type") == "REPLACED_BY"
                ]
                if replacements:
                    discontinued_by_line.setdefault(line_name, set()).add(
                        graph.nodes[part_node].get("name", part_node)
                    )

    return {
        "failures_by_line_and_component": [
            {"line": line, "component": component, "count": count}
            for (line, component), count in sorted(failures_by_line_component.items())
        ],
        "failures_by_line_and_model": [
            {"line": line, "model": model, "count": count}
            for (line, model), count in sorted(failures_by_line_model.items())
        ],
        "discontinued_parts_by_line": [
            {"line": line, "parts": sorted(parts)}
            for line, parts in sorted(discontinued_by_line.items())
        ],
    }


def _edge_sources(
    graph: nx.DiGraph,
    edges: set[tuple[str, str]],
) -> list[str]:
    """Возвращает документы, подтверждающие выбранные рёбра."""
    sources: set[str] = set()
    for source, target in edges:
        data = graph.get_edge_data(source, target)
        if not data:
            continue
        sources.update(
            chunk_id.split("#", 1)[0]
            for chunk_id in data.get("sources", [])
        )
    return sorted(sources)


def _direct_local_answer(
    question: str,
    kb: KnowledgeBase,
    anchor_nodes: set[str],
) -> Answer | None:
    """Отвечает напрямую по однозначным связям графа без генерации LLM."""
    graph = kb.graph
    lowered = question.casefold()
    asks_procedure = bool(
        re.search(r"регламент\w*|процедур\w*", lowered)
    )
    asks_indication = bool(
        re.search(
            r"что\s+(?:означает|указывает)|на\s+отказ\s+какого\s+узла|"
            r"какой\s+узел\s+(?:отказал|неисправен)",
            lowered,
        )
    )

    asks_product_line = bool(
        re.search(
            r"в\s+какую\s+линейк\w*|к\s+какой\s+линейк\w*",
            lowered,
        )
    )
    asks_installed_models = bool(
        re.search(
            r"(?:в\s+каких\s+моделях|моделях?\s*,?\s+где).*"
            r"(?:установлен\w*|использу\w*|применя\w*)",
            lowered,
        )
    )

    if asks_installed_models:
        answers = []
        selected_edges: set[tuple[str, str]] = set()
        facts = []
        part_nodes = {
            node
            for node in anchor_nodes
            if node in graph and graph.nodes[node].get("type") == "Part"
        }
        for part_node in sorted(part_nodes):
            for model_node, _, data in graph.in_edges(part_node, data=True):
                if (
                    data.get("type") != "HAS_PART"
                    or graph.nodes[model_node].get("type") != "Model"
                ):
                    continue
                selected_edges.add((model_node, part_node))
                facts.append(
                    _format_edge(graph, model_node, part_node, data)
                )
                answers.append(
                    f"Деталь {graph.nodes[part_node].get('name', part_node)} "
                    f"установлена в модели "
                    f"{graph.nodes[model_node].get('name', model_node)}."
                )

        if answers:
            return Answer(
                text="\n".join(sorted(set(answers))),
                sources=_edge_sources(graph, selected_edges),
                facts=list(dict.fromkeys(facts)),
            )

    if asks_product_line:
        answers = []
        selected_edges: set[tuple[str, str]] = set()
        facts = []
        model_nodes = {
            node
            for node in anchor_nodes
            if node in graph and graph.nodes[node].get("type") == "Model"
        }
        for model_node in sorted(model_nodes):
            for _, target, data in graph.out_edges(model_node, data=True):
                if data.get("type") != "BELONGS_TO":
                    continue
                selected_edges.add((model_node, target))
                facts.append(
                    _format_edge(graph, model_node, target, data)
                )
                answers.append(
                    f"Модель {graph.nodes[model_node].get('name', model_node)} "
                    f"входит в линейку "
                    f"«{graph.nodes[target].get('name', target)}»."
                )
        if answers:
            return Answer(
                text="\n".join(answers),
                sources=_edge_sources(graph, selected_edges),
                facts=list(dict.fromkeys(facts)),
            )

    if not (asks_procedure or asks_indication):
        return None

    error_nodes = {
        node
        for node in anchor_nodes
        if node in graph and graph.nodes[node].get("type") == "ErrorCode"
    }

    # Запрос может быть сформулирован через название узла, без кода ошибки.
    component_nodes = {
        node
        for node in anchor_nodes
        if node in graph and graph.nodes[node].get("type") == "Component"
    }
    if component_nodes:
        for component_node in component_nodes:
            for error_node, _, data in graph.in_edges(
                component_node,
                data=True,
            ):
                if data.get("type") == "INDICATES":
                    error_nodes.add(error_node)

    selected_edges: set[tuple[str, str]] = set()
    facts: list[str] = []
    answers: list[str] = []

    for error_node in sorted(error_nodes):
        if asks_procedure:
            candidates = [
                (error_node, target)
                for _, target, data in graph.out_edges(
                    error_node,
                    data=True,
                )
                if data.get("type") == "FIXED_BY"
            ]
            for source, target in candidates:
                selected_edges.add((source, target))
                facts.append(_format_edge(graph, source, target, graph.edges[source, target]))
                answers.append(
                    f"Ошибка {graph.nodes[source].get('name', source)} "
                    f"устраняется по регламенту "
                    f"{graph.nodes[target].get('name', target)}."
                )
        else:
            candidates = [
                (error_node, target)
                for _, target, data in graph.out_edges(
                    error_node,
                    data=True,
                )
                if data.get("type") == "INDICATES"
            ]
            for source, target in candidates:
                selected_edges.add((source, target))
                facts.append(_format_edge(graph, source, target, graph.edges[source, target]))
                answers.append(
                    f"Ошибка {graph.nodes[source].get('name', source)} "
                    f"указывает на отказ узла "
                    f"{graph.nodes[target].get('name', target)}."
                )

    if not answers:
        return None

    return Answer(
        text="\n".join(dict.fromkeys(answers)),
        sources=_edge_sources(graph, selected_edges),
        facts=list(dict.fromkeys(facts)),
    )


def _statistics_answer(
    question: str,
    kb: KnowledgeBase,
) -> Answer | None:
    """Программно отвечает на вопросы о максимальной частоте отказов."""
    lowered = question.casefold()
    asks_model_max = bool(
        re.search(
            r"больше\s+всего\s+отказ|наибольш\w*\s+количеств\w*\s+отказ|"
            r"максимальн\w*\s+количеств\w*\s+отказ",
            lowered,
        )
    )

    if not asks_model_max:
        return None

    line_names = sorted(
        {
            name
            for kind, name in find_names(question, kb.vocabulary)
            if kind == "ProductLine"
        }
    )
    if not line_names:
        return None

    graph = kb.graph
    statistics = _global_graph_statistics(graph)
    rows = statistics["failures_by_line_and_model"]
    label_key = "model"

    selected_rows = [
        row for row in rows
        if row["line"] in line_names
    ]
    if not selected_rows:
        return None

    grouped: dict[str, list[dict]] = {}
    for row in selected_rows:
        grouped.setdefault(row["line"], []).append(row)

    selected_edges: set[tuple[str, str]] = set()
    facts: list[str] = []
    answers: list[str] = []

    for line_name in line_names:
        line_rows = grouped.get(line_name, [])
        if not line_rows:
            continue

        maximum = max(row["count"] for row in line_rows)
        winners = sorted(
            (row for row in line_rows if row["count"] == maximum),
            key=lambda row: row[label_key],
        )
        winner_names = [row[label_key] for row in winners]

        if len(winner_names) == 1:
            answer_subject = winner_names[0]
            answers.append(
                f"В линейке «{line_name}» наибольшее количество отказов "
                f"у модели {answer_subject}: {maximum}."
            )
        else:
            joined = ", ".join(winner_names)
            answers.append(
                f"В линейке «{line_name}» максимальное количество отказов "
                f"({maximum}) имеют несколько вариантов: {joined}."
            )

        winning_set = set(winner_names)
        for source, target, data in graph.edges(data=True):
            if data.get("type") != "FAILED":
                continue
            source_name = graph.nodes[source].get("name", source)
            if source_name not in winning_set:
                continue

            # Учитываем только модели выбранной линейки.
            belongs = [
                (source, line_target)
                for _, line_target, edge_data in graph.out_edges(
                    source,
                    data=True,
                )
                if edge_data.get("type") == "BELONGS_TO"
                and graph.nodes[line_target].get("name") == line_name
            ]
            if not belongs:
                continue

            selected_edges.add((source, target))
            selected_edges.update(belongs)
            facts.append(_format_edge(graph, source, target, data))
            for left, right in belongs:
                facts.append(
                    _format_edge(graph, left, right, graph.edges[left, right])
                )

    if not answers:
        return None

    return Answer(
        text="\n".join(answers),
        sources=_edge_sources(graph, selected_edges),
        facts=list(dict.fromkeys(facts)),
    )


def _discontinued_parts_answer(
    question: str,
    kb: KnowledgeBase,
) -> Answer | None:
    """Формирует ответ о снятых деталях линейки по отношениям графа."""
    if not re.search(r"снят\w*\s+с\s+производств", question, re.IGNORECASE):
        return None

    line_names = sorted(
        {
            name
            for kind, name in find_names(question, kb.vocabulary)
            if kind == "ProductLine"
        }
    )
    if not line_names:
        return None

    statistics = _global_graph_statistics(kb.graph)
    discontinued_by_line = {
        item["line"]: item["parts"]
        for item in statistics["discontinued_parts_by_line"]
    }

    graph = kb.graph
    selected_nodes: set[str] = set()
    facts: list[str] = []
    answers: list[str] = []

    for line_name in line_names:
        line_node = f"ProductLine:{line_name}"
        if line_node in graph:
            selected_nodes.add(line_node)

        parts = discontinued_by_line.get(line_name, [])
        if not parts:
            answers.append(
                f"Для линейки «{line_name}» в графе не найдены детали, "
                "связанные с заменой."
            )
            continue

        part_descriptions = []

        for part_name in parts:
            part_node = f"Part:{part_name}"
            if part_node not in graph:
                continue

            selected_nodes.add(part_node)
            replacements = sorted(
                target
                for _, target, data in graph.out_edges(part_node, data=True)
                if data.get("type") == "REPLACED_BY"
            )

            if line_node in graph:
                for model_node, _, belongs_data in graph.in_edges(
                    line_node,
                    data=True,
                ):
                    if belongs_data.get("type") != "BELONGS_TO":
                        continue
                    if not graph.has_edge(model_node, part_node):
                        continue

                    has_part_data = graph.get_edge_data(model_node, part_node)
                    if has_part_data and has_part_data.get("type") == "HAS_PART":
                        selected_nodes.add(model_node)
                        facts.append(
                            _format_edge(graph, model_node, part_node, has_part_data)
                        )
                        facts.append(
                            _format_edge(graph, model_node, line_node, belongs_data)
                        )

            for replacement in replacements:
                selected_nodes.add(replacement)
                replacement_data = graph.get_edge_data(part_node, replacement)
                if replacement_data:
                    facts.append(
                        _format_edge(graph, part_node, replacement, replacement_data)
                    )

            if len(replacements) == 1:
                part_descriptions.append(
                    f"{part_name} (заменена на "
                    f"{graph.nodes[replacements[0]].get('name', replacements[0])})"
                )
            elif replacements:
                replacement_names = [
                    graph.nodes[node].get("name", node)
                    for node in replacements
                ]
                part_descriptions.append(
                    f"{part_name} (указаны варианты замены: "
                    f"{', '.join(replacement_names)}; цепочка неоднозначна)"
                )
            else:
                part_descriptions.append(part_name)

        if part_descriptions:
            answers.append(
                f"Для линейки «{line_name}» снятые с производства детали: "
                + "; ".join(part_descriptions)
                + "."
            )
        else:
            answers.append(
                f"Для линейки «{line_name}» не удалось сопоставить "
                "снятые детали с узлами графа."
            )

    return Answer(
        text="\n".join(answers),
        sources=_sources(graph, selected_nodes),
        facts=list(dict.fromkeys(facts)),
    )


def local_search(
    question: str,
    kb: KnowledgeBase,
    llm: LLMClient,
) -> Answer:
    """Ищет опорные сущности и расширяет найденный контекст деталями и заменами."""
    graph = kb.graph
    anchor_nodes: set[str] = set()
    codes = find_codes(question)

    for code in codes:
        node_id = f"{code_type(code)}:{code}"
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

    direct_answer = _direct_local_answer(question, kb, anchor_nodes)
    if direct_answer is not None:
        return direct_answer

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
                        for left, right in zip(
                            path[:-1],
                            path[1:],
                            strict=True,
                        ):
                            # Путь вычисляется по неориентированному графу,
                            # но в факты добавляются только существующие
                            # направленные отношения.
                            if graph.has_edge(left, right):
                                selected_edges.add((left, right))
                            if graph.has_edge(right, left):
                                selected_edges.add((right, left))
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

    # Для найденных деталей добавляем тип, поставщика и все связи замен.
    path_nodes, part_edges = _expand_part_context(graph, path_nodes)
    selected_edges.update(part_edges)

    # Сохраняем только однозначные цепочки: при нескольких заменах
    # не выбираем конечную деталь произвольно.
    replacement_paths = _replacement_paths(
        graph,
        {
            node
            for node in path_nodes
            if graph.nodes[node].get("type") == "Part"
        },
    )

    facts = []
    for source, target in sorted(selected_edges):
        data = graph.get_edge_data(source, target)
        if data:
            facts.append(_format_edge(graph, source, target, data))

    for path in replacement_paths.values():
        names = [
            graph.nodes[node].get("name", node)
            for node in path
        ]
        facts.append("Цепочка замен: " + " -> ".join(names))
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
                    "Отвечай только по переданным фактам. Не добавляй отсутствующие "
                    "коды, цены, остатки или связи. Если вопрос спрашивает, какая "
                    "деталь используется сейчас, а в фактах есть однозначная цепочка "
                    "REPLACED_BY, назови последнюю деталь цепочки. Не переноси "
                    "поставщика со старой детали на новую, если для новой детали "
                    "нет отдельной связи SUPPLIED_BY. Если цепочка замен неоднозначна "
                    "или фактов недостаточно, прямо сообщи об этом."
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


def global_search(
    question: str,
    kb: KnowledgeBase,
    llm: LLMClient,
) -> Answer:
    """Объединяет сводки сообществ и точные глобальные статистики графа."""
    # Вопросы с конкретными кодами должны опираться на факты графа,
    # а не на пересказ сводок сообществ.
    if find_codes(question):
        return local_search(question, kb, llm)

    # Список снятых с производства деталей считается программно,
    # а не генерируется LLM из статистики отказов.
    discontinued_answer = _discontinued_parts_answer(question, kb)
    if discontinued_answer is not None:
        return discontinued_answer

    statistics_answer = _statistics_answer(question, kb)
    if statistics_answer is not None:
        return statistics_answer

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
                partial_answers.append({"answer": answer, "score": score})
                sources.update(_sources(kb.graph, set(community.nodes)))
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