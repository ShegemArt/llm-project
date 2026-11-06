
"""Построение направленного графа знаний (FR-2)."""

import networkx as nx

from assistant.infra.schemas import ExtractionResult


def build_graph(results: list[ExtractionResult]) -> nx.DiGraph:
    """Объединяет результаты извлечения в направленный граф."""
    graph = nx.DiGraph()
    edge_chunks: dict[tuple[str, str, str], set[str]] = {}

    for result in results:
        for entity in result.entities:
            node_id = entity.node_id

            if not graph.has_node(node_id):
                graph.add_node(
                    node_id,
                    type=entity.type,
                    name=entity.name,
                    sources=[],
                )

            if result.chunk_id not in graph.nodes[node_id]["sources"]:
                graph.nodes[node_id]["sources"].append(result.chunk_id)

        for relation in result.relations:
            if not relation.matches_schema():
                continue

            source = relation.subject.node_id
            target = relation.object.node_id

            # Не создаём рёбра, ссылающиеся на отсутствующие сущности.
            if source not in graph or target not in graph:
                continue

            key = (source, target, relation.type)
            edge_chunks.setdefault(key, set()).add(result.chunk_id)

    for (source, target, relation_type), chunks in edge_chunks.items():
        graph.add_edge(
            source,
            target,
            type=relation_type,
            weight=len(chunks),
            sources=sorted(chunks),
        )

    # Разрешаем конфликты в цепочках замены.
    conflicts = 0
    to_remove = set()

    for source, target, data in list(graph.edges(data=True)):
        if data.get("type") != "REPLACED_BY":
            continue

        if source == target:
            graph.remove_edge(source, target)
            continue

        reverse = graph.get_edge_data(target, source)

        if not reverse or reverse.get("type") != "REPLACED_BY":
            continue

        pair = frozenset((source, target))

        if pair in to_remove:
            continue

        forward_weight = data["weight"]
        reverse_weight = reverse["weight"]
        conflicts += 1

        if forward_weight > reverse_weight:
            graph.remove_edge(target, source)
        elif reverse_weight > forward_weight:
            graph.remove_edge(source, target)
        else:
            graph.remove_edge(source, target)
            graph.remove_edge(target, source)

        to_remove.add(pair)

    graph.graph["conflicts"] = conflicts

    return graph
