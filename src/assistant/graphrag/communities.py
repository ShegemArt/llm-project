import networkx as nx

from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Community


def detect_communities(graph: nx.DiGraph) -> list[Community]:
    """Выделяет сообщества алгоритмом Лувена."""
    if graph.number_of_nodes() == 0:
        return []

    undirected = nx.Graph()
    undirected.add_nodes_from(graph.nodes)

    # В исходном графе связи направлены, но для группировки
    # важно, насколько тесно связаны узлы.
    for u, v, data in graph.edges(data=True):
        weight = data.get("weight", 1)
        if undirected.has_edge(u, v):
            undirected[u][v]["weight"] += weight
        else:
            undirected.add_edge(u, v, weight=weight)

    groups = nx.community.louvain_communities(
        undirected,
        weight="weight",
        seed=7,
    )

    # Стабилизируем нумерацию сообществ.
    groups = sorted(groups, key=lambda group: min(group))

    return [
        Community(
            community_id=index,
            nodes=sorted(group),
            summary="",
        )
        for index, group in enumerate(groups)
    ]


def summarize_community(
    community: Community,
    graph: nx.DiGraph,
    llm: LLMClient,
) -> str:
    """Генерирует сводку сообщества объёмом до 150 слов."""
    node_ids = set(community.nodes)

    entities = [
        {
            "id": node,
            "type": graph.nodes[node].get("type"),
            "name": graph.nodes[node].get("name"),
        }
        for node in sorted(node_ids)
        if node in graph
    ]

    relations = []
    for source, target, data in graph.edges(data=True):
        if source not in node_ids or target not in node_ids:
            continue

        relations.append({
            "source": source,
            "target": target,
            "type": data.get("type"),
            "weight": data.get("weight", 1),
        })

    # Статистику считает Python, а не языковая модель.
    relation_counts = {}
    for relation in relations:
        kind = relation["type"]
        relation_counts[kind] = (
            relation_counts.get(kind, 0) + relation["weight"]
        )

    facts = {
        "entities": entities,
        "relations": relations,
        "relation_counts": relation_counts,
    }

    response = llm.chat([
        {
            "role": "system",
            "content": (
                "Составь точную сводку графа знаний на русском языке. "
                "Не добавляй отсутствующие факты. Не более 150 слов. "
                "Используй только переданные сущности, связи и числа."
            ),
        },
        {
            "role": "user",
            "content": str(facts),
        },
    ])

    return " ".join(response.content.split()[:150])