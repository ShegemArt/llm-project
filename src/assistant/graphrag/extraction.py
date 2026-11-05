"""Извлечение сущностей и связей из одного чанка (FR-1)."""

import json

from assistant.infra.config import vocabulary
from assistant.infra.grounding import (
    canonical_name,
    find_codes,
    normalize_code,
)
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import (
    EXTRACTION_SCHEMA,
    Chunk,
    Entity,
    ExtractionResult,
    Relation,
)

_CODE_ENTITY_TYPES = {"Model", "Part", "ErrorCode", "Procedure"}


def _validated_entity(
    entity_type: str,
    raw_name: str,
    chunk_text: str,
    codes_in_text: set[str],
    vocab: dict[str, list[str]],
) -> Entity | None:
    """Возвращает сущность, подтверждённую текстом чанка и схемой словаря."""
    if not isinstance(entity_type, str) or not isinstance(raw_name, str):
        return None

    if entity_type in _CODE_ENTITY_TYPES:
        canonical = normalize_code(raw_name)

        if canonical is None or canonical not in codes_in_text:
            return None

        canonical = canonical_name(entity_type, canonical, vocab)

        if canonical is None:
            return None

        return Entity(entity_type, canonical)

    canonical = canonical_name(entity_type, raw_name, vocab)

    if canonical is None or canonical.casefold() not in chunk_text.casefold():
        return None

    return Entity(entity_type, canonical)


def _extraction_prompt(
    chunk_text: str,
    vocab: dict[str, list[str]],
) -> str:
    """Формирует исчерпывающую инструкцию с явной семантикой каждого ребра."""
    return f"""
Ты выполняешь извлечение фактов из технического документа. Не пересказывай документ.
Нужно перечислить ВСЕ явно подтверждённые сущности и ВСЕ отношения из текущего чанка.
Обрабатывай текст последовательно, строка за строкой. Не ограничивайся одним примером
или одним отношением на документ. Если в строке перечислено несколько деталей или ошибок,
извлеки каждую из них.

Допустимые типы сущностей:
ProductLine, Model, Component, Part, ErrorCode, Procedure, Supplier.

Допустимые отношения и их точное направление:
- Model -> ProductLine: BELONGS_TO — модель входит в линейку.
- Model -> Part: HAS_PART — в этой модели установлена указанная деталь.
- Part -> Component: IS_A — деталь относится к указанному виду узла.
- Part -> Supplier: SUPPLIED_BY — указан поставщик этой детали.
- ErrorCode -> Component: INDICATES — код ошибки указывает на отказ указанного узла.
- ErrorCode -> Procedure: FIXED_BY — ошибка устраняется по указанному регламенту.
- Part -> Part: REPLACED_BY — старая деталь заменена новой.
- Model -> Component: FAILED — в отчёте о выезде у модели зафиксирован отказ узла.

Правила для типовых фрагментов:
1. В руководстве найди модель и линейку. Для КАЖДОЙ строки состава узлов вида
   «название узла — деталь P-..., поставщик ...» извлеки три отношения:
   Model -> Part (HAS_PART), Part -> Component (IS_A), Part -> Supplier (SUPPLIED_BY).
   Код модели берётся из заголовка/описания этого же руководства.
2. Для КАЖДОЙ строки таблицы ошибок вида «E-... — ... отказ узла ...» извлеки
   ErrorCode -> Component (INDICATES).
3. В бюллетене, если сказано, что старая деталь снята с производства и вместо неё
   устанавливается новая, извлеки старую Part -> новую Part (REPLACED_BY).
   Если рядом с кодом детали в скобках указан вид узла, извлеки Part -> Component (IS_A).
   Если указан поставщик детали, извлеки Part -> Supplier (SUPPLIED_BY) именно для
   той детали, к которой относится поставщик.
4. В регламенте фраза «применяется при ошибке E-...» означает
   ErrorCode -> Procedure (FIXED_BY). Фраза «ошибка E-... указывает на отказ узла ...»
   означает ErrorCode -> Component (INDICATES).
5. В отчёте о выезде извлеки Model -> Component (FAILED), когда в документе названа
   модель оборудования и явно сказано «Диагностика показала отказ узла: ...».

Примеры направления (это только примеры формата; не добавляй их, если фактов нет
в текущем чанке):
- AR-210 входит в линейку Арктика:
  Model:AR-210 -BELONGS_TO-> ProductLine:Арктика
- В AR-210 установлен компрессор P-1010 от поставщика КомпрессорМаш:
  Model:AR-210 -HAS_PART-> Part:P-1010
  Part:P-1010 -IS_A-> Component:компрессор
  Part:P-1010 -SUPPLIED_BY-> Supplier:КомпрессорМаш
- E-17 указывает на отказ пускового реле:
  ErrorCode:E-17 -INDICATES-> Component:пусковое реле
- Ошибка E-17 устраняется по SP-104:
  ErrorCode:E-17 -FIXED_BY-> Procedure:SP-104
- P-1042 заменена на P-1042B:
  Part:P-1042 -REPLACED_BY-> Part:P-1042B
- В AR-220 диагностирован отказ датчика температуры:
  Model:AR-220 -FAILED-> Component:датчик температуры

Требования к результату:
- Верни JSON строго по переданной JSON-схеме: поля entities и relations.
- В entities включи все сущности, участвующие в relations.
- Направление отношений не меняй.
- Каждая связь должна подтверждаться текущим чанком; не достраивай её по знаниям
  из других документов и не делай предположений.
- Не создавай связь только потому, что две сущности встретились в одном документе.
- Имена ProductLine, Component и Supplier бери строго из словаря ниже.
- Коды нормализуй: латинские заглавные буквы, дефис, без пробелов.
- Не используй номера документов SB-..., R-... как сущности.
- Не добавляй цены, остатки или другие числовые показатели как сущности.
- Не дублируй одинаковые сущности и отношения.

Словарь допустимых названий:
{json.dumps(vocab, ensure_ascii=False)}

Текст чанка:
{chunk_text}
""".strip()


def extract_chunk(chunk: Chunk, llm: LLMClient) -> ExtractionResult:
    """Извлекает проверенные сущности и связи из одного чанка."""
    vocab = vocabulary()
    empty = ExtractionResult(chunk.chunk_id, chunk.doc_id)
    codes_in_text = set(find_codes(chunk.text))
    prompt = _extraction_prompt(chunk.text, vocab)

    for attempt in range(2):
        try:
            if attempt == 0:
                system_prompt = (
                    "Ты — точный экстрактор фактов. Полностью обработай весь чанк. "
                    "Не сокращай списки повторяющихся строк. Верни только JSON по схеме. "
                    "КРИТИЧЕСКИ ВАЖНО: subject всегда является началом отношения, "
                    "object всегда является его концом. Не меняй их местами. "
                    "Для BELONGS_TO: Model -> ProductLine. "
                    "Для HAS_PART: Model -> Part. "
                    "Для IS_A: Part -> Component. "
                    "Для INDICATES: ErrorCode -> Component. "
                    "Для FIXED_BY: ErrorCode -> Procedure. "
                    "Для REPLACED_BY: старая Part -> новая Part. "
                    "Для SUPPLIED_BY: Part -> Supplier. "
                    "Для FAILED: Model -> Component."
                )
                max_tokens = 2048
            else:
                # Новый системный запрос и лимит токенов формируют новый ключ кеша.
                system_prompt = (
                    "Предыдущий ответ оказался невалидным или неполным. "
                    "Заново извлеки ВСЕ сущности и отношения из полного текста. "
                    "Проверь закрытие JSON-массивов и объектов. Не сокращай отношения. "
                    "subject — начало отношения, object — конец отношения. "
                    "Строго соблюдай направления: "
                    "BELONGS_TO: Model -> ProductLine; "
                    "HAS_PART: Model -> Part; "
                    "IS_A: Part -> Component; "
                    "INDICATES: ErrorCode -> Component; "
                    "FIXED_BY: ErrorCode -> Procedure; "
                    "REPLACED_BY: старая Part -> новая Part; "
                    "SUPPLIED_BY: Part -> Supplier; "
                    "FAILED: Model -> Component. "
                    "Верни только JSON по схеме."
                )
                max_tokens = 3072

            response = llm.chat(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                json_schema=EXTRACTION_SCHEMA,
                max_tokens=max_tokens,
            )

            data = json.loads(response.content)

            if not isinstance(data, dict):
                raise ValueError("Ответ модели должен быть JSON-объектом.")

            raw_entities = data.get("entities")
            raw_relations = data.get("relations")

            if (
                not isinstance(raw_entities, list)
                or not isinstance(raw_relations, list)
            ):
                raise ValueError(
                    "Поля entities и relations должны быть списками."
                )

            entities: dict[str, Entity] = {}
            dropped = 0

            for raw in raw_entities:
                if not isinstance(raw, dict):
                    dropped += 1
                    continue

                entity = _validated_entity(
                    raw.get("type"),
                    raw.get("name"),
                    chunk.text,
                    codes_in_text,
                    vocab,
                )

                if entity is None:
                    dropped += 1
                    continue

                entities[entity.node_id] = entity

            relations: list[Relation] = []
            seen_relations: set[tuple[str, str, str]] = set()

            for raw in raw_relations:
                if not isinstance(raw, dict):
                    dropped += 1
                    continue

                subject = _validated_entity(
                    raw.get("subject_type"),
                    raw.get("subject"),
                    chunk.text,
                    codes_in_text,
                    vocab,
                )

                obj = _validated_entity(
                    raw.get("object_type"),
                    raw.get("object"),
                    chunk.text,
                    codes_in_text,
                    vocab,
                )

                relation_type = raw.get("type")

                if subject is None or obj is None:
                    dropped += 1
                    continue

                relation = Relation(subject, relation_type, obj)

                # Qwen3 может менять местами начало и конец отношения.
                # Сначала проверяем исходное направление.
                if not relation.matches_schema():
                    reversed_relation = Relation(obj, relation_type, subject)

                    # Исправляем направление только если обратное соответствует схеме.
                    if reversed_relation.matches_schema():
                        subject, obj = obj, subject
                        relation = reversed_relation
                    else:
                        dropped += 1
                        continue

                key = (subject.node_id, relation_type, obj.node_id)

                if key in seen_relations:
                    continue

                seen_relations.add(key)
                relations.append(relation)

                entities[subject.node_id] = subject
                entities[obj.node_id] = obj

            return ExtractionResult(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                entities=list(entities.values()),
                relations=relations,
                dropped=dropped,
            )

        except Exception:
            if attempt == 1:
                return empty

    return empty