"""BIO -> entity spans (identical to the Phase 2 error-analysis implementation)."""


def bio_to_entities(tokens, labels):
    """
    Convert BIO labels to entity spans.

    Returns:
    {
        "attribute": str,
        "start": int,
        "end": int,       # exclusive
        "text": str
    }
    """

    entities = []

    current_attr = None
    start = None

    def close_entity(end_idx):
        nonlocal current_attr, start

        if current_attr is not None:
            entities.append({
                "attribute": current_attr,
                "start": start,
                "end": end_idx,
                "text": " ".join(tokens[start:end_idx]),
            })

        current_attr = None
        start = None

    for i, label in enumerate(labels):

        if label == "O":

            if current_attr is not None:
                close_entity(i)

            continue

        prefix, attribute = label.split("-", 1)

        if prefix == "B":

            if current_attr is not None:
                close_entity(i)

            current_attr = attribute
            start = i

        elif prefix == "I":

            if current_attr != attribute:

                if current_attr is not None:
                    close_entity(i)

                # Defensive fallback
                current_attr = attribute
                start = i

    if current_attr is not None:
        close_entity(len(tokens))

    return entities
