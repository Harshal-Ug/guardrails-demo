import re
from gliner import GLiNER


class PIIAnonymizer:

    def __init__(self):
        print("Loading GLiNER model...")

        self.model = GLiNER.from_pretrained(
            "urchade/gliner_small-v2.1"
        )

        # Entity types GLiNER should detect.
        # These are labels, NOT hardcoded names/patterns.
        self.labels = [
            "person",
            "location",
            "email address",
            "phone number",
        ]

    def mask(self, text: str):
        """
        Detect PII in text and replace it with placeholders.

        Returns:
            masked_text: sanitized text
            mapping: placeholder -> original value
        """

        mapping = {}

        # --------------------------------------------------
        # 1. GLiNER NER detection
        # --------------------------------------------------

        predictions = self.model.predict_entities(
            text,
            self.labels,
            threshold=0.5,
        )

        # --------------------------------------------------
        # 2. Add structured PII detected by regex
        # --------------------------------------------------

        regex_entities = []

        # Email
        for match in re.finditer(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            text,
        ):
            regex_entities.append({
                "start": match.start(),
                "end": match.end(),
                "text": match.group(),
                "label": "email address",
            })

        # Phone number
        #
        # Supports common international / Indian formats.
        # GLiNER remains the main semantic detector.
        for match in re.finditer(
            r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)",
            text,
        ):
            value = match.group().strip()

            # Avoid treating very short numeric sequences as phones.
            digits = re.sub(r"\D", "", value)

            if 8 <= len(digits) <= 15:
                regex_entities.append({
                    "start": match.start(),
                    "end": match.end(),
                    "text": value,
                    "label": "phone number",
                })

        # --------------------------------------------------
        # 3. Merge GLiNER + regex detections
        # --------------------------------------------------

        entities = []

        for entity in predictions:
            entities.append({
                "start": entity["start"],
                "end": entity["end"],
                "text": entity["text"],
                "label": entity["label"],
            })

        entities.extend(regex_entities)

        # --------------------------------------------------
        # 4. Remove overlapping detections
        # --------------------------------------------------

        entities.sort(
            key=lambda entity: (
                entity["start"],
                -(entity["end"] - entity["start"]),
            )
        )

        filtered_entities = []

        for entity in entities:

            overlaps = False

            for existing in filtered_entities:

                if (
                    entity["start"] < existing["end"]
                    and entity["end"] > existing["start"]
                ):
                    overlaps = True
                    break

            if not overlaps:
                filtered_entities.append(entity)

        # --------------------------------------------------
        # 5. Create placeholders
        # --------------------------------------------------

        counters = {
            "PERSON": 0,
            "LOCATION": 0,
            "EMAIL": 0,
            "PHONE": 0,
        }

        replacements = []

        for entity in filtered_entities:

            label = entity["label"].lower()
            original_value = entity["text"]

            if label == "person":
                entity_type = "PERSON"

            elif label == "location":
                entity_type = "LOCATION"

            elif label == "email address":
                entity_type = "EMAIL"

            elif label == "phone number":
                entity_type = "PHONE"

            else:
                continue

            index = counters[entity_type]

            token = f"[{entity_type}_{index}]"

            counters[entity_type] += 1

            mapping[token] = original_value

            replacements.append({
                "start": entity["start"],
                "end": entity["end"],
                "token": token,
            })

        # --------------------------------------------------
        # 6. Replace from right to left
        # --------------------------------------------------

        masked_text = text

        for replacement in reversed(replacements):

            start = replacement["start"]
            end = replacement["end"]
            token = replacement["token"]

            masked_text = (
                masked_text[:start]
                + token
                + masked_text[end:]
            )

        return masked_text, mapping

    def unmask(self, text: str, mapping: dict[str, str]):
        """
        Restore original PII values in the LLM response.
        """

        if not text or not mapping:
            return text

        restored_text = text

        # Replace longer tokens first.
        # This prevents partial token collisions.
        for token, original_value in sorted(
            mapping.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            restored_text = restored_text.replace(
                token,
                original_value,
            )

        return restored_text