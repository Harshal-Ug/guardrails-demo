import re
import time
from typing import Any, Callable
from typing_extensions import NotRequired

from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ToolCallRequest,
)
from langchain.messages import HumanMessage, AIMessage
from langgraph.runtime import Runtime

from anonymizer import PIIAnonymizer


PLACEHOLDER_PATTERN = re.compile(
    r"\[(?:PERSON|LOCATION|EMAIL|PHONE)_\d+\]"
)


class GuardRailState(AgentState):
    pii_mapping: NotRequired[dict[str, str]]
    sanitized_prompt: NotRequired[str]
    detected_pii: NotRequired[list[dict[str, str]]]
    raw_response: NotRequired[str]
    final_response: NotRequired[str]

    masking_latency_ms: NotRequired[float]
    unmasking_latency_ms: NotRequired[float]


class GuardRailMiddleware(AgentMiddleware[GuardRailState]):

    state_schema = GuardRailState

    def __init__(self):
        super().__init__()
        self.anonymizer = PIIAnonymizer()

    # --------------------------------------------------
    # BEFORE MODEL
    # --------------------------------------------------

    def before_model(
        self,
        state: GuardRailState,
        runtime: Runtime
    ) -> dict[str, Any] | None:

        start_time = time.perf_counter()

        mapping = dict(
            state.get("pii_mapping", {})
        )

        masked_messages = []

        sanitized_prompt = state.get(
            "sanitized_prompt",
            ""
        )

        detected_pii = list(
            state.get(
                "detected_pii",
                []
            )
        )

        for message in state["messages"]:

            if not isinstance(message, HumanMessage):
                masked_messages.append(message)
                continue

            content = message.content

            if not isinstance(content, str):
                masked_messages.append(message)
                continue

            # --------------------------------------------------
            # IMPORTANT:
            # If this message has already been sanitized,
            # do not run PII detection on it again.
            # --------------------------------------------------

            if PLACEHOLDER_PATTERN.search(content):

                masked_messages.append(message)

                sanitized_prompt = content

                continue

            # --------------------------------------------------
            # First-time PII masking
            # --------------------------------------------------

            masked_content, new_mapping = (
                self.anonymizer.mask(content)
            )

            mapping.update(new_mapping)

            sanitized_prompt = masked_content

            # --------------------------------------------------
            # Store detected PII
            # --------------------------------------------------

            for token, original_value in new_mapping.items():

                entity_type = (
                    token
                    .strip("[]")
                    .rsplit("_", 1)[0]
                )

                already_exists = any(
                    item["token"] == token
                    for item in detected_pii
                )

                if not already_exists:

                    detected_pii.append({
                        "type": entity_type,
                        "value": original_value,
                        "token": token
                    })

            masked_messages.append(
                message.model_copy(
                    update={
                        "content": masked_content
                    }
                )
            )

        # --------------------------------------------------
        # ACCUMULATE MASKING LATENCY
        # --------------------------------------------------

        current_masking_latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        previous_masking_latency_ms = state.get(
            "masking_latency_ms",
            0.0
        )

        total_masking_latency_ms = (
            previous_masking_latency_ms
            + current_masking_latency_ms
        )

        return {
            "messages": masked_messages,

            "pii_mapping": mapping,

            "sanitized_prompt": sanitized_prompt,

            "detected_pii": detected_pii,

            "masking_latency_ms": (
                total_masking_latency_ms
            )
        }

    # --------------------------------------------------
    # TRUSTED TOOL BOUNDARY
    # --------------------------------------------------

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable
    ):

        mapping = request.state.get(
            "pii_mapping",
            {}
        )

        tool_call = request.tool_call

        arguments = dict(
            tool_call.get("args", {})
        )

        # Restore PII only immediately before
        # sending data to the trusted backend.

        for argument_name, argument_value in arguments.items():

            if not isinstance(argument_value, str):
                continue

            for token, original_value in mapping.items():

                argument_value = argument_value.replace(
                    token,
                    original_value
                )

            arguments[argument_name] = argument_value

        updated_tool_call = {
            **tool_call,
            "args": arguments
        }

        updated_request = request.override(
            tool_call=updated_tool_call
        )

        return handler(updated_request)

    # --------------------------------------------------
    # AFTER AGENT
    # --------------------------------------------------

    def after_agent(
        self,
        state: GuardRailState,
        runtime: Runtime
    ) -> dict[str, Any] | None:

        start_time = time.perf_counter()

        messages = state.get(
            "messages",
            []
        )

        if not messages:
            return None

        last_ai_message = None

        for message in reversed(messages):

            if isinstance(message, AIMessage):
                last_ai_message = message
                break

        if last_ai_message is None:
            return None

        raw_response = last_ai_message.content

        if not isinstance(raw_response, str):
            return None

        mapping = state.get(
            "pii_mapping",
            {}
        )

        # --------------------------------------------------
        # Restore original PII
        # --------------------------------------------------

        final_response = self.anonymizer.unmask(
            raw_response,
            mapping
        )

        unmasking_latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        return {
            "raw_response": raw_response,

            "final_response": final_response,

            "sanitized_prompt": state.get(
                "sanitized_prompt",
                ""
            ),

            "detected_pii": state.get(
                "detected_pii",
                []
            ),

            "pii_mapping": mapping,

            "masking_latency_ms": state.get(
                "masking_latency_ms",
                0.0
            ),

            "unmasking_latency_ms": (
                unmasking_latency_ms
            ),

            "messages": [
                AIMessage(
                    content=final_response
                )
            ]
        }