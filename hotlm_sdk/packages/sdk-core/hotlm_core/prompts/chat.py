from __future__ import annotations

from typing import Any, Dict, List, Union

from hotlm_core.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage

from .base import BaseChatPromptTemplate


class MessagesPlaceholder:
    """Placeholder for injecting memory or other message lists into a chat prompt."""
    def __init__(self, variable_name: str):
        self.variable_name = variable_name

class HumanMessagePromptTemplate:
    """Template for formatting a HumanMessage with a string template."""
    def __init__(self, template: str):
        self.template = template

    @classmethod
    def from_template(cls, template: str) -> HumanMessagePromptTemplate:
        return cls(template)

    def format(self, **kwargs: Any) -> HumanMessage:
        content = self.template.format(**kwargs)
        return HumanMessage(content=content)

class AIMessagePromptTemplate:
    """Template for formatting an AIMessage with a string template."""
    def __init__(self, template: str):
        self.template = template

    @classmethod
    def from_template(cls, template: str) -> AIMessagePromptTemplate:
        return cls(template)

    def format(self, **kwargs: Any) -> AIMessage:
        content = self.template.format(**kwargs)
        return AIMessage(content=content)

class ChatPromptTemplate(BaseChatPromptTemplate):
    """Basic chat prompt that stitches static messages, placeholders, and formatted messages."""
    def __init__(self, parts: List[Union[SystemMessage, MessagesPlaceholder, HumanMessagePromptTemplate, AIMessagePromptTemplate]]):
        self.parts = parts

    @classmethod
    def from_messages(
        cls,
        parts: List[Union[SystemMessage, MessagesPlaceholder, HumanMessagePromptTemplate, AIMessagePromptTemplate]]
    ) -> ChatPromptTemplate:
        return cls(parts)

    def format_messages(self, **kwargs: Any) -> List[BaseMessage]:
        messages: List[BaseMessage] = []
        for part in self.parts:
            if isinstance(part, SystemMessage):
                messages.append(part)
            elif isinstance(part, MessagesPlaceholder):
                msgs = kwargs.get(part.variable_name, [])
                if isinstance(msgs, list):
                    messages.extend(msgs)
            elif isinstance(part, HumanMessagePromptTemplate):
                messages.append(part.format(**kwargs))
            elif isinstance(part, AIMessagePromptTemplate):
                messages.append(part.format(**kwargs))
        return messages
