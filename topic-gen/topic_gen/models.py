import xml.etree.ElementTree as ET
from typing import Generic, List, TypeVar, get_args, Optional

from pydantic import BaseModel, Field, PrivateAttr, computed_field
import uuid
import json

T = TypeVar("T")  # generic type that will be provided later


class BaseTopic(BaseModel):
    """Base class for topics, providing a common interface for XML conversion."""
    _id: Optional[str] = PrivateAttr(default=str(uuid.uuid4()))

    @computed_field
    @property
    def topic_id(self) -> str:
        return self._id

    def to_xml(self, output: str = None) -> str:
        """Converts the topic object to an XML string."""
        xml = ""
        for field, value in self:
            xml += f"<{field}>{value}</{field}>\n"
        xml = f"<topic>\n{xml}</topic>"
        if output:
            with open(output, "w") as f:
                f.write(xml)
        return xml.strip()


class TRECTopic(BaseTopic):
    title: str = Field()
    description: str = Field()
    narrative: str = Field()


# Reference: Developing a test collection for the evaluation of integrated search, Lykke et al. (2010)
class iSearchTopic(BaseTopic):
    current_information_need: str = Field()
    work_task: str = Field()
    background_knowledge: str = Field()
    ideal_answer: str = Field()
    search_terms: List[str] = Field()


class Topics(BaseModel, Generic[T]):
    """A list of generated topics, used for the final output."""

    topics: List[T] = Field(description="A list of generated Topic objects.")

    def to_xml(self, output: str = None) -> str:
        """Converts the TopicList object to an XML string."""
        topics_xml = "\n".join(topic.to_xml() for topic in self.topics)
        xml = f"<topics>\n{topics_xml}\n</topics>"
        if output:
            with open(output, "w") as f:
                f.write(xml)
        return xml.strip()

    @classmethod
    def load_ird_topics(cls, ird: str, k: Optional[int] = None) -> "Topics[T]":
        """Loads topics from ir_datasets."""
        import ir_datasets
        dataset = ir_datasets.load(ird)
        queries = list(dataset.queries)
        if k:
            queries = queries[:k]
        return Topics[dataset.queries.type](topics=queries)

    @classmethod
    def read_xml(cls, file_path: str) -> "Topics[T]":
        """Loads topics from an XML file."""
        field_annotation = cls.model_fields["topics"].annotation
        topic_model = get_args(field_annotation)[0]

        tree = ET.parse(file_path)
        root = tree.getroot()
        topics = []
        for topic in root.findall("topic"):
            topic_data = {}
            for elem in topic:
                topic_data[elem.tag] = elem.text
            topic_obj = topic_model.model_validate(topic_data)
            topics.append(topic_obj)
        return cls(topics=topics)

    @classmethod
    def read_jsonl(cls, file_path: str) -> "Topics[T]":
        """Loads topics from a JSONL file."""
        field_annotation = cls.model_fields["topics"].annotation
        topic_model = get_args(field_annotation)[0]
        topics = []
        with open(file_path, "r") as f:

            for line in f:
                topic_json = json.loads(line)
                topic = topic_model.model_validate(topic_json)
                if topic_id := topic_json.get("topic_id"):
                    topic._id = topic_id
                topics.append(topic)

        return cls(topics=topics)

    @classmethod
    def from_csv(cls, file_path: str) -> "Topics[T]":
        """Loads topics from a CSV file."""
        import pandas as pd
        field_annotation = cls.model_fields["topics"].annotation
        topic_model = get_args(field_annotation)[0]
        df = pd.read_csv(file_path)
        topics = []
        for _, row in df.iterrows():
            topic_json = row.to_dict()
            topic = topic_model.model_validate(topic_json)
            if topic_id := topic_json.get("topic_id"):
                topic._id = topic_id
            topics.append(topic)
        return cls(topics=topics)

    def to_jsonl(self, file_path: str):
        """Dumps the topics to a JSONL file."""
        with open(file_path, "w") as f:
            for topic in self.topics:
                json.dump(topic.model_dump(), f)
                f.write("\n")

    def to_csv(self, file_path: str):
        """Dumps the topics to a CSV file."""
        import pandas as pd
        df = pd.DataFrame([topic.model_dump() for topic in self.topics])
        df.to_csv(file_path, index=False)


class Qrels(BaseModel, Generic[T]):
    """A list of generated qrels, used for the final output."""

    qrels: List[T] = Field(description="List of Qrels items")


# Responds class for intent match (M), trustworthiness (T), and final score (O) annotation as used by Thomas et al. (2024).
# Reference: Large Language Models can Accurately Predict Searcher Preferences, Thomas et al. (2024)
class MTO_responds(BaseModel):
    M: int = Field(description="Intent matching (M)")
    T: int = Field(description="Trustworthiness (T)")
    O: int = Field(description="Final score (O)")
