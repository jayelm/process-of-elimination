from typing import List, Optional
import dataclasses
from dataclasses import dataclass


LETTERS = "ABCDE"


@dataclass
class Example:
    question: str
    choices: List[str]
    answer: str
    answer_index: int
    text: Optional[str] = None

    def to_str(self, format: str) -> str:
        if format == "test":
            # Format without answer, used for testing
            return self._test_format()
        elif format == "direct":
            return self._direct_format()
        elif format == "poe":
            return self._poe_format()
        else:
            raise NotImplementedError(f"format={format}")

    def _test_format(self):
        return f"{self.question}"

    def _direct_format(self):
        return f"{self.question}\nThe answer is {self.answer}."

    def _poe_format(self):
        choices = []
        for i, choice in enumerate(self.choices):
            if i == self.answer_index:
                continue
            letter = LETTERS[i]
            explanation = choice['explanation']
            # Lowercase just in case
            explanation = explanation[0].lower() + explanation[1:]
            if not explanation.endswith('.'):
                maybe_end = "."
            else:
                maybe_end = ""
            choices.append(
                f"The answer is not {letter}, since {explanation}{maybe_end}"
            )
        choices = "\n".join(choices)
        return f"{self.question}\n{choices}\nTherefore, the answer is {self.answer}."


@dataclass
class Dataset:
    examples: List[Example]
    format: Optional[str] = "test"
    prompt: Optional[str] = None
    instruction: Optional[str] = None

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> str:
        text =  self.examples[i].to_str(format=self.format)
        if self.prompt is not None:
            text = f"{self.prompt}\n\n{text}"
        if self.instruction is not None:
            text = f"{self.instruction}\n\n{text}"
        return dataclasses.replace(self.examples[i], text=text)

    def get_prompt(self):
        return "\n\n".join(list([x.text for x in self]))

    def __str__(self):
        return f"<Dataset of {len(self)} examples>"

    def __repr__(self):
        return str(self)
