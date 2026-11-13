"""Connect the generated local schema to DIN-SQL execution."""
from sklearn.feature_extraction.text import TfidfVectorizer
from tarqa.reasoning.utils import digest
from tarqa.reasoning.din_sql import DemoBank, DINSQLEngine

class ArchivedDemoBank(DemoBank):
    """Reuse exactly the original TRAIN annotations and retrieval order."""
    def __init__(self, rows, *, num_examples):
        self.rows = rows
        self.num_examples = int(num_examples)
        if not rows or self.num_examples < 1:
            raise ValueError('An existing DIN training demonstration bank is required')
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True,
                                         token_pattern=r'(?u)\b\w+\b')
        self.matrix = self.vectorizer.fit_transform([r['question'] for r in rows])
        self.signature = digest(rows)


class ContextDINSQLEngine(DINSQLEngine):
    """Apply the TSG-selected schema with unchanged training-demo retrieval."""
    def __init__(self, client, bank, *, trace_root, on_stage=None):
        super().__init__(client, bank, '', trace_root=trace_root)
        self.task_spec = None
        self.stage_records = []
        self.on_stage = on_stage

    def configure(self, public):
        self.schema = public['local_schema']
        if self.schema is None:
            raise ValueError('No usable predicted local schema; no full-schema fallback')
        self.task_spec = public['task_spec']
        self.stage_records = []

    def messages(self, stage, question, examples, **fields):
        messages = super().messages(stage, question, examples, **fields)
        messages[0]['content'] = messages[0]['content'].replace(
            'The schema below also applies to every demonstration.',
            'The schema below applies to the current question. Training demonstrations may reference other tables.')
        return messages

    def call(self, uid, stage, question, **fields):
        if self.on_stage:
            self.on_stage(uid, stage)
        text, record = super().call(uid, stage, question, **fields)
        self.stage_records.append(record)
        return text, record
