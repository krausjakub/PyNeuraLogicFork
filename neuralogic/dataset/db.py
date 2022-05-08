import io
from typing import Optional, List, Union

from neuralogic.core.constructs.atom import BaseAtom, WeightedAtom
from neuralogic.core.constructs.rule import Rule
from neuralogic.dataset.logic import Dataset
from neuralogic.dataset.csv import CSVDataset, CSVFile, Mode
from neuralogic.dataset.base import BaseDataset

DatasetEntries = Union[BaseAtom, WeightedAtom, Rule]


class DBSource:
    __slots__ = "relation_name", "table_name", "term_columns", "value_column", "sep", "default_value", "skip_rows", "n_rows", "replace_empty_column"

    def __init__(
        self,
        relation_name: str,
        table_name: str,
        term_columns: List[str],
        value_column: Optional[str] = None,
        default_value: Union[float, int] = 1.0,
        skip_rows: int = 0,
        n_rows: Optional[int] = None,
        replace_empty_column: Union[str, float, int] = 0,
        sep=",",
    ):
        self.table_name = table_name
        self.relation_name = relation_name
        self.sep = sep
        self.value_column = value_column
        self.default_value = default_value
        self.term_columns = term_columns
        self.skip_rows = skip_rows
        self.n_rows = n_rows
        self.replace_empty_column = replace_empty_column

        if len(term_columns) == 0:
            raise NotImplementedError(f"Cannot create DBSource with zero terms")

    def to_csv(self, cursor) -> CSVFile:
        source = io.StringIO()

        columns = [term for term in self.term_columns]
        term_columns = list(range(len(columns)))
        value_column = None

        if self.value_column is not None:
            columns.append(self.value_column)
            value_column = len(columns) - 1

        cursor.copy_to(source, self.table_name, sep=self.sep, null="", columns=columns)
        source.seek(0)

        return CSVFile(
            self.relation_name, source, self.sep, value_column, self.default_value,
            term_columns, False, self.skip_rows, self.n_rows, self.replace_empty_column
        )


class DBDataset(BaseDataset):
    def __init__(
        self,
        connection,
        db_sources: Union[List[DBSource], DBSource],
        queries: Optional[List[Union[List[DatasetEntries], DatasetEntries]]] = None,
        mode: Mode = Mode.ONE_EXAMPLE,
    ):
        self.connection = connection
        self.db_sources = [db_sources] if isinstance(db_sources, DBSource) else db_sources
        self.queries: List[Union[List[DatasetEntries], DatasetEntries]] = queries if queries is not None else []
        self.mode = mode

    def add_db_source(self, db_source: DBSource):
        self.db_sources.append(db_source)

    def add_query(self, query):
        self.add_queries([query])

    def add_queries(self, queries: List):
        self.queries.extend(queries)

    def set_queries(self, queries: List):
        self.queries = queries

    def to_dataset(self) -> Dataset:
        with self.connection.cursor() as cur:
            csv_files = [db_source.to_csv(cur) for db_source in self.db_sources]
        csv_dataset = CSVDataset(csv_files, self.queries, self.mode)

        return csv_dataset.to_dataset()
