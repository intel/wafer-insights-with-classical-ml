class OrionError(Exception):
    msg = """Base WaferInsights Exception"""


class FailedConnToDB(OrionError):
    msg = """Connection to database was broken due to the following stack trace."""


class MetaTableNotCreated(OrionError):
    msg = """Cannot create metadata table."""


class MetadataFailedInsertLoad(OrionError):
    msg = """Execute and commit load to metadata failed."""


class NoLoadData(OrionError):
    msg = """There was no real load date available, data may be corrupt."""
