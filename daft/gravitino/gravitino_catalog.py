from __future__ import annotations

import dataclasses
import warnings
from typing import Literal
from urllib.parse import urlparse

from daft.io import AzureConfig, IOConfig, S3Config
from gravitino import GravitinoClient as OfficialGravitinoClient
from gravitino import NameIdentifier
from gravitino.exceptions.base import (
    NoSuchCatalogException,
    NoSuchSchemaException,
    NotFoundException,
)


@dataclasses.dataclass(frozen=True)
class GravitinoTableInfo:
    """Information about a Gravitino table."""

    name: str
    catalog: str
    schema: str
    table_type: str
    storage_location: str
    format: str
    properties: dict[str, str]


@dataclasses.dataclass(frozen=True)
class GravitinoFilesetInfo:
    """Information about a Gravitino fileset."""

    name: str
    catalog: str
    schema: str
    fileset_type: str
    storage_location: str
    properties: dict[str, str]


@dataclasses.dataclass(frozen=True)
class GravitinoCatalog:
    """Represents a catalog in Gravitino."""

    name: str
    type: str
    provider: str
    properties: dict[str, str]


@dataclasses.dataclass(frozen=True)
class GravitinoTable:
    """Represents a table in Gravitino catalog."""

    table_info: GravitinoTableInfo
    table_uri: str
    io_config: IOConfig | None


@dataclasses.dataclass(frozen=True)
class GravitinoFileset:
    """Represents a fileset in Gravitino catalog."""

    fileset_info: GravitinoFilesetInfo
    io_config: IOConfig | None


def _io_config_from_storage_location(storage_location: str, properties: dict[str, str]) -> IOConfig | None:
    """Create IOConfig from storage location and properties."""
    scheme = urlparse(storage_location).scheme

    if scheme == "s3" or scheme == "s3a":
        # Extract S3 credentials from properties if available
        access_key = properties.get("s3-access-key-id")
        secret_key = properties.get("s3-secret-access-key")
        endpoint_url = properties.get("s3-endpoint")
        session_token = properties.get("s3.session-token")
        region_name = properties.get("s3-region")

        # Try to extract region from endpoint URL if not explicitly provided
        if not region_name and endpoint_url:
            import re

            # Match patterns like "s3.ap-northeast-1.amazonaws.com" or "s3-ap-northeast-1.amazonaws.com"
            region_match = re.search(r"s3[.-]([a-z0-9-]+)\.amazonaws\.com", endpoint_url)
            if region_match:
                region_name = region_match.group(1)

        if access_key and secret_key:
            s3_config = S3Config(
                region_name=region_name,
                key_id=access_key,
                access_key=secret_key,
                endpoint_url=endpoint_url,
                session_token=session_token,
            )

            return IOConfig(s3=s3_config)
        return None
    elif scheme == "gcs" or scheme == "gs":
        # GCS configuration would go here
        warnings.warn("GCS credential configuration from Gravitino is not yet fully supported.")
        return None
    elif scheme == "az" or scheme == "abfs" or scheme == "abfss":
        # Extract Azure credentials from properties if available
        sas_token = properties.get("azure.sas-token")
        if sas_token:
            return IOConfig(azure=AzureConfig(sas_token=sas_token))
        return None
    else:
        warnings.warn(f"Credentials for scheme {scheme} are not yet supported.")
        return None


class GravitinoClient:
    """Client to access Apache Gravitino catalog.

    Apache Gravitino is an open-source data catalog that provides unified metadata management
    for various data sources and storage systems.

    Example of reading a dataframe from a table in Gravitino:

    >>> client = GravitinoClient("http://localhost:8090", "my_metalake", auth_type="simple", username="admin")
    >>> table = client.load_table("my_catalog.my_schema.my_table")
    >>> df = daft.read_iceberg(table)
    """

    def __init__(
        self,
        endpoint: str,
        metalake_name: str,
        auth_type: Literal["simple", "oauth2"] = "simple",
        username: str | None = None,
        password: str | None = None,
        token: str | None = None,
    ):
        """Initialize Gravitino client.

        Args:
            endpoint: Gravitino server endpoint URL
            metalake_name: Name of the metalake to connect to
            auth_type: Authentication type ("simple" or "oauth2")
            username: Username for simple auth
            password: Password for simple auth
            token: OAuth2 token for oauth2 auth
        """
        self._endpoint = endpoint.rstrip("/")
        self._metalake_name = metalake_name
        self._auth_type = auth_type
        self._username = username
        self._password = password
        self._token = token

        # Store connection parameters - create client lazily when needed
        self._client = None

    def _get_client(self) -> OfficialGravitinoClient:
        """Get or create the Gravitino client."""
        if self._client is None:
            self._client = OfficialGravitinoClient(uri=self._endpoint, metalake_name=self._metalake_name)
        return self._client

    def list_catalogs(self) -> list[str]:
        """List all available catalogs in the metalake."""
        try:
            catalog_identifiers = self._get_client().list_catalogs()
            return [ident.name() for ident in catalog_identifiers]
        except Exception as e:
            warnings.warn(f"Failed to list catalogs: {e}")
            return []

    def load_catalog(self, catalog_name: str) -> GravitinoCatalog:
        """Load a Gravitino catalog.

        Args:
            catalog_name: Name of the catalog to load

        Returns:
            GravitinoCatalog object

        Raises:
            Exception: If catalog is not found or cannot be loaded
        """
        try:
            catalog = self._get_client().load_catalog(name=catalog_name)

            return GravitinoCatalog(
                name=catalog.name(),
                type=catalog.type().value[0],
                provider=catalog.provider(),
                properties=catalog.properties(),
            )

        except NoSuchCatalogException:
            raise Exception(f"Catalog {catalog_name} not found")
        except Exception as e:
            raise Exception(f"Failed to load catalog {catalog_name}: {e}")

    def list_schemas(self, catalog_name: str) -> list[str]:
        """List schemas in a catalog."""
        try:
            catalog = self._get_client().load_catalog(name=catalog_name)
            schema_identifiers = catalog.as_schemas().list_schemas()
            return [f"{catalog_name}.{ident.name()}" for ident in schema_identifiers]
        except Exception as e:
            warnings.warn(f"Failed to list schemas for catalog {catalog_name}: {e}")
            return []

    def list_tables(self, schema_name: str) -> list[str]:
        """List tables in a schema."""
        if schema_name.count(".") != 1:
            raise ValueError(
                f"Expected fully-qualified schema name with format `catalog_name`.`schema_name`, but received: {schema_name}"
            )

        catalog_name, schema_name_only = schema_name.split(".")

        try:
            catalog = self._get_client().load_catalog(name=catalog_name)

            # Check if catalog type is relational
            if catalog.type().value[0] != "relational":
                warnings.warn(
                    f"Catalog '{catalog_name}' is of type '{catalog.type().value[0]}', not 'relational'. Returning empty table list."
                )
                return []

            schema = catalog.as_schemas().load_schema(schema_name=schema_name_only)
            table_identifiers = schema.as_table_catalog().list_tables()
            return [f"{catalog_name}.{schema_name_only}.{ident.name()}" for ident in table_identifiers]
        except Exception as e:
            warnings.warn(f"Failed to list tables for schema {schema_name}: {e}")
            return []

    def load_table(self, table_name: str) -> GravitinoTable:
        """Load an existing Gravitino table.

        Args:
            table_name: Name of the table in the form catalog.schema.table

        Returns:
            GravitinoTable object

        Raises:
            ValueError: If table name format is invalid
            Exception: If table is not found or cannot be loaded
        """
        parts = table_name.split(".")
        if len(parts) != 3:
            raise ValueError(f"Expected table name format 'catalog.schema.table', got: {table_name}")

        catalog_name, schema_name, table_name_only = parts

        try:
            catalog = self._get_client().load_catalog(name=catalog_name)

            # Check if catalog type is relational
            if catalog.type().value[0] != "relational":
                raise Exception(
                    f"Only relational catalog supports 'load_table' method, but catalog '{catalog_name}' is of type '{catalog.type().value[0]}'"
                )

            schema = catalog.as_schemas().load_schema(schema_name=schema_name)
            table_ident = NameIdentifier.of(schema_name, table_name_only)
            table = schema.as_table_catalog().load_table(ident=table_ident)

            properties = table.properties()

            # Handle storage locations - try to get from table properties
            storage_location = ""

            # Check for location in properties (common pattern)
            if "location" in properties:
                storage_location = properties["location"]
            elif "path" in properties:
                storage_location = properties["path"]
            elif "warehouse" in properties:
                # For Iceberg tables, warehouse might be the base location
                storage_location = properties["warehouse"]

            # Convert Gravitino file URL format to Daft-compatible format
            # Gravitino returns "file:/path" but Daft expects "file:///path"
            if storage_location.startswith("file:/") and not storage_location.startswith("file:///"):
                storage_location = storage_location.replace("file:/", "file:///", 1)

            # Determine table format from properties or provider
            table_format = properties.get("format", "ICEBERG")
            if not table_format:
                # Try to infer from provider or other properties
                provider = getattr(table, "provider", lambda: "")()
                if provider.upper() in ["ICEBERG", "HIVE", "DELTA"]:
                    table_format = provider.upper()
                else:
                    table_format = "ICEBERG"  # Default fallback

            table_info = GravitinoTableInfo(
                name=table.name(),
                catalog=catalog_name,
                schema=schema_name,
                table_type=getattr(table, "provider", lambda: "")(),
                storage_location=storage_location,
                format=table_format,
                properties=properties,
            )

        except NotFoundException:
            raise Exception(f"Table {table_name} not found")
        except NoSuchSchemaException:
            raise Exception(f"Schema {catalog_name}.{schema_name} not found")
        except NoSuchCatalogException:
            raise Exception(f"Catalog {catalog_name} not found")
        except Exception as e:
            raise Exception(f"Failed to load table {table_name}: {e}")

        # Create IO config from table properties
        io_config = _io_config_from_storage_location(table_info.storage_location, table_info.properties)

        return GravitinoTable(
            table_info=table_info,
            table_uri=table_info.storage_location,
            io_config=io_config,
        )

    def load_fileset(self, fileset_name: str) -> GravitinoFileset:
        """Load a Gravitino fileset.

        Args:
            fileset_name: Name of the fileset in the form catalog.schema.fileset

        Returns:
            GravitinoFileset object
        """
        parts = fileset_name.split(".")
        if len(parts) != 3:
            raise ValueError(f"Expected fileset name format 'catalog.schema.fileset', got: {fileset_name}")

        catalog_name, schema_name, fileset_name_only = parts

        try:
            catalog = self._get_client().load_catalog(name=catalog_name)

            # Check if catalog type is fileset
            if catalog.type().value[0] != "fileset":
                raise Exception(
                    f"Only fileset catalog supports 'load_fileset' method, but catalog '{catalog_name}' is of type '{catalog.type().value[0]}'"
                )

            fileset_ident = NameIdentifier.of(schema_name, fileset_name_only)
            fileset = catalog.as_fileset_catalog().load_fileset(ident=fileset_ident)

            properties = fileset.properties()

            # Get storage location from fileset
            storage_locations = fileset.storage_locations()
            # Use the first storage location from the dictionary
            storage_location = next(iter(storage_locations.values())) if storage_locations else ""

            # Convert Gravitino URL formats to Daft-compatible formats
            # Gravitino returns "file:/path" but Daft expects "file:///path"
            if storage_location.startswith("file:/") and not storage_location.startswith("file:///"):
                storage_location = storage_location.replace("file:/", "file:///", 1)

            fileset_info = GravitinoFilesetInfo(
                name=fileset.name(),
                catalog=catalog_name,
                schema=schema_name,
                fileset_type=fileset.type().value,
                storage_location=storage_location,
                properties=properties,
            )

            # Create IO config from fileset properties
            io_config = _io_config_from_storage_location(fileset_info.storage_location, fileset_info.properties)

            return GravitinoFileset(fileset_info=fileset_info, io_config=io_config)

        except Exception as e:
            raise Exception(f"Failed to load fileset {fileset_name}: {e}")

    def to_io_config(self) -> IOConfig:
        """Convert client configuration to IOConfig.

        Returns an IOConfig with only the Gravitino configuration from this client.
        S3 and other storage credentials are handled per-fileset by the Gravitino source.
        """
        from daft.io import GravitinoConfig

        gravitino_config = GravitinoConfig(
            endpoint=self._endpoint,
            metalake_name=self._metalake_name,
            auth_type=self._auth_type,
            username=self._username,
            password=self._password,
            token=self._token,
        )
        # Only include Gravitino config - let Gravitino source handle storage credentials
        return IOConfig(gravitino=gravitino_config)
