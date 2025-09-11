import os
import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Tuple, Optional as OptionalType
from pony.orm import Database, Required, Optional, PrimaryKey, db_session, sql_debug, LongStr, commit, Json
from dotenv import load_dotenv


# Define enums for logging status and severity
class Severity(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    DEBUG = "DEBUG"


class Status(Enum):
    IN_PROGRESS = "in progress"
    SUCCESS = "success"
    ERROR = "error"


# Database instance for logs
db = Database()


class DBLogger:
    def __init__(self, table_name="model_generation", id=None, schema=None, logger=True):
        """
        Initialize the DBLogger with the specified table name and schema.
        Connection details are loaded from environment variables.

        Args:
            table_name: Name of the database table to use
            id: Optional ID to use for logging
            schema: Dictionary mapping field names to their types and constraints
                   Example: {'model_name': ('str', True), 'generation': ('str', False)}
                   where the tuple contains (type, required)
            logger: Whether to perform logging (default: True)
        """
        self.logger = logger
        self.generation_id = None
        self.object_key = None
        self.table_name = table_name
        self.entity_name = f"ModelGeneration_{table_name}"
        self.id = id
        self.schema = schema or {}  # Default to empty schema if none provided

        # Load environment variables
        load_dotenv()

        # Setup database connection
        self._setup_database()

    def _setup_database(self):
        """Set up the MySQL database connection and create the table if it doesn't exist."""
        # Get database connection details from environment variables
        db_host = os.environ.get("DB_HOST", "localhost")
        db_user = os.environ.get("DB_USER", "root")
        db_password = os.environ.get("DB_PASSWORD", "")
        db_name = os.environ.get("DB_NAME", "logs_db")
        db_port = int(os.environ.get("DB_PORT", "3306"))

        # Enable SQL debugging if specified in environment
        if os.environ.get("SQL_DEBUG", "false").lower() == "true":
            sql_debug(True)

        # Create the model class dynamically with the specified table name and schema
        self._create_entity_class()

        # Connect to the MySQL database if not already connected
        if not db.provider:
            db.bind(
                provider='mysql',
                host=db_host,
                user=db_user,
                passwd=db_password,
                db=db_name,
                port=db_port
            )

            # Generate mapping and create tables if they don't exist
            db.generate_mapping(create_tables=True)

    def _create_entity_class(self):
        """Dynamically create the entity class with the fixed fields and user-defined schema."""
        # Check if entity with this name already exists
        if self.entity_name not in globals():
            if self.logger:
                # Start with fixed attributes
                entity_attrs = {
                    "_table_": self.table_name,
                    "key": PrimaryKey(int, auto=True),
                    "id": Required(str),
                    "log": Optional(LongStr, nullable=True),  # Explicitly set nullable=True
                    "status": Required(str),
                    "created_at": Required(datetime, default=datetime.now)
                }
            else:
                entity_attrs = {
                    "_table_": self.table_name,
                    "id": PrimaryKey(int, auto=True),
                }

            # Map Python types to Pony ORM types
            type_mapping = {
                'str': str,
                'longstr': LongStr,
                'int': int,
                'float': float,
                'bool': bool,
                'datetime': datetime,
                'json': Json,
            }

            # Add schema-defined attributes
            for field_name, field_spec in self.schema.items():
                field_type, is_required = field_spec

                # Skip fields that might conflict with fixed fields
                if field_name in entity_attrs:
                    continue

                # Get the appropriate Pony type
                pony_type = type_mapping.get(field_type.lower(), str)

                # Add as either Required or Optional field
                if is_required:
                    entity_attrs[field_name] = Required(pony_type)
                else:
                    # For optional fields, explicitly set nullable=True
                    entity_attrs[field_name] = Optional(pony_type, nullable=True)

            # Create the entity class and register it globally
            entity_class = type(self.entity_name, (db.Entity,), entity_attrs)
            globals()[self.entity_name] = entity_class

            # Store the class reference for later use
            self.model_class = entity_class
        else:
            # Use the existing entity class
            self.model_class = globals()[self.entity_name]

    def _format_log_entry(self, message, severity=Severity.INFO):
        """
        Format a log entry with timestamp and severity level
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"[{timestamp}] [{severity.value}] {message}"

    @db_session
    def start_logger(self, data=None, id=None, status=Status.IN_PROGRESS):
        """
        Create a new log entry in the database with initial information.
        Returns the ID of the created record.

        Args:
            data: Dictionary containing the values for all fields in the schema
            id: Optional ID to use, defaults to a UUID if not provided
        """
        unique_id = str(uuid.uuid4()) if id is None else id

        if self.id is not None and id is None:
            unique_id = self.id

        # Prepare the entity creation args
        entity_args = {}
        if self.logger:
            entity_args['log'] = self._format_log_entry(f"Process started with status: {status.value}")
            entity_args['status'] = status.value
            entity_args['id'] = unique_id


        # Add data for schema fields if provided
        if data:
            for field, value in data.items():
                if field not in entity_args:  # Don't override fixed fields
                    entity_args[field] = value

        # Create and persist the entity object
        new_entity = self.model_class(**entity_args)
        commit()  # Commit to get the assigned key

        # Store the ID and key for later updates
        self.generation_id = unique_id
        self.object_key = new_entity.key if hasattr(new_entity, 'key') else new_entity.id

        return unique_id

    @db_session
    def log(self, message, severity=Severity.INFO):
        """
        Update the log field for the current entity and persist to the database.
        Allows specifying the severity level of the log entry.
        """
        if not self.object_key:
            raise ValueError("Logger not initialized. Call start_logger first.")

        # Get the current object
        entity = self.model_class[self.object_key]

        # Format and append new log message
        new_entry = self._format_log_entry(message, severity)

        # Append to existing log or create new log
        entity.log = f"{entity.log}\n{new_entry}" if entity.log else new_entry

        commit()

    @db_session
    def update(self, data):
        """
        Update specific fields of the current record.
        Handles NULL values by setting fields to None.

        Args:
            data: Dictionary containing the field names and values to update
        """
        if not self.object_key:
            raise ValueError("Logger not initialized. Call start_logger first.")

        # Get the current object
        entity = self.model_class[self.object_key]

        # Update fields
        for field, value in data.items():
            if hasattr(entity, field):
                # Set value to None if it's explicitly meant to be NULL
                setattr(entity, field, value)

        commit()

    @db_session(optimistic=False)
    def finalize(self, status=Status.SUCCESS, data=None):
        """
        Update the status and optionally other fields, marking the process as complete.

        Args:
            status: Status enum value (SUCCESS or ERROR)
            data: Optional dictionary with additional fields to update
        """
        if not self.object_key:
            raise ValueError("Logger not initialized. Call start_logger first.")

        if status not in [Status.SUCCESS, Status.ERROR]:
            raise ValueError("Status must be either Status.SUCCESS or Status.ERROR")

        # Get the current object
        entity = self.model_class[self.object_key]

        # Update status
        entity.status = status.value

        # Update any additional fields if provided
        if data:
            for field, value in data.items():
                if hasattr(entity, field):
                    setattr(entity, field, value)

        # Add a final log entry with appropriate severity
        severity = Severity.INFO if status == Status.SUCCESS else Severity.ERROR
        self.log(f"Process finalized with status: {status.value}", severity)

        commit()
        return self.generation_id