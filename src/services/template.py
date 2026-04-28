"""Template service for managing SQL templates."""

from typing import Any

import dill
import pandas as pd

from ..exceptions.base import ServiceInitializationError, TemplateError
from ..utils.timing import log_with_time_elapsed
from .base import Service


class TemplateService(Service):
    """Service for managing SQL templates and compilation."""

    def __init__(self, config, logger=None):
        super().__init__(config, logger)
        self._templates_df: pd.DataFrame | None = None

    def initialize(self) -> None:
        """Initialize template service by loading templates data."""
        try:
            with log_with_time_elapsed("Template service initialization", self.logger):
                templates_path = self.config.templates_path

                if not templates_path.exists():
                    raise TemplateError(None, f"Templates file not found: {templates_path}")

                self._templates_df = pd.read_json(templates_path, lines=True)
                self._mark_initialized()

                self.logger.info(
                    f"Template service initialized with {len(self._templates_df)} templates"
                )

        except Exception as e:
            self.logger.exception(f"Failed to initialize TemplateService: {str(e)}")
            raise ServiceInitializationError("TemplateService", str(e)) from e

    def cleanup(self) -> None:
        """Cleanup template service resources."""
        self._templates_df = None
        self._initialized = False

    def load_compiled_template(self, template_id: int) -> dict[str, Any]:
        """Load a compiled template by ID.

        Args:
            template_id: Template ID to load

        Returns:
            Compiled template dictionary

        Raises:
            TemplateError: If template loading fails
        """
        self.ensure_initialized()

        try:
            template_file = self.config.compiled_templates_path / f"{template_id}.pkl"

            if not template_file.exists():
                raise TemplateError(
                    template_id, f"Compiled template file not found: {template_file}"
                )

            with open(template_file, "rb") as f:
                compiled_template = dill.load(f)

            return compiled_template

        except Exception as e:
            self.logger.exception(f"Failed to load compiled template {template_id}: {str(e)}")
            raise TemplateError(template_id, f"Failed to load compiled template: {str(e)}") from e

    def get_template_info(self, template_id: int) -> dict[str, Any]:
        """Get information about a template.

        Args:
            template_id: Template ID

        Returns:
            Template information dictionary
        """
        self.ensure_initialized()

        if not (0 <= template_id < len(self._templates_df)):
            raise TemplateError(template_id, f"Template ID {template_id} out of range")

        template_row = self._templates_df.iloc[template_id]

        return {
            "template_id": template_id,
            "template": template_row.get("template", ""),
            "q_ids": template_row.get("q_ids", []),
            "num_questions": template_row.get("num_questions", 0),
            "timestamp": template_row.get("timestamp", ""),
        }

    def get_all_templates(self) -> pd.DataFrame:
        """Get all templates DataFrame.

        Returns:
            Templates DataFrame
        """
        self.ensure_initialized()
        return self._templates_df.copy()

    def template_exists(self, template_id: int) -> bool:
        """Check if a template exists.

        Args:
            template_id: Template ID to check

        Returns:
            True if template exists
        """
        self.ensure_initialized()
        return 0 <= template_id < len(self._templates_df)

    def compiled_template_exists(self, template_id: int) -> bool:
        """Check if a compiled template exists.

        Args:
            template_id: Template ID to check

        Returns:
            True if compiled template exists
        """
        template_file = self.config.compiled_templates_path / f"{template_id}.pkl"
        return template_file.exists()

    def get_templates_count(self) -> int:
        """Get total number of templates.

        Returns:
            Number of templates
        """
        self.ensure_initialized()
        return len(self._templates_df)

    def save_compiled_template(self, template_id: int, compiled_template: dict[str, Any]) -> None:
        """Save a compiled template to disk.

        Args:
            template_id: Template ID
            compiled_template: Compiled template data
        """
        try:
            # Ensure directory exists
            self.config.compiled_templates_path.mkdir(parents=True, exist_ok=True)

            template_file = self.config.compiled_templates_path / f"{template_id}.pkl"

            with open(template_file, "wb") as f:
                dill.dump(compiled_template, f)

        except Exception as e:
            self.logger.exception(f"Failed to save compiled template {template_id}: {str(e)}")
            raise TemplateError(template_id, f"Failed to save compiled template: {str(e)}") from e

    @property
    def templates(self) -> pd.DataFrame:
        """Return a snapshot of the templates DataFrame.

        A copy is returned so that callers cannot mutate the service's
        internal state; matches the behaviour of get_all_templates().
        """
        self.ensure_initialized()
        return self._templates_df.copy()
