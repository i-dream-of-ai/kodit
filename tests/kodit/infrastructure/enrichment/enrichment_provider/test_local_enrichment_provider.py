"""Tests for the local enrichment provider."""

import pytest

from kodit.domain.value_objects import EnrichmentRequest
from kodit.infrastructure.enrichment.local_enrichment_provider import (
    LocalEnrichmentProvider,
)


class TestLocalEnrichmentProvider:
    """Test the local enrichment provider."""

    def test_init_default_values(self) -> None:
        """Test initialization with default values."""
        provider = LocalEnrichmentProvider()
        assert provider.model_name == "Qwen/Qwen3-0.6B"
        assert provider.context_window == 2048
        assert provider.model is None
        assert provider.tokenizer is None

    def test_init_custom_values(self) -> None:
        """Test initialization with custom values."""
        provider = LocalEnrichmentProvider(model_name="test-model", context_window=1024)
        assert provider.model_name == "test-model"
        assert provider.context_window == 1024

    @pytest.mark.asyncio
    async def test_enrich_empty_requests(self) -> None:
        """Test enrichment with empty requests."""
        provider = LocalEnrichmentProvider()
        requests = []

        results = [result async for result in provider.enrich(requests)]

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_enrich_empty_text_requests(self) -> None:
        """Test enrichment with requests containing empty text."""
        provider = LocalEnrichmentProvider()
        requests = [
            EnrichmentRequest(snippet_id=1, text=""),
            EnrichmentRequest(snippet_id=2, text="   "),
        ]

        results = [result async for result in provider.enrich(requests)]

        # The local provider actually processes whitespace-only text
        # So we expect 1 result for the whitespace-only request
        assert len(results) == 1
        assert results[0].snippet_id == 2

    # Note: Complex integration tests for the full enrichment pipeline would require
    # actual model loading and are complex to mock properly. These could be added
    # in a separate test suite that has access to real models.
