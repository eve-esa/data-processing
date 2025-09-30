import aiohttp
import asyncio
from typing import List, Optional
from eve.model.document import Document
from eve.base_step import PipelineStep


class PiiStep(PipelineStep):

    async def _remove_pii(
        self,
        document: Document,
        entities: Optional[List[str]] = None,
        threshold: float = 0.35,
        return_analysis: bool = False,
        url: str = None,
    ) -> Document:
        """Make a call to the litserve API and remove PII (async with aiohttp)."""

        if not url:
            self.logger.error("No URL provided for PII service")
            return document

        if entities is None:
            entities = ["PERSON", "EMAIL_ADDRESS"]

        payload = {
            "text": document.content,
            "entities": entities,
            "score_threshold": threshold,
            "return_analysis": return_analysis,
        }

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
                async with session.post(
                    url,
                    json = payload,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    self.logger.debug(f"PII API result: {result}")
                    document.content = result.get("anonymized_text", document.content)
                    return document

        except aiohttp.ClientError as e:
            self.logger.error(f"PII API request failed: {e}")
            return document

    async def remove_pii(
        self,
        document: Document,
        entities: Optional[List[str]] = None,
        threshold: float = 0.35,
        return_analysis: bool = False,
        url: str = None,
    ) -> Document:
        return await self._remove_pii(document, entities, threshold, return_analysis, url)

    async def execute(self, documents: List[Document]) -> List[Document]:
        base_url = self.config.get("url")
        if not base_url:
            self.logger.error("No URL provided for PII service")
            return []

        url = f"{base_url}/predict"

        # Run all requests concurrently
        tasks = [self.remove_pii(document, url=url) for document in documents]
        results = await asyncio.gather(*tasks, return_exceptions = True)

        final = []
        for doc in results:
            if doc and getattr(doc, "content_length", 0) > 1:
                final.append(doc)
                self.logger.info(f"Successfully anonymized {doc.filename}")

        return final
