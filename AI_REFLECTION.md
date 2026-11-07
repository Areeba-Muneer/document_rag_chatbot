# AI Usage Reflection

## Understand
I understood the RAG workflow as two stages. First, the system processes documents, splits their text into chunks, creates embeddings and stores the vectors with source metadata. Second, it embeds a user question, retrieves the three most relevant chunks and provides those chunks to an LLM as context.

## Verify
I checked whether the retrieved passages came from the selected documents and whether the generated answer was supported by those passages. I used direct questions, a question requiring multiple document sections, and a question whose answer was not included. The testing table in README should be completed with actual application outputs.

## Adapt
I used source metadata to show which file and chunk supported an answer. I also added a missing-information response, environment-based configuration, upload support, and basic exception handling. The similarity threshold and chunk settings can be adjusted after evaluating retrieval quality.
