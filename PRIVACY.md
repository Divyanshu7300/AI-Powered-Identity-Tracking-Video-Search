# Privacy and Retention Policy

This project processes uploaded video and may generate person crops, evidence frames, clips, Re-ID vectors, face embeddings, captions, and search metadata. Operators must obtain the required notice, consent, and legal authorization before processing people.

Generated data is stored in the authenticated session namespace under `data/users/<session-id>/`. Operators should define a retention period, delete data when it expires or when a valid deletion request is received, and protect backups and logs with the same controls. The reset action deletes the current authenticated namespace only.

When Groq RAG is enabled, the text query and retrieved evidence metadata are sent to Groq for answer generation. Operators must disclose that transfer, review the applicable provider terms and privacy policy, and disable `GROQ_API_KEY` or `use_llm` when external processing is not authorized.
