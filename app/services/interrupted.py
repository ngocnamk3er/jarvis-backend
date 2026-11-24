"""The markers kept in an answer the user stopped part-way (see chat_service.save_interrupted)."""

# Read by the model on the next turn, so it knows the answer was cut off rather than what it meant to say.
# conversation_service.serialize_messages strips them again when the message is shown on the page.
INTERRUPTED_NOTE = "\n\n[The user stopped this response here. It is incomplete.]"

# When the user stopped it while the model was still thinking: there is no text to put it after, and a message
# with no content at all is refused by some providers, so the marker stands alone.
INTERRUPTED_NO_TEXT = "[The user stopped this response before it wrote any text.]"
