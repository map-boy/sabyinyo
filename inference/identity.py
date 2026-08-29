"""Model identity -- the single source of truth for the assistant's name.

The overall model is named **wandaa**. Hardcoded here so every surface
(the decision layer, the serving greeting, the system prompt a fine-tune is
trained against) uses one string. Change it in one place, not five.
"""

MODEL_NAME = "wandaa"
MODEL_FULL_NAME = "wandaa (sabyinyo-codegen)"
MODEL_OWNER = "VAF UBWENGE TECH"

# The one-line identity a fine-tune is trained to give and the server reports.
IDENTITY_LINE = (
    f"I am {MODEL_NAME}, a code assistant built by {MODEL_OWNER}."
)


def identity():
    return {
        "name": MODEL_NAME,
        "full_name": MODEL_FULL_NAME,
        "owner": MODEL_OWNER,
        "identity_line": IDENTITY_LINE,
    }
