"""Recover the original Phase IV runs after an artifact adapter correction."""

import hashlib

from .config import digest
from .results import ResultStore


# Keyed by the corrected module's normalized content SHA, so later science
# edits cannot silently inherit the old run's checkpoints.
COMPATIBLE = {
    "d6f44b986ac9c93efb36f69ceaecb02c99ed8435f55691aa3cee13166fead695":
        "0d7350ecaa6d581bbd3a49ae77b49014847eb7ae7dafc502a543e0365b48634a",
    "e5d9bc2a46e8121d2d74c4f753fb344ee541587133b118739ec70071a3f72a44":
        "0e015d7bb35605e122d5b8707aee0abe1bb987641d5d9f19bbd926576289baf7",
}


def compatible_store(root, context, source):
    content = source.read_bytes().replace(b"\r\n", b"\n")
    old_sha = COMPATIBLE.get(hashlib.sha256(content).hexdigest())
    if old_sha:
        if "code" in context:
            previous = context | {"code": context["code"] | {source.name: old_sha}}
        else:
            previous = context | {"code_sha256": old_sha}
        if (root / f"phase3-{digest(previous)[:20]}" / "context.json").is_file():
            context = previous
    return ResultStore(root, context, resume=True)

