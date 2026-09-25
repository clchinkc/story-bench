import pytest
from annotation_blinding import *
from dataset_contract import DatasetContractError, sha256_digest


def packet(source=b"source", **k):
    x = dict(itemId="i", locale="en", sourceDigest=sha256_digest(source),
             rubricVersion="v1", label=None, confidence=None, cannotAssessAllowed=True)
    x.update(k)
    # Recompute packetDigest AFTER overrides so a non-digest override does not trip
    # the generic packetDigest-mismatch guard before the specific guard it targets.
    # (The previous helper computed the digest first, then applied overrides, so every
    # override raised "packetDigest mismatch" and the label/locale/sourceBytes guards
    # were never exercised.) An explicit packetDigest is left exactly as supplied.
    if "packetDigest" not in k:
        x["packetDigest"] = packet_digest(x)
    return x


def test_blank_packet_accepts():
    assert validate_packet(packet(), source_bytes=b"source").label is None


@pytest.mark.parametrize("field", ["arm", "treatment", "model", "provider",
                                   "prediction", "reward", "outcome", "split"])
def test_treatment_leak_rejected(field):
    x = packet()
    x[field] = 1
    with pytest.raises(DatasetContractError):
        validate_packet(x, source_bytes=b"source")


def test_prefilled_label_rejected():
    """A prefilled label must be caught by the human-fields-blank guard itself."""
    with pytest.raises(DatasetContractError, match="human fields must be blank"):
        validate_packet(packet(label="resolved"), source_bytes=b"source")


def test_prefilled_confidence_rejected():
    with pytest.raises(DatasetContractError, match="human fields must be blank"):
        validate_packet(packet(confidence=0.9), source_bytes=b"source")


def test_invalid_locale_rejected():
    """An out-of-whitelist locale must be caught by the locale guard, not a digest guard."""
    with pytest.raises(DatasetContractError, match="locale"):
        validate_packet(packet(locale="fr"), source_bytes=b"source")


def test_source_digest_must_match_source_bytes():
    """A well-formed digest of DIFFERENT bytes must fail the sourceBytes binding.

    This is the guard that ties the packet to the actual source; a malformed-digest
    case cannot isolate it because the format check rejects those first.
    """
    other = packet(sourceDigest=sha256_digest(b"other"))
    with pytest.raises(DatasetContractError, match="sourceDigest does not match source bytes"):
        validate_packet(other, source_bytes=b"source")


def test_packet_digest_mismatch_rejected():
    with pytest.raises(DatasetContractError, match="packetDigest mismatch"):
        validate_packet(packet(packetDigest="sha256:" + "0" * 64), source_bytes=b"source")


def test_malformed_source_digest_rejected():
    with pytest.raises(DatasetContractError):
        validate_packet(packet(sourceDigest="not-a-digest"), source_bytes=b"source")
