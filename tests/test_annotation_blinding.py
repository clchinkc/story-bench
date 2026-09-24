import pytest
from annotation_blinding import *
from dataset_contract import DatasetContractError,sha256_digest
def packet(source=b"source",**k):
 x=dict(itemId="i",locale="en",sourceDigest=sha256_digest(source),rubricVersion="v1",label=None,confidence=None,cannotAssessAllowed=True,packetDigest=""); x["packetDigest"]=packet_digest(x); x.update(k); return x
def test_blank_packet_accepts(): assert validate_packet(packet(),source_bytes=b"source").label is None
@pytest.mark.parametrize("field",["arm","treatment","model","provider","prediction","reward","outcome","split"])
def test_treatment_leak_rejected(field):
 x=packet(); x[field]=1
 with pytest.raises(DatasetContractError): validate_packet(x,source_bytes=b"source")
def test_invalid_locale_source_and_packet_digest_rejected():
 for change in ({"locale":"fr"},{"sourceDigest":"not-a-digest"},{"packetDigest":"sha256:"+"0"*64}):
  x=packet(**change)
  with pytest.raises(DatasetContractError): validate_packet(x,source_bytes=b"source")
def test_prefilled_label_rejected():
 with pytest.raises(DatasetContractError): validate_packet(packet(label="resolved"),source_bytes=b"source")
