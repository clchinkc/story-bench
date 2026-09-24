import pytest
from dataset_contract import *
def payload(): return b"authorized payload"
def item(**k):
 x=dict(itemId="i",groupId="g",locale="en",contentDigest=sha256_digest(payload()),rightsStatus="owned",rightsEvidenceRef="rights:r",split="train",labelSchemaVersion="v1",payloadRef="ref:p"); x.update(k); return x
def test_accept_closed_item(): assert parse_item(item(),payload_bytes=payload()).itemId=="i"
@pytest.mark.parametrize("k",[{"rightsStatus":"unknown"},{"rightsEvidenceRef":""},{"contentDigest":"sha256:"+"0"*64},{"locale":"fr"},{"split":"bad"},{"rightsEvidenceRef":"sk-secret"},{"payloadRef":"THE PRIVATE STORY TEXT"}])
def test_rejects_invalid_metadata(k):
 with pytest.raises(DatasetContractError): parse_item(item(**k),payload_bytes=payload())
def test_payload_required_and_bound():
 with pytest.raises(DatasetContractError): parse_item(item())
 with pytest.raises(DatasetContractError): parse_item(item(),payload_bytes=b"different")
def test_rejects_duplicate():
 with pytest.raises(DatasetContractError): validate_manifest([item(),item()],{"ref:p":payload()})
