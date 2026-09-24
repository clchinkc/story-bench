import pytest
from dataset_contract import parse_item,DatasetContractError,sha256_digest
from split_leakage import *
def i(split="train",item_id="i"): return parse_item(dict(itemId=item_id,groupId="g",locale="en",contentDigest=sha256_digest(b"p"),rightsStatus="owned",rightsEvidenceRef="r",split=split,labelSchemaVersion="v1",payloadRef="ref:p"),payload_bytes=b"p")
def test_group_cannot_cross_splits():
 with pytest.raises(DatasetContractError): validate_group_splits([i("train"),i("holdout","j")])
def test_holdout_seal_and_forgery_rejected():
 xs=[i("holdout")]; r=seal_holdout(xs,manifest_seal(xs)); require_release(r,r["seal"])
 with pytest.raises(DatasetContractError): seal_holdout(xs,"sha256:"+"0"*64)
 with pytest.raises(DatasetContractError): require_release(r,"wrong")
