import hashlib,json
from collections.abc import Iterable,Mapping
from dataset_contract import DatasetContractError,DatasetItemRef
def validate_group_splits(items:Iterable[DatasetItemRef]):
 out={}
 for x in items:
  if x.groupId in out and out[x.groupId]!=x.split: raise DatasetContractError("group crosses splits")
  out[x.groupId]=x.split
 return out
def manifest_seal(items):
 body=[(x.itemId,x.groupId,x.locale,x.contentDigest,x.split) for x in sorted(items,key=lambda x:x.itemId)]
 return "sha256:"+hashlib.sha256(json.dumps(body,separators=(",",":"),ensure_ascii=True).encode()).hexdigest()
def seal_holdout(items,seal):
 items=tuple(items); expected=manifest_seal(items)
 if not any(x.split=="holdout" for x in items) or seal!=expected: raise DatasetContractError("holdout seal mismatch")
 return {"seal":expected,"holdout_count":sum(x.split=="holdout" for x in items)}
def require_release(record:Mapping,token):
 if token!=record.get("seal"): raise DatasetContractError("holdout release refused")
