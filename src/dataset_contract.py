from dataclasses import dataclass
import hashlib,re
from typing import Any,Mapping
class DatasetContractError(ValueError): pass
_DIGEST=re.compile(r"^sha256:[0-9a-f]{64}$"); _REF=re.compile(r"^ref:[A-Za-z0-9._:-]+$")
@dataclass(frozen=True,slots=True)
class DatasetItemRef:
 itemId:str; groupId:str; locale:str; contentDigest:str; rightsStatus:str; rightsEvidenceRef:str; split:str; labelSchemaVersion:str; payloadRef:str
_FIELDS=frozenset(DatasetItemRef.__dataclass_fields__)
def sha256_digest(data:bytes)->str: return "sha256:"+hashlib.sha256(data).hexdigest()
def canonical_item_bytes(raw): return repr(tuple((k,raw[k]) for k in sorted(raw))).encode()
def parse_item(raw:Mapping[str,Any], *, payload_bytes:bytes|None=None)->DatasetItemRef:
 if not isinstance(raw,Mapping) or set(raw)!=_FIELDS: raise DatasetContractError("closed item fields required")
 if any(type(raw[k]) is not str or not raw[k].strip() for k in _FIELDS): raise DatasetContractError("all item fields must be nonblank strings")
 if raw["locale"] not in {"en","zh-Hant","zh-Hans"}: raise DatasetContractError("unknown locale")
 if raw["rightsStatus"] not in {"owned","licensed","explicit-permission"} or raw["rightsEvidenceRef"].startswith(("sk-","Bearer ")): raise DatasetContractError("rights metadata invalid")
 if raw["split"] not in {"train","dev","calibration","holdout"}: raise DatasetContractError("unknown split")
 if not _DIGEST.fullmatch(raw["contentDigest"]): raise DatasetContractError("invalid content digest")
 if not _REF.fullmatch(raw["payloadRef"]): raise DatasetContractError("payloadRef must be opaque and non-content")
 if payload_bytes is None: raise DatasetContractError("payload bytes required to bind contentDigest")
 if not isinstance(payload_bytes,bytes) or sha256_digest(payload_bytes)!=raw["contentDigest"]: raise DatasetContractError("contentDigest does not match payload bytes")
 return DatasetItemRef(**raw)
def validate_manifest(items,payloads:Mapping[str,bytes]):
 parsed=tuple(parse_item(x,payload_bytes=payloads.get(x.get("payloadRef"))) for x in items); ids=[x.itemId for x in parsed]
 if len(ids)!=len(set(ids)): raise DatasetContractError("duplicate itemId")
 return parsed
