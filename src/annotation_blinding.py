import hashlib,json,re
from dataclasses import dataclass
from collections.abc import Mapping
from dataset_contract import DatasetContractError,sha256_digest
_DIGEST=re.compile(r"^sha256:[0-9a-f]{64}$"); _LOCALES={"en","zh-Hant","zh-Hans"}
@dataclass(frozen=True,slots=True)
class AnnotationPacket:
 itemId:str; locale:str; sourceDigest:str; rubricVersion:str; label:None; confidence:None; cannotAssessAllowed:bool; packetDigest:str
_FIELDS=frozenset(AnnotationPacket.__dataclass_fields__); FORBIDDEN={"arm","treatment","model","provider","prediction","reward","outcome","split"}
def packet_digest(raw):
 body={k:raw[k] for k in sorted(_FIELDS) if k!="packetDigest"}; return sha256_digest(json.dumps(body,separators=(",",":"),ensure_ascii=True).encode())
def validate_packet(raw:Mapping, *, source_bytes:bytes|None=None):
 if not isinstance(raw,Mapping) or set(raw)!=_FIELDS or set(raw)&FORBIDDEN: raise DatasetContractError("blind packet fields invalid")
 if raw["locale"] not in _LOCALES or not _DIGEST.fullmatch(raw["sourceDigest"]) or not _DIGEST.fullmatch(raw["packetDigest"]): raise DatasetContractError("packet locale/digest invalid")
 if source_bytes is None or sha256_digest(source_bytes)!=raw["sourceDigest"]: raise DatasetContractError("sourceDigest does not match source bytes")
 if raw["packetDigest"]!=packet_digest(raw): raise DatasetContractError("packetDigest mismatch")
 if raw["label"] is not None or raw["confidence"] is not None: raise DatasetContractError("human fields must be blank")
 if type(raw["cannotAssessAllowed"]) is not bool: raise DatasetContractError("cannotAssessAllowed must be boolean")
 return AnnotationPacket(**raw)
