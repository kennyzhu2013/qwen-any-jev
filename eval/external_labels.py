"""Helpers for parsing the external Chongqing ``CQ_Agent_res`` labels.

The external batch stores a compact string such as

    违规-低风险-贷款相关/不看征信
    涉诈-高风险-贷款相关-ab贷
    涉诈-高风险-验证码/短信转发诈骗

which mixes the 4.1 scene categories and the 4.2 fraud types in one field.
These helpers normalize that string and produce a *suggested* v2 label.  The
suggestion is a review aid only: on the v2 overlap the agent's 4.2 precision
is far below human review, so nothing here should be used as gold without a
manual decision.
"""
from __future__ import annotations

FRAUD_TYPES = {
    "个体工商户年报补录收费", "引导贷款用户添加第三方微信", "手机租赁套路贷诈骗",
    "套路运诈骗", "网贷平台退息退费", "证券投资类", "贷款相关-ab贷",
    "机票退、改签诈骗", "验证码/短信转发诈骗", "冒充快递中心理赔诈骗",
    "失物招领/车祸目击人敲诈诈骗", "店铺搭建诈骗", "拼多多退保诈骗",
    "电商平台刷单送礼物诈骗", "六合彩特码诈骗",
}
VIOLATION_SCENES = {
    "引流第三方平台", "贷款相关", "法律服务", "企业营销与招商服务",
    "商品推销", "商业地产", "违规催收", "其他",
}
RISK_ZH = {"高": "高风险", "中": "中风险", "低": "低风险"}


def parse_agent_result(value: str) -> dict:
    """Split ``CQ_Agent_res`` into kind / risk / scene / subtype."""
    value = (value or "").strip()
    if value == "正常":
        return {"kind": "正常", "risk": None, "scene": "正常", "subtype": None}
    parts = value.split("-", 2)
    if len(parts) < 3:
        return {"kind": "未知", "risk": None, "scene": value, "subtype": None}
    kind, risk, rest = parts
    risk = risk.replace("风险", "")
    if rest.startswith("验证码/"):
        scene, subtype = "验证码/短信转发诈骗", None
    elif "/" in rest:
        scene, subtype = rest.split("/", 1)
    else:
        scene, subtype = rest, None
    return {"kind": kind, "risk": risk, "scene": scene, "subtype": subtype}


def suggested_v2(parsed: dict) -> dict:
    """Map a parsed agent result to a suggested v2 label (review aid only)."""
    kind = parsed["kind"]
    is_fraud = 1 if kind == "涉诈" else 0
    is_violation = 0 if kind == "正常" else 1
    risk = "合规" if kind == "正常" else RISK_ZH.get(parsed["risk"] or "", "")
    if is_fraud:
        risk = "高风险"
    scene = parsed["scene"]
    scene_conflict = bool(is_fraud and scene in VIOLATION_SCENES)
    return {
        "is_fraud": is_fraud,
        "is_violation": is_violation,
        "risk_level": risk,
        "scene_category": scene,
        "scene_subtype": parsed["subtype"] or "",
        "scene_conflict": scene_conflict,
    }
