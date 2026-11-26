#!/usr/bin/env python
"""Relabel the 0729 human-reviewed data to the 2026-11 review taxonomy.

The old CSV has a coarse binary label (`正常` / `涉诈`) and a category.  The new
standard separates:

  is_violation : any 4.1 violation or 4.2 fraud
  is_fraud     : 4.2 fraud scenarios only
  risk_level   : 合规 / 低风险 / 中风险 / 高风险 / 升级涉诈
  scene_category / scene_subtype / label_detail

This script is deliberately conservative:

* normal rows keep the human label (`合规`);
* rows whose old category is a 4.2 fraud type are `is_fraud=1`, `高风险`;
* rows whose old category is a 4.1 violation get a rule-based risk level;
* a 4.1 row is upgraded to 4.2 only on a strong, explicit pattern
  (loan + guide user to add third-party WeChat, Zhima score rental loan, ...);
* rows where the risk level cannot be established are marked `needs_review=1`
  and default to `低风险` (the least severe level consistent with being a
  violation).  They are exported to a review queue.

The old columns are preserved.  `label_v2` is the new binary target:
`涉诈` iff `is_fraud=1`, otherwise `非涉诈` (normal + non-fraud violations).

Usage:
  python eval/relabel.py
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import re
from pathlib import Path

# 4.1 violation categories (new taxonomy).
V_LEGAL = "法律服务"
V_ENTERPRISE = "企业营销与招商服务"
V_GOODS = "商品推销"
V_ESTATE = "商业地产"
V_COLLECTION = "违规催收"
V_OTHER = "其他"
V_LOAN = "贷款相关"
V_TRAFFIC = "引流第三方平台"

# 4.2 fraud categories (new taxonomy); all are high risk.
FRAUD_CATEGORIES = {
    "机票退、改签诈骗",
    "手机租赁套路贷诈骗",
    "证券投资类",
    "网贷平台退息退费",
    "个体工商户年报补录收费",
    "贷款相关-ab贷",
    "引导贷款用户添加第三方微信",
    "套路运诈骗",
    "抖音会员关闭诈骗",
    "电商平台刷单送礼物诈骗",
    "套路租房诈骗",
    "六合彩特码诈骗",
    "货运送现金诈骗",
    "创业导师诈骗",
    "拼多多退保诈骗",
    "聊天代付外卖诈骗",
    "失物招领/车祸目击人敲诈诈骗",
    "抖音/快手网红奖金冒领诈骗",
    "店铺搭建诈骗",
    "扶贫救助诈骗",
    "开箱密码诈骗",
    "冒充快递中心理赔诈骗",
}

# Old category -> (new category, is_fraud, default risk, subtype).
# Only deterministic mappings live here; ambiguous risk levels are rule-based.
CATEGORY_MAP = {
    "贷款相关-ab贷": ("贷款相关-ab贷", 1, "高风险", "ab贷"),
    "个体工商户年报补录收费": ("个体工商户年报补录收费", 1, "高风险", "年报补录收费"),
    "引导贷款用户添加第三方微信": ("引导贷款用户添加第三方微信", 1, "高风险", "引导添加第三方微信"),
    "证券投资类": ("证券投资类", 1, "高风险", "证券投资荐股"),
    "网贷平台退息退费": ("网贷平台退息退费", 1, "高风险", "网贷退息退费"),
    "手机租赁套路贷诈骗": ("手机租赁套路贷", 1, "高风险", "手机租赁套现"),
    "套路运诈骗": ("套路运", 1, "高风险", "套路运"),
    "贷款相关": (V_LOAN, 0, None, ""),
    "违规催收": (V_COLLECTION, 0, None, ""),
    "引流第三方平台": (V_TRAFFIC, 0, "高风险", "引流第三方平台"),
    "企业营销与招商服务": (V_ENTERPRISE, 0, None, ""),
    "法律服务": (V_LEGAL, 0, None, ""),
    "商业地产": (V_ESTATE, 0, None, ""),
    "商品推销": (V_GOODS, 0, None, ""),
    "其他": (V_OTHER, 0, "高风险", "其他异常风险"),
}

# Rule order matters: first match wins.  Each rule is (name, pattern, risk, subtype).
# `risk=None` means "do not assign a level; fall through".
LOAN_RULES = [
    # 4.2.7 requires a third-party role target (经理/客服/专员/...), not the
    # outbound's own WeChat.  "微信搜索平台客服公众号" is 4.1.2 high risk.
    ("loan_guide_add_third", re.compile(
        r"(?:添加|加|搜索|搜|扫|关注|通过).{0,8}"
        r"(?:经理|专员|主任|老师|业务员|审核|第三方).{0,4}(?:微信|薇信|企微|企业微信)"
        r"|客服.{0,2}(?:微信|薇信|企微|企业微信)"
        r"|(?:添加|加).{0,4}(?:企业微信|企微)"
        r"|(?:微信|薇信).{0,4}(?:搜索|添加|加).{0,6}(?:经理|专员|主任|老师|业务员|审核|第三方)"
        r"|(?:添加|加).{0,6}(?:放款|下款|审核|客户|业务)经理.{0,4}(?:微信|企微|企业微信)"),
     "升级涉诈", "引导添加第三方微信"),
    ("loan_zhima_rental", re.compile(
        r"芝麻(?:信用)?分|租(?:手机|机)|手机租赁|租机.{0,8}(?:套现|回收|变现)"
        r"|套现.{0,8}租机"),
     "升级涉诈", "芝麻分/租机套现"),
    ("loan_debt_opt", re.compile(
        r"债务(?:优化|重组|减免|托管)|垫资|过桥|协商(?:债务|减免)|债务(?:本金|利息)减免"),
     "高风险", "债务优化"),
    ("loan_cut_interest", re.compile(
        r"砍头息|(?:服务费|手续费|担保费|包装费|资料费).{0,8}(?:1[0-9]|[2-9][0-9])\s*(?:个点|%|％)"
        r"|(?:1[0-9]|[2-9][0-9])\s*(?:个点|%|％).{0,6}(?:服务费|手续费|担保费|包装费|资料费)"
        r"|18\.5%|18个点"
        r"|(?:十一|十二|十三|十四|十五|十六|十七|十八|十九|二十|三十)\s*个点"
        r"|十[到至]二十\s*个点|十几\s*个点|十多\s*个点"),
     "高风险", "砍头息/高额费用"),
    ("loan_gold", re.compile(r"黄金贷|黄金分期"), "高风险", "黄金贷"),
    ("loan_high_channel", re.compile(
        r"境外网贷|境外高炮|高炮|不用还|日息|千分之五|年化.{0,8}(?:2[4-9]|[3-9][0-9])%"
        r"|高息贷款"),
     "高风险", "高风险渠道"),
    # Platform guidance to an app / service account for withdrawal or account
    # operation is 4.1.2 high risk, not 4.2.7.
    ("loan_platform_guide", re.compile(
        r"(?:微信|薇信).{0,6}(?:搜索|关注|打开).{0,8}(?:客服|公众号|服务号|小程序)"
        r"|(?:打开|登录|进入).{0,6}(?:app|APP|小程序|公众号|服务号).{0,16}"
        r"(?:额度|提现|操作|借款|放款|登录)"),
     "高风险", "引导平台操作"),
    ("loan_user_adds_wechat", re.compile(
        r"(?:您|你|客户).{0,6}(?:添加|加|扫|关注|通过).{0,8}(?:微信|薇信|企微|企业微信)"),
     "高风险", "引导用户添加微信"),
    ("loan_withdraw", re.compile(
        r"(?:提现|套现).{0,12}(?:app|APP|小程序|公众号|服务号)"
        r"|(?:app|APP|小程序|公众号|服务号).{0,12}(?:提现|套现)"),
     "高风险", "引导平台操作提现"),
    ("loan_medium", re.compile(
        r"提取公积金|公积金.{0,6}提取|提(?:取)?公积金|修复征信|征信修复|消除征信|征信洗白"
        r"|清除数据|消除数据"),
     "中风险", "公积金提取/征信修复"),
    ("loan_low", re.compile(
        r"不看征信|不上征信|不查征信|不看查询|不查查询|不看负债|不查负债|不看逾期|黑户|白户"
        r"|只看流水|只看收入|大数据|包过|包批|包下|100%下款|百分百下款|内部渠道|零利息|免息"
        r"|定金|诚意金|保证金|解冻金|前期费用|下款前|先交|先付|私人借款|私人资金|保险贷"
        r"|度小满|安逸花|拍拍贷|分期乐|理想花|花呗|好分期|还呗|众安贷|信用飞|三六零借条|360借条"
        r"|京东金融|闪小融|云闪付|有钱花|宜享花|优客花|小花钱包|微钱包|微信钱包|抖音放心借"
        r"|美团|借呗|微粒贷|招联|中信消费金融|马上金融|平安普惠|宜人贷|人人贷|你我贷|小赢"
        r"|洋钱罐|省呗|国美易卡|桔多多|豆豆钱|时光分期|召集令|融360"),
     "低风险", "不看征信/非合规渠道"),
]

COLLECTION_RULES = [
    ("collection_impersonate", re.compile(
        r"(?:冒充|自称|我(?:们)?是|我这边是|这边是|这里是).{0,12}"
        r"(?:法院|人民法院|公安|公安局|检察院|警察|司法|调解中心|诉调|诉讼中心"
        r"|诉讼服务中心|案件受理中心|执行局|政府)"
        r"|(?:法院|公安|检察院|司法调解|诉调站|诉讼中心).{0,12}(?:工作人员|通知|要求|传票)"),
     "高风险", "冒充公检法催收"),
    # Court / legal-process props without an explicit "冒充/自称" marker.
    # 诉前告知书/档案袋/法警/执行局/案件受理中心 are the signature props of
    # 诉前通知-style impersonation; genuine lawyer calls usually say 律师/律所
    # and 起诉 without them, so these rows go to review as well.
    ("collection_legal_process", re.compile(
        r"诉前告知书|档案袋|财产保全|法警|执行局|案件受理中心"
        r"|[诉塑数四苏][条调]站|诉前(?:通知|调查)"
        r"|(?:人民法院|法院).{0,8}(?:调解中心|诉调|诉讼服务中心|立案庭|执行局|法警队)"
        r"|(?:人民法院|法院|公安|检察院|司法调解|调解中心|诉调|诉讼中心)"
        r".{0,40}(?:通知(?:你|您)|要求(?:你|您)|传票|到现场|配合|工作人员|调解员)"),
     "高风险", "冒充公检法催收"),
    ("collection_refinance", re.compile(
        r"还借宝|购买超市卡|二次借款"
        r"|解除逾期.{0,8}(?:额度|借款)"
        r"|(?:可以|建议|帮你|给你|不如|要不|支持).{0,8}(?:以贷养贷|借新还旧)"),
     "高风险", "诱导以贷养贷"),
    ("collection_low", re.compile(
        r"通讯录|亲友|同事|家人|父母|配偶|第三方|上门|村委会|社区|单位"
        r"|律师|律所|诉讼|起诉|立案|传票|威胁|施压|冻结"),
     "低风险", "威胁施压/律师催收"),
]

TRAFFIC_RULES = [
    ("traffic_guide_add_third", re.compile(
        r"(?:添加|加|搜索|搜|扫|关注|通过).{0,8}"
        r"(?:经理|专员|主任|老师|业务员|审核|第三方).{0,4}(?:微信|薇信|企微|企业微信)"
        r"|客服.{0,2}(?:微信|薇信|企微|企业微信)"
        r"|(?:添加|加).{0,4}(?:企业微信|企微)"),
     "升级涉诈", "引导添加第三方微信"),
    ("traffic_user_adds", re.compile(
        r"(?:您|你|客户).{0,6}(?:添加|加|扫|关注|通过).{0,8}(?:微信|薇信|企微|企业微信)"
        r"|加我微信|加一下我|加我呗"),
     "高风险", "引导用户添加微信"),
    ("traffic_high_channel", re.compile(
        r"QQ|YY|微信群|福利群|qq群|群聊|下载|app|APP|链接|二维码|扫码|公众号|服务号|小程序"),
     "高风险", "非个人微信渠道"),
    ("traffic_third_party_adds", re.compile(
        r"(?:第三方|业务员|客户经理|经理|客服).{0,8}(?:加|添加).{0,6}(?:您|你).{0,4}(?:微信|好友)"),
     "低风险", "第三方添加微信"),
    ("traffic_outbound_adds", re.compile(
        r"(?:我|我们|这边).{0,6}(?:加|添加).{0,6}(?:您|你).{0,4}(?:微信|好友)"),
     "合规", "外呼方主动添加微信"),
]

ENTERPRISE_RULES = [
    ("enterprise_high_training", re.compile(
        r"(?:教培|考证|培训).{0,20}(?:大额|长周期|预收|个人账户|培训贷|0首付|先上课|包过|包拿证)"
        r"|(?:大额|长周期|预收|个人账户|培训贷|0首付|先上课|包过|包拿证).{0,20}(?:教培|考证|培训)"),
     "高风险", "教培/考证违规收费"),
    ("enterprise_medium_tax", re.compile(
        r"虚开|成本票|核定征收|重点人群|补贴申报|涉税筹划"),
     "中风险", "违规涉税/补贴申报"),
    ("enterprise_low", re.compile(
        r"推广获客|曝光|会展|招生|申报代办|提前收费|保过|包拿证|资质代办|软过|办证"),
     "低风险", "企业服务违规收费"),
]

LEGAL_RULES = [
    ("legal_high_refund", re.compile(
        r"维权机构.{0,12}(?:退|追回).{0,8}(?:律所|律师)费|帮退律所|退回律所"
        r"|退款到账后.{0,8}(?:分成|收费)|成功后.{0,6}(?:收取|分成)"),
     "高风险", "维权机构退律所费"),
    ("legal_low_fee", re.compile(
        r"提前收费|手续费|保证金|调档费|追回|维权|悬赏公告|起诉|诉讼服务"),
     "低风险", "法律服务提前收费"),
]

ESTATE_RULES = [
    # 商铺包租提前收推广费 is explicitly 合规 (spec 商业地产), so bare
    # 包租/长期出租 must not trigger.  Only the 代运营+装修 pattern is 中风险.
    ("estate_medium_renovation", re.compile(
        r"代运营|运营包装|长期(?:合作)?代运营|委托.{0,6}装修"
        r"|(?:装修|运营).{0,12}(?:收费|费用|服务费)"
        r"|(?:长期出租|包租).{0,20}(?:装修|运营|代管)"),
     "中风险", "代客运营/装修收费"),
    ("estate_low_auction", re.compile(r"法拍房|拍卖房"), "低风险", "法拍房处理"),
]

GOODS_RULES = [
    ("goods_medium_health", re.compile(
        r"保健品|高额到付|到付|免费赠送.{0,8}(?:酒|礼品).{0,8}(?:99|运费)"),
     "中风险", "保健品/高额到付"),
    ("goods_low_pos", re.compile(r"POS|pos|刷卡机|刷流水"), "低风险", "POS机刷流水"),
]

RISK_RULES = {
    V_LOAN: LOAN_RULES,
    V_COLLECTION: COLLECTION_RULES,
    V_TRAFFIC: TRAFFIC_RULES,
    V_ENTERPRISE: ENTERPRISE_RULES,
    V_LEGAL: LEGAL_RULES,
    V_ESTATE: ESTATE_RULES,
    V_GOODS: GOODS_RULES,
    V_OTHER: [],
}

RISK_ORDER = {"合规": 0, "低风险": 1, "中风险": 2, "高风险": 3, "升级涉诈": 4}

# Old label=正常 rows are not re-labelled silently.  These strict patterns
# only flag them for human review when the new rubric finds a concrete
# violation signal that a 正常 label would miss.
LOAN_CONTEXT = re.compile(
    r"贷款|借款|放款|下款|额度|征信|利息|还款|资金|网贷|分期|助贷"
    r"|京东金融|度小满|安逸花|分期乐|花呗|借呗|微粒贷")

NORMAL_AUDIT_RULES = [
    ("loan_third_party_wechat", re.compile(
        r"(?:客户经理|业务员|下款经理|放款经理|审核老师|专员|主任).{0,6}"
        r"(?:微信|薇信|企微|企业微信)"
        r"|(?:添加|加|搜索|扫|通过).{0,6}(?:客户经理|业务员|经理|专员|主任)"
        r".{0,4}(?:微信|薇信|企微|企业微信)"),
     "升级涉诈", "贷款相关", "引导添加第三方微信"),
    ("loan_debt_opt", re.compile(
        r"债务优化|债务重组|债务减免|债务托管|债务置换|代务置换|垫资|过桥"
        r"|(?:结清|还清).{0,8}(?:逾期|债务).{0,8}(?:再|后).{0,8}(?:贷|借)"
        r"|债务.{0,8}(?:结清|还清).{0,12}(?:再|后).{0,8}(?:贷|借)"),
     "高风险", "贷款相关", "债务优化"),
    ("loan_cut_interest", re.compile(
        r"百分之(?:十[一二三四五六七八九]|二十|三十)"
        r"|(?:十一|十二|十三|十四|十五|十六|十七|十八|十九|二十|三十)\s*个点"
        r"|十[到至]二十\s*个点|十几\s*个点|十多\s*个点"),
     "高风险", "贷款相关", "砍头息/高额费用"),
    ("loan_user_adds_wechat", re.compile(
        r"(?:你|您)(?:直接|先|可以|这边|那边|方便)?(?:加|添加)"
        r"(?:一下|下|一个|个)?(?:我|我的)?(?:的)?(?:微信|薇信|企微|企业微信)"
        r"|(?:你|您)(?:通过|扫)(?:一下|下)?(?:我|我的)(?:的)?(?:微信|二维码)"
        r"|加我(?:的)?微信|加一下我"),
     "高风险", "贷款相关", "引导用户添加微信"),
    ("securities", re.compile(
        r"(?:股票|炒股|证券|交易所|基金|白银|黄金|荐股|K线|大盘|行情|投资平台)"
        r".{0,20}(?:下载|签约|老师|群|平台|产品|开户|操作)"),
     "升级涉诈", "证券投资类", "证券投资"),
    ("lawyer_collection", re.compile(
        r"(?:律师事务所|律所).{0,40}(?:调解|催收|起诉|案件)"),
     "低风险", "违规催收", "威胁施压/律师催收"),
    ("collection_impersonate", re.compile(
        r"诉前告知书|档案袋|财产保全|法警|执行局|案件受理中心"
        r"|[诉塑数四苏][条调]站|诉前(?:通知|调查)"
        r"|(?:冒充|自称|我(?:们)?是|我这边是|这边是|这里是).{0,12}"
        r"(?:法院|人民法院|公安|公安局|检察院|警察|司法|调解中心|诉调|诉讼中心"
        r"|诉讼服务中心|执行局|政府)"
        r"|留意.{0,10}(?:法院|人民法院).{0,10}通知"
        r"|app.{0,4}司法局"),
     "高风险", "违规催收", "冒充公检法催收"),
    ("collection_corporate_transfer", re.compile(
        r"(?:转|打款|还款).{0,6}对公|对公.{0,6}(?:转账|入账|还款)"),
     "高风险", "违规催收", "要求对公转账/解冻"),
]


def audit_normal(text: str) -> tuple[str, str, str, str] | None:
    """Return (risk, scene, subtype, rule) for an old-正常 row, if any."""
    for name, pat, risk, scene, subtype in NORMAL_AUDIT_RULES:
        if scene == "贷款相关" and not LOAN_CONTEXT.search(text):
            continue
        if pat.search(text):
            return risk, scene, subtype, name
    return None


def note_risk(text: str) -> str | None:
    """Explicit risk level written by the human reviewer, if any."""
    for lv in ("升级涉诈", "高风险", "中风险", "低风险"):
        if lv in text:
            return lv
    return None


def apply_rules(category: str, text: str) -> tuple[str | None, str, str]:
    """Return (risk, subtype, rule_name) for a 4.1 category."""
    for name, pat, risk, subtype in RISK_RULES.get(category, []):
        if pat.search(text):
            return risk, subtype, name
    return None, "", ""


def relabel_row(row: dict) -> dict:
    old_label = row["label"].strip()
    old_cat = row["category"].strip()
    text = re.sub(r"\s+", " ", " ".join([row["content"], row["人工复核"], row["备注"]]))

    out = {
        "data_id": row["data_id"],
        "content": row["content"],
        "old_label": old_label,
        "old_category": old_cat,
        "is_violation": 0,
        "is_fraud": 0,
        "risk_level": "合规",
        "scene_category": "正常",
        "scene_subtype": "",
        "label_detail": "正常合规，无违规场景",
        "label_source": "old_label",
        "needs_review": 0,
        "proposed_risk_level": "",
        "proposed_scene_category": "",
        "proposed_subtype": "",
        "proposed_is_fraud": 0,
        "人工复核": row["人工复核"],
        "备注": row["备注"],
    }

    if old_label == "正常":
        out["label_v2"] = "非涉诈"
        audit = audit_normal(text)
        if audit is not None:
            risk, scene, subtype, rule = audit
            out.update(label_source="normal_audit", needs_review=1,
                       proposed_risk_level=risk,
                       proposed_scene_category=scene,
                       proposed_subtype=subtype,
                       proposed_is_fraud=1 if risk == "升级涉诈" else 0,
                       label_detail=f"旧标签正常，但命中{risk}信号：{subtype}，待复核")
        return out

    out["is_violation"] = 1
    out["label_v2"] = "非涉诈"
    mapped = CATEGORY_MAP.get(old_cat)
    if mapped is None:
        # Unknown old category: keep as violation, require review.
        out.update(risk_level="低风险", scene_category=old_cat or V_OTHER,
                   scene_subtype="", label_source="unknown_category",
                   needs_review=1, label_detail=f"{old_cat or V_OTHER}：旧类型未映射，待复核")
        return out

    scene, is_fraud, default_risk, default_subtype = mapped
    out["scene_category"] = scene

    if is_fraud:
        out.update(is_fraud=1, risk_level="高风险",
                   scene_subtype=default_subtype,
                   label_source="old_category",
                   label_detail=f"{scene}：{default_subtype}")
        out["label_v2"] = "涉诈"
        return out

    # 4.1 violation: explicit human risk note wins.
    nr = note_risk(row["人工复核"] + " " + row["备注"])
    risk, subtype, rule = apply_rules(scene, text)
    if nr is not None:
        out.update(risk_level=nr, scene_subtype=subtype or default_subtype,
                   label_source="manual_note",
                   label_detail=f"{scene}：{subtype or row['人工复核'] or row['备注']}")
    elif risk is not None:
        if risk == "合规":
            # The old human label says violation, but the new rule says the
            # behavior is compliant.  Do not silently flip the human label;
            # keep the minimum violation level and put the row in review.
            out.update(risk_level="低风险", scene_subtype=subtype,
                       label_source="rule_conflict",
                       label_detail=(f"{scene}：规则判断为{subtype}（合规），"
                                     f"但旧人工标签为涉诈，待复核"),
                       needs_review=1)
            return out
        out.update(risk_level=risk, scene_subtype=subtype,
                   label_source=f"rule:{rule}",
                   label_detail=f"{scene}：{subtype}")
        if rule == "collection_legal_process" or (
                rule == "collection_impersonate" and re.search(r"调解员|调解平台", text)):
            out["needs_review"] = 1
        if risk == "升级涉诈":
            out.update(is_fraud=1, label_v2="涉诈", scene_category=scene,
                       label_detail=f"{scene}：{subtype}（由4.1升级）")
    else:
        fallback = default_risk or "低风险"
        out.update(risk_level=fallback, scene_subtype=default_subtype,
                   label_source="default_low" if fallback == "低风险" else "default_category",
                   label_detail=f"{scene}：待复核（旧标签仅到类型，未识别到明确风险话术）",
                   needs_review=1 if fallback == "低风险" else 0)
    return out


FIELDS = [
    "data_id", "content", "label_v2", "is_violation", "is_fraud", "risk_level",
    "scene_category", "scene_subtype", "label_detail", "label_source",
    "needs_review", "proposed_risk_level", "proposed_scene_category",
    "proposed_subtype", "proposed_is_fraud",
    "old_label", "old_category", "人工复核", "备注",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="assets/0729复核人工确认数据.csv")
    ap.add_argument("--out", default="assets/0729复核人工确认数据_v2.csv")
    ap.add_argument("--out-dir", default="runs/labels_v2")
    args = ap.parse_args()

    src = Path(args.csv)
    with src.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{src}: no rows")

    out_rows = [relabel_row(r) for r in rows]
    out = Path(args.out)
    with out.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out_rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Summary.
    summary = {
        "source": str(src),
        "n": len(out_rows),
        "label_v2": dict(collections.Counter(r["label_v2"] for r in out_rows)),
        "is_violation": dict(collections.Counter(r["is_violation"] for r in out_rows)),
        "is_fraud": dict(collections.Counter(r["is_fraud"] for r in out_rows)),
        "risk_level": dict(collections.Counter(r["risk_level"] for r in out_rows)),
        "scene_category": dict(collections.Counter(r["scene_category"] for r in out_rows)),
        "label_source": dict(collections.Counter(r["label_source"] for r in out_rows)),
        "needs_review": dict(collections.Counter(r["needs_review"] for r in out_rows)),
        "risk_by_old_category": {
            cat: dict(collections.Counter(
                r["risk_level"] for r in out_rows if r["old_category"] == cat and r["old_label"] == "涉诈"
            ))
            for cat in sorted({r["old_category"] for r in out_rows if r["old_label"] == "涉诈"})
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Review queue.
    queue = [r for r in out_rows if r["needs_review"]]
    with (out_dir / "review_queue.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(queue)

    print(f"wrote {out} ({len(out_rows)} rows)")
    print(f"label_v2: {summary['label_v2']}")
    print(f"is_fraud: {summary['is_fraud']}")
    print(f"risk_level: {summary['risk_level']}")
    print(f"label_source: {summary['label_source']}")
    print(f"needs_review: {len(queue)} rows -> {out_dir / 'review_queue.csv'}")
    print("risk by old category:")
    for cat, dist in summary["risk_by_old_category"].items():
        print(f"  {cat}: {dist}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
