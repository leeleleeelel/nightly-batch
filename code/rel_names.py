# Korean names for feature columns (shared by report builders)
import re
SK = {"r": "수익률", "dv": "거래대금", "rng": "하루 변동폭", "gap": "시가 갭", "intr": "장중 수익", "clv": "종가 위치", "rsi": "RSI", "macd": "MACD", "mhist": "MACD 히스토그램",
      "pb": "볼린저 %b", "vz": "거래량 급증도", "svr": "공매도 거래 비율", "stk": "스토캐스틱", "mkt": "시장 수익률", "sv": "방향 붙은 거래량"}
LIT = {"fip20": "급등 전 20일 상승의 꾸준함(FIP)", "fip60": "급등 전 60일 상승의 꾸준함(FIP)", "surge_conc": "급등 3일 중 하루 몰림 정도", "maxret21": "급등 전 한 달 최대 하루 상승",
       "hvp": "급등 직전 5일 거래량 급증", "llor_c2": "거래량-수익률 상호작용 계수(Llorente)"}
SB = {"sb_n_z2": "평소보다 튄 지표 쌍의 수(|z|≥2)", "sb_n_z3": "크게 튄 지표 쌍의 수(|z|≥3)", "sb_n_zmax3": "최근 20일 안에 크게 튄 쌍의 수", "sb_max_absz": "가장 크게 튄 쌍의 크기",
      "sb_mean_absz": "지표 쌍 평균 튐 크기", "sb_frob": "지표 관계 구조 전체의 변화량", "sb_eig1_20": "최근 지표들이 한 몸처럼 움직인 정도", "sb_eig1_60": "예전 지표들이 한 몸처럼 움직인 정도",
      "sb_d_eig1": "지표들이 한 몸처럼 움직이는 정도의 변화", "sb_n_uz2": "평소보다 튄 지표 수"}
fa = lambda x: "–" if x is None or pd.isna(x) else f"{x:.3f}"
WS = {"ret": "수익률", "absret": "하루 움직임 크기", "logv": "거래량(로그)", "rng": "하루 변동폭", "intr": "장중 수익", "gap": "시가 갭", "clv": "종가 위치", "vwapdev": "VWAP 이탈",
      "rel": "시장 대비 수익률", "grouprel": "업종 대비 수익률", "xsrank": "전체 종목 중 순위", "svr": "공매도 거래 비율", "wick": "윗꼬리"}
WO = {"mean": "{n}일 평균", "std": "{n}일 변동성", "z": "{n}일 기준 z점수", "rank": "{n}일 안 순위", "delta": "{n}일 전 대비 변화", "decay": "{n}일 최근 가중 평균", "slope": "{n}일 기울기",
      "scale": "{n}일 기준 상대 크기", "argmax": "{n}일 중 최고가 며칠 전", "argmin": "{n}일 중 최저가 며칠 전", "accel": "{n}일 가속", "skew": "{n}일 치우침", "kurt": "{n}일 꼬리 두께", "r": "{n}일 수익률"}
EV = {"r3": "급등 3일 상승률", "dtc": "공매도 상환 일수(days to cover)", "co10_05": "최근 10일 같이 급등한 종목 수", "sic4_hold10": "같은 세부 업종 급등주의 10일 유지",
      "p_r1_rec": "최근 하루 상승률 순위", "nonacc": "소형 신고 기업(비가속)", "pscore": "초소형 프로필 점수", "days_below5_30": "최근 30일 중 5달러 아래였던 날 수", "pre14_60": "최근 60일 주총 위임장 예비 공시(PRE 14A) 수", "weekday": "요일", "i503_60": "최근 60일 정관 변경 공시(8-K 5.03: 주식 병합·수권 주식 증가 등) 수", "trig_same_day": "같은 날 급등 신호가 뜬 다른 종목 수", "si_chg6": "공매도 잔고 6회 전 대비 변화", "mkt_median": "시장 중간 종목 수익률", "p_hiclose20": "최근 20일 고가 근처 마감 비율"}
def fname(f):
    m = re.match(r"^(c10|jump|z10|zmax20|zlag)\|(\w+)~(\w+)$", f)
    if m:
        k, a, b = m.groups(); a, b = SK.get(a, a), SK.get(b, b)
        return {"c10": f"{a}–{b} 최근 10일 상관", "jump": f"{a}–{b} 상관 변화(최근 10일 − 그 전 60일)", "z10": f"{a}–{b} 상관이 그 종목 평소 대비 튄 정도",
                "zmax20": f"최근 20일 중 {a}–{b} 상관이 가장 크게 튄 정도", "zlag": f"{a}–{b} 상관이 가장 크게 튄 날이 며칠 전인가"}[k]
    m = re.match(r"^(ll20|lljump)\|(\w+)>(\w+)$", f)
    if m:
        k, a, b = m.groups(); a, b = SK.get(a, a), SK.get(b, b)
        return f"어제 {a} → 오늘 {b} 상관" + (" 변화" if k == "lljump" else "")
    m = re.match(r"^(uz1|uz5|ud2z|uflip)\|(\w+)$", f)
    if m:
        k, a = m.groups(); a = SK.get(a, a)
        return {"uz1": f"{a} 급등일 값이 평소 대비", "uz5": f"{a} 최근 5일 평균이 평소 대비", "ud2z": f"{a} 5일 평균의 가속(2차 차분)", "uflip": f"{a} 추세가 거꾸로 꺾임(2차 차분)"}[k]
    m = re.match(r"^sb_hub\|(\w+)$", f)
    if m: return f"가장 많이 튄 지표가 {SK.get(m.group(1), m.group(1))}"
    m = re.match(r"^(\w+)\|([a-z]+?)(\d+)$", f)
    if m and m.group(1) in WS and m.group(2) in WO:
        if m.group(2) == "r": return f"{m.group(3)}일 수익률의 전체 종목 중 순위" if m.group(1) == "xsrank" else f"{WS[m.group(1)]} ({m.group(3)}일)"
        return f"{WS[m.group(1)]} {WO[m.group(2)].format(n=m.group(3))}"
    return EV.get(f, LIT.get(f, SB.get(f, f)))
