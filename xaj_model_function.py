# coding=utf-8
# written by wjdlut 2020
import numpy as np
from datetime import *
import math


def antecedent_soil_moisture(Kc, Um, Lm, C, Wm, B, Im, WInitialDay, Eall, DayPrecip, DayEvap, DayDate, ET):
    WFinalDay = np.zeros(3)  # 前期土壤含水量(张力水含量) 以 day 为尺度

    if len(DayPrecip) == 0:
        WFinalDay = WInitialDay

    Dm = Wm - Um - Lm  # 深层张力水蓄水容量
    Wmm = Wm * (1.0 + B) / (1.0 - Im)  # 流域单点最大蓄水容量

    WuDay = np.zeros(len(DayPrecip) + 1)  # 流域上层张力水蓄水容量(面均值)
    WlDay = np.zeros(len(DayPrecip) + 1)  # 流域下层张力水蓄水容量(面均值)
    WdDay = np.zeros(len(DayPrecip) + 1)  # 流域深层张力水蓄水容量(面均值)

    WuDay[0] = WInitialDay[0]
    WlDay[0] = WInitialDay[1]
    WdDay[0] = WInitialDay[2]

    EuDay = np.zeros(len(DayPrecip))
    ElDay = np.zeros(len(DayPrecip))
    EdDay = np.zeros(len(DayPrecip))

    RunoffDay = np.zeros(len(DayPrecip))

    if Dm < 0:
        print('Dm<0，计算错误！')
        pass
    else:
        for i in range(len(DayPrecip)):
            Wmt = WuDay[i] + WlDay[i] + WdDay[i]
            if Wmt > Wm:
                Wmt = Wm

            # 蒸发量计算，可采用两种方法
            day_Ept = 0.0
            # day_Ept = Kc * DayEvap[i]  # 实测日蒸散发无法统计
            if len(ET) == 2:  # 三级蒸散发
                E1, E2 = ET[0], ET[1]
                day_Ept = Kc * calc_evaporation_potential_three(Eall, 24, DayDate[i], DayPrecip[i], E1, E2)
            if len(ET) == 3:  # 四级蒸散发
                E1, E2, E3 = ET[0], ET[1], ET[2]
                day_Ept = Kc * calc_evaporation_potential_four(Eall, 24, DayDate[i], DayPrecip[i], E1, E2, E3)
            if len(ET) == 4:  # 五级蒸散发
                E1, E2, E3, E4 = ET[0], ET[1], ET[2], ET[3]
                day_Ept = Kc * calc_evaporation_potential_five(Eall, 24, DayDate[i], DayPrecip[i], E1, E2, E3, E4)

            PEt = DayPrecip[i] - day_Ept
            if PEt >= 0:
                temp1 = math.pow((1 - Wmt / Wm), 1.0 / (1 + B))
                At = Wmm * (1 - temp1)

                if (PEt + At) < Wmm:
                    temp2 = math.pow((1 - (PEt + At) / Wmm), (1 + B))
                    RunoffDay[i] = PEt - (Wm - Wmt) + Wm * temp2
                else:
                    RunoffDay[i] = PEt - (Wm - Wmt)

                EuDay[i] = day_Ept  # 产流了，只有上层蒸发
                ElDay[i] = 0.0
                EdDay[i] = 0.0

                if (WuDay[i] + PEt - RunoffDay[i]) < Um:  # 如果上层未饱和
                    WuDay[i + 1] = WuDay[i] + PEt - RunoffDay[i]
                    WlDay[i + 1] = WlDay[i]
                    WdDay[i + 1] = WdDay[i]
                else:  # 如果上层饱和，分情况讨论下层是否饱和
                    if (WuDay[i] + WlDay[i] + PEt - RunoffDay[i]) < (Um + Lm):  # 如果上层饱和，下层未饱和
                        WuDay[i + 1] = Um
                        WlDay[i + 1] = WuDay[i] + WlDay[i] + PEt - RunoffDay[i] - Um
                        WdDay[i + 1] = WdDay[i]
                    else:  # 如果上层饱和，下层饱和
                        WuDay[i + 1] = Um
                        WlDay[i + 1] = Lm
                        WdDay[i + 1] = Wmt + PEt - WuDay[i + 1] - WlDay[i + 1] - RunoffDay[i]
            if PEt < 0:
                # 如果没净雨产生，上、下、深层张力水蓄量主要受到蒸发影响
                RunoffDay[i] = 0

                # 如果超过蒸散发能力，即 WuDay[i] + dayPrecip[i] - EPt >= 0 仅有上层蒸发
                if (WuDay[i] + PEt) >= 0:
                    EuDay[i] = day_Ept
                    ElDay[i] = 0.0
                    EdDay[i] = 0.0
                    WuDay[i + 1] = WuDay[i] + PEt
                    WlDay[i + 1] = WlDay[i]
                    WdDay[i + 1] = WdDay[i]

                else:  # 如果没超过蒸散发能力，即WuDay[i] + dayPrecip[i] - EPt < 0，分情况讨论是否超过下层蒸散发能力
                    EuDay[i] = WuDay[i] + DayPrecip[i]
                    WuDay[i + 1] = 0.0

                    if WlDay[i] >= C * Lm:  # 如果超过下层蒸散发能力
                        ElDay[i] = (day_Ept - EuDay[i]) * WlDay[i] / Lm
                        EdDay[i] = 0
                        WlDay[i + 1] = WlDay[i] - ElDay[i]
                        WdDay[i + 1] = WdDay[i]
                    # 如果没超过下层蒸散发能力，分两种情况讨论是否超过深层蒸散发能力，以 C * (EPt-EuDay[i])为分界
                    else:
                        if WlDay[i] >= C * (day_Ept - EuDay[i]):  # 如果大于，下层按能力蒸散发，深层无蒸散发
                            ElDay[i] = C * (day_Ept - EuDay[i])
                            EdDay[i] = 0
                            WlDay[i + 1] = WlDay[i] - ElDay[i]
                            WdDay[i + 1] = WdDay[i]
                        else:  # 如果小于，下层全部都蒸散发，深层有蒸散发
                            ElDay[i] = WlDay[i]
                            EdDay[i] = C * (day_Ept - EuDay[i]) - WlDay[i]
                            WlDay[i + 1] = 0.0
                            WdDay[i + 1] = WdDay[i] - EdDay[i]

            # 下一个时段初值不能小于零
            if WuDay[i + 1] < 0:
                WuDay[i + 1] = 0.0
            if WlDay[i + 1] < 0:
                WlDay[i + 1] = 0.0
            if WdDay[i + 1] < 0:
                WdDay[i + 1] = 0.0
            # 下一个时段初值不能大于最大张力水容量
            if WuDay[i + 1] > Um:
                WuDay[i + 1] = Um
            if WlDay[i + 1] > Lm:
                WlDay[i + 1] = Lm
            if WdDay[i + 1] > Dm:
                WdDay[i + 1] = Dm

            WFinalDay[0] = WuDay[i + 1]
            WFinalDay[1] = WlDay[i + 1]
            WFinalDay[2] = WdDay[i + 1]


    return WFinalDay


def calc_evaporation_potential_three(Eall, time_Interval, date, precip, E1, E2):
    Month = datetime.strptime(date, "%Y-%m-%d %H:%M:%S").month
    Ev = np.zeros(3)
    Ep = 0.0
    dict = {4: Eall[0], 5: Eall[1], 6: Eall[2], 7: Eall[3], 8: Eall[4], 9: Eall[5], 10: Eall[6]}

    if Month in dict:
        Ev = dict[Month]
    else:
        print("The Month is not exised！")
    precip = precip * 24 / time_Interval

    if precip < E1:
        Ep = Ev[0]
    elif E1 <= precip < E2:
        Ep = Ev[0] * Ev[1]
    elif precip >= E2:
        Ep = Ev[0] * Ev[1] * Ev[2]
    else:
        print('Ep simulation is wrong.')

    Ep = Ep * time_Interval / 24

    return Ep


def calc_evaporation_potential_four(Eall, timeInterval, date, precip, E1, E2, E3):
    Month = datetime.strptime(date, "%Y-%m-%d %H:%M:%S").month
    Ev = np.zeros(4)
    Ep = 0.0
    dict = {4: Eall[0], 5: Eall[1], 6: Eall[2], 7: Eall[3], 8: Eall[4], 9: Eall[5], 10: Eall[6]}

    if Month in dict:
        Ev = dict[Month]
    else:
        print("The Month is not exised！")
    precip = precip * 24 / timeInterval

    if precip < E1:
        Ep = Ev[0]
    elif E1 <= precip < E2:
        Ep = Ev[0] * Ev[1]
    elif E2 <= precip < E3:
        Ep = Ev[0] * Ev[1] * Ev[2]
    elif precip >= E3:
        Ep = Ev[0] * Ev[1] * Ev[2] * Ev[3]
    else:
        print('Ep simulation is wrong.')

    Ep = Ep * timeInterval / 24

    return Ep

def calc_evaporation_potential_five(Eall, timeInterval, date, precip, E1, E2, E3, E4):
    Month = datetime.strptime(date, "%Y-%m-%d %H:%M:%S").month
    Ev = np.zeros(5)
    Ep = 0.0
    dict = {4: Eall[0], 5: Eall[1], 6: Eall[2], 7: Eall[3], 8: Eall[4], 9: Eall[5], 10: Eall[6]}

    if Month in dict:
        Ev = dict[Month]
    else:
        print("The Month is not exised！")
    precip = precip * 24 / timeInterval

    if precip < E1:
        Ep = Ev[0]
    elif E1 <= precip < E2:
        Ep = Ev[0] * Ev[1]
    elif E2 <= precip < E3:
        Ep = Ev[0] * Ev[1] * Ev[2]
    elif E3 <= precip < E4:
        Ep = Ev[0] * Ev[1] * Ev[2] * Ev[3]
    elif precip >= E4:
        Ep = Ev[0] * Ev[1] * Ev[2] * Ev[3] * Ev[4]
    else:
        print('Ep simulation is wrong.')

    Ep = Ep * timeInterval / 24

    return Ep


def calc_runoff(Kc, Um, Lm, C, Wm, B, Im, InitialW, timeInterval, Eall, date, precip, evapor, ET):
    runoff = [0.0 for k in range(len(precip))]
    runoff_imp = [0.0 for k in range(len(precip))]

    Dm = Wm - Lm - Um
    Wmm = Wm * (1.0 + B) / (1.0 - Im)

    Wu = np.zeros(len(precip) + 1)
    Wl = np.zeros(len(precip) + 1)
    Wd = np.zeros(len(precip) + 1)

    Wu[0] = InitialW[0]
    Wl[0] = InitialW[1]
    Wd[0] = InitialW[2]

    Eu = np.zeros(len(precip))
    El = np.zeros(len(precip))
    Ed = np.zeros(len(precip))

    if Dm < 0:
        # print("Dm is negative, Runoff simulation is wrong!")
        pass
    else:  # 以小时为时段的循环开始
        for i in range(len(precip)):
            Wmt = Wu[i] + Wl[i] + Wd[i]
            if Wmt > Wm:
                Wmt = Wm

            Evpt = 0.0
            if len(ET) == 2:
                E1, E2 = ET[0], ET[1]
                Evpt = Kc * calc_evaporation_potential_three(Eall, timeInterval, date[i], precip[i], E1, E2)
            if len(ET) == 3:
                E1, E2, E3 = ET[0], ET[1], ET[2]
                Evpt = Kc * calc_evaporation_potential_four(Eall, timeInterval, date[i], precip[i], E1, E2, E3)
            if len(ET) == 4:
                E1, E2, E3, E4 = ET[0], ET[1], ET[2], ET[3]
                Evpt = Kc * calc_evaporation_potential_five(Eall, timeInterval, date[i], precip[i], E1, E2, E3, E4)

            PEt = precip[i] - Evpt

            if PEt >= 0:
                runoff_imp[i] = PEt * Im  # 不透水面积,直接产流
                PEt = PEt * (1 - Im)  # 透水面积蓄满产流

                temp1 = math.pow((1 - Wmt / Wm), 1.0 / (1 + B))
                At = Wmm * (1 - temp1)

                if At + PEt < Wmm:
                    temp2 = math.pow((1 - (PEt + At) / Wmm), (1 + B))
                    runoff[i] = PEt - (Wm - Wmt) + Wm * temp2
                else:
                    runoff[i] = PEt - (Wm - Wmt)

                Eu[i] = Evpt
                El[i] = 0.0
                Ed[i] = 0.0

                # 计算时段末WuDay、WlDay、WdDay
                if Wu[i] + PEt - runoff[i] < Um:
                    Wu[i + 1] = Wu[i] + PEt - runoff[i]
                    Wl[i + 1] = Wl[i]
                    Wd[i + 1] = Wd[i]
                else:
                    if Wu[i] + PEt + Wl[i] - runoff[i] < Um + Lm:
                        Wu[i + 1] = Um
                        Wl[i + 1] = Wu[i] + Wl[i] + PEt - runoff[i] - Um
                        Wd[i + 1] = Wd[i]
                    else:
                        Wu[i + 1] = Um
                        Wl[i + 1] = Lm
                        Wd[i + 1] = Wmt + PEt - runoff[i] - Wu[i + 1] - Wl[i + 1]
            elif PEt < 0:
                runoff[i], runoff_imp[i] = 0, 0  # 不透水面积上产流为零，全部蒸发了

                if Wu[i] + PEt >= 0:
                    Eu[i] = Evpt
                    El[i] = 0
                    Ed[i] = 0

                    Wu[i + 1] = Wu[i] + PEt
                    Wl[i + 1] = Wl[i]
                    Wd[i + 1] = Wd[i]
                else:
                    Eu[i] = Wu[i] + precip[i]
                    Wu[i + 1] = 0.0
                    if Wl[i] >= C * Lm:
                        El[i] = (Evpt - Eu[i]) * Wl[i] / Lm
                        Ed[i] = 0

                        Wl[i + 1] = Wl[i] - El[i]
                        Wd[i + 1] = Wd[i]
                    else:
                        if Wl[i] >= C * (Evpt - Eu[i]):
                            El[i] = C * (Evpt - Eu[i])
                            Ed[i] = 0

                            Wl[i + 1] = Wl[i] - El[i]
                            Wd[i + 1] = Wd[i]
                        else:
                            El[i] = Wl[i]
                            Ed[i] = C * (Evpt - Eu[i]) - Wl[i]

                            Wl[i + 1] = 0
                            Wd[i + 1] = Wd[i] - Ed[i]


            if Wu[i + 1] < 0:
                Wu[i + 1] = 0.0
            if Wl[i + 1] < 0:
                Wl[i + 1] = 0.0
            if Wd[i + 1] < 0:
                Wd[i + 1] = 0.0

            if Wu[i + 1] > Um:
                Wu[i + 1] = Um
            if Wl[i + 1] > Lm:
                Wl[i + 1] = Lm
            if Wd[i + 1] > Dm:
                Wd[i + 1] = Dm


    Ev = []
    for i in range(len(Eu)):
        Ev_h = Eu[i] + El[i] + Ed[i]
        Ev.append(Ev_h)

    return runoff, runoff_imp, Ev, Wu[-1], Wl[-1], Wd[-1]


def runoff_division_v3(KC, SM, EX, KI, time_Interval, initial_fr, initial_s, Eall, precip, date, runoff, ET):
    FR0 = initial_fr  # 上时段的产流面积(占总流域面积的比值)
    S0 = initial_s  # 上时段的产流面积上的平均水深
    S = 0  # 自由水在产流面积上的平均蓄水深
    FR = 0  # 产流面积(占总流域面积的比值)

    AU = 0  # S对应的纵坐标
    SMMF = 0  # 产流面积上最大一点的自由水蓄水容量
    SMF = 0  # 产流面积上的自由水深平均蓄水容量
    SMM = SM * (1 + EX)  # 流域单点最大自由水蓄水容量

    KG = 0.7 - KI

    D = 0
    N = 0
    KSSD = 0
    KGD = 0

    length = len(runoff)
    length_water_withdrawal = 5 * 24 / time_Interval  # 3为退水天数

    RS = [0.0 for k in range(length + int(length_water_withdrawal))]
    RI = [0.0 for k in range(length + int(length_water_withdrawal))]
    RG = [0.0 for k in range(length + int(length_water_withdrawal))]

    for i in range(length):
        rs_ = 0
        ri_ = 0
        rg_ = 0

        E = 0.0
        if len(ET) == 2:
            E1, E2 = ET[0], ET[1]
            E = KC * calc_evaporation_potential_three(Eall, time_Interval, date[i], precip[i], E1, E2)
        if len(ET) == 3:
            E1, E2, E3 = ET[0], ET[1], ET[2]
            E = KC * calc_evaporation_potential_four(Eall, time_Interval, date[i], precip[i], E1, E2, E3)
        if len(ET) == 4:
            E1, E2, E3, E4 = ET[0], ET[1], ET[2], ET[3]
            E = KC * calc_evaporation_potential_five(Eall, time_Interval, date[i], precip[i], E1, E2, E3, E4)

        R = runoff[i]
        P = precip[i]

        if (P - E) > 0:
            FR = R / (P - E)

            FR = 0.001 if FR <= 0.0 else FR
            FR = 1.0 if FR >= 1.0 else FR

            S = S0 * FR0 / FR
            SMMF = SMM * (1 - math.pow((1 - FR), 1 / EX))
            SMF = SMMF / (1 + EX)

            if S > SMF:
                S = S0
                FR = FR0
                SMMF = SMM * (1 - math.pow((1 - FR), 1 / EX))
                SMF = SMMF / (1 + EX)

            D = int(R / (5 * FR)) + 1
            N = 24 * D / time_Interval

            KSSD=(1 - math.pow((1 - (KI + KG)), (1 / N))) / (1 + KG / KI)
            KGD=KSSD * KG / KI

            PE = R / (D * FR)

            for j in range(D):
                rs_tempt = 0
                rg_tempt = 0
                ri_tempt = 0

                AU = SMMF * (1 - math.pow((1 - S / SMF), 1 / (1 + EX)))

                if PE + AU <= 0:
                    rs_tempt = 0
                    rg_tempt = 0
                    ri_tempt = 0
                    S = 0
                elif PE + AU >= SMMF:
                    rs_tempt = (PE + S - SMF) * FR
                    ri_tempt = SMF * KSSD * FR
                    rg_tempt = SMF * KGD * FR
                    S = SMF - (rg_tempt + ri_tempt) / FR
                else:
                    rs_tempt = (PE - SMF + S + SMF * math.pow(1 - (PE + AU) / SMMF, (1 + EX))) * FR
                    ri_tempt = (PE + S - rs_tempt / FR) * KSSD * FR
                    rg_tempt = (PE + S - rs_tempt / FR) * KGD * FR
                    S = S + PE - (rs_tempt + ri_tempt + rg_tempt) / FR

                rs_ += rs_tempt
                rg_ += rg_tempt
                ri_ += ri_tempt

        else:
            KSSD = (1 - math.pow(1 - (KI + KG), (1 / (24.0 / time_Interval)))) / (1 + KG / KI)
            KGD = KSSD * KG / KI

            if S > 0:
                rs_ = 0
                rg_ = S * KGD * FR
                ri_ = S * KSSD * FR
                S = S - (rs_ + ri_ + rg_) / FR
            else:
                rs_ = 0
                rg_ = 0
                ri_ = 0

        S0 = S
        FR0 = FR
        RS[i] = rs_
        RG[i] = rg_
        RI[i] = ri_

    KSSD = (1 - math.pow(1 - (KI + KG), (1 / (24.0 / time_Interval)))) / (1 + KG / KI)
    KGD = KSSD * KG / KI

    for k in range(length, length + int(length_water_withdrawal)):
        if S > 0:
            RS[k] = 0
            RG[k] = S * KGD * FR
            RI[k] = S * KSSD * FR
            S = S - (RS[k] + RG[k] + RI[k]) / FR
        else:
            RS[k] = 0
            RG[k] = 0
            RI[k] = 0

    # return RS[:len(precip)], RI[:len(precip)], RG[:len(precip)]
    return RS, RI, RG


def surface_flow_concentration(Ci, Cg, timeInterval, Area, runoffIm, Rs, Ri, Rg):
    QR = [0.0 for k in range(len(Rs))]
    # 单位转换系数
    U = Area / (3.6 * timeInterval)

    QRs = np.zeros(len(Rs))
    QRi = np.zeros(len(Rs))
    QRg = np.zeros(len(Rs))

    QRs[0] = (Rs[0] + runoffIm[0]) * U
    QRi[0] = Ri[0] * (1 - Ci) * U
    QRg[0] = Rg[0] * (1 - Cg) * U

    QR[0] = QRs[0] + QRi[0] + QRg[0]

    if QR[0] < 0:
        QR[0] = 0

    for i in range(len(Rs)):
        if i >= 1:
            QRs[i] = (Rs[i] + runoffIm[i]) * U
            QRi[i] = QRi[i - 1] * Ci + Ri[i] * (1 - Ci) * U
            QRg[i] = QRg[i - 1] * Cg + Rg[i] * (1 - Cg) * U

            QR[i] = QRs[i] + QRi[i] + QRg[i]  # 河网总入流，单元面积河网总入流

            if QR[i] < 0:
                QR[i] = 0
    return QR


def stream_network_concentration(Cs, L, QR):  # 河网汇流 滞后演算
    # L 河网汇流-滞后时间，代表平移作用; Cs 河网蓄水消退系数
    Qf = [0.0 for k in range(len(QR))]

    T = int(L)
    if T <= 0:
        T = 0
        for i in range(len(QR)):
            if i == 0:
                Qf[0] = (1 - Cs) * QR[0]
            else:
                Qf[i] = Cs * Qf[i - 1] + (1 - Cs) * QR[i]
    else:
        for i in range(len(QR)):
            if i == 0:
                Qf[0] = 0
            elif i < T:
                Qf[i] = Cs * Qf[i - 1]
            else:
                Qf[i] = Cs * Qf[i - 1] + (1 - Cs) * QR[i - T]

    return Qf


def river_channel_concentration(Ke, Xe, timeInterval, Qf):
    Q_outlet = [0.0 for k in range(len(Qf))]

    C0 = (0.5 * timeInterval - Ke * Xe) / (0.5 * timeInterval + Ke - Ke * Xe)
    C1 = (0.5 * timeInterval + Ke * Xe) / (0.5 * timeInterval + Ke - Ke * Xe)
    C2 = 1 - C0 - C1

    Q_outlet[0] = Qf[0]

    if C0 >= 0 and C2 >= 0:  # C0和C2要大于零，否则马斯京根法不适用
        for i in range(len(Qf)):
            if i >= 1:
                Q_outlet[i] = C0 * Qf[i] + C1 * Qf[i - 1] + C2 * Q_outlet[i - 1]
    else:  # 当马斯京根法不适用时候，就不考虑河道演进了，当前参数取值范围情况来看，此种情况不可能发生
        for i in range(len(Qf)):
            Q_outlet[i] = Qf[i]

    return Q_outlet

