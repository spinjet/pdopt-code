# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 10:49:57 2021

Improved version of the energy management simulation code.
Energy Method is used to calculate the required power from a given flight path.

Method used to calculate emission index (EI) of the gas turbine is based on the
Boeing FuelFlow2 method (https://doi.org/10.4271/2006-01-1987), with data 
pulled from Filippone et al. (https://doi.org/10.1016/j.trd.2018.01.019).

@author: Andrea Spinelli, Hossein B. Enalou
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

from scipy.interpolate import interp1d
from engine_map import engine_map


def Ma(alt, V, delta_T=0):
    # ISA Atmosphere Mach

    # Temperature in Kelvin
    if alt < 11000:
        T = delta_T + 288.15 - alt * 0.0065
    else:
        T = delta_T + 216.65

    # Speed in m/s
    mach = V / np.sqrt(1.44 * 287.058 * T)

    return mach


def calculate_segment_Pe_loss(P_req, eta_gbp, eta_e_comp, **kwargs):
    # Power required from electrical system
    if "p_doh" in kwargs:
        P_EC = (P_req * kwargs["p_doh"]) / eta_gbp
    else:
        P_EC = kwargs["P_EM"] / eta_gbp

    P_comp, P_loss = [P_EC], [
        0,
    ]

    for key in eta_e_comp:
        eta_ith = eta_e_comp[key]
        P_comp.append(P_comp[-1] / eta_ith)  # Calc P_in of component
        P_loss.append(P_comp[-1] * ((1 - eta_ith)))  # Calc P_loss of component

    return P_comp[1:], P_loss[1:]


def calculate_segment_Pgt_loss(P_req, eta_gbp, ma, alt, **kwargs):
    # Power required from electrical system

    if "p_doh" in kwargs:
        P_GT = (P_req * (1 - kwargs["p_doh"])) / eta_gbp
    else:
        P_GT = (P_req / eta_gbp) - kwargs["P_EM"]

    # GT power after the gas turbine
    P_out = [P_GT]
    _, gt_eff, _, _ = engine_map(alt, ma, P_GT)

    P_out.append(P_out[-1] / gt_eff)
    P_out.append(P_out[-1] * ((1 - gt_eff)))
    P_out.append(gt_eff)

    return P_out


def calculate_segment_doh(m, alt, V, Vv, time, LD, p_doh, eta_e, eta_gbp, e_bat):
    # Required Power

    P_req_total = V * 9.81 * (m / LD) + Vv * m * 9.81
    P_req = P_req_total / 2

    if P_req > 0:
        # Power splits
        P_GT = (P_req * (1 - p_doh)) / eta_gbp
        # P_GT  = (P_req * (1-p_doh)) / (eta_gbp * 0.3)
        P_EC = (P_req * p_doh) / (eta_gbp * eta_e)

        mach = Ma(alt, V)
        f_flow, gt_eff, f_NOx, f_CO = engine_map(alt, mach, P_GT)

        # Calculate fuel and battery mass
        # e_fl   = 11900 #Wh/kg

        m_fl = time * f_flow
        m_NOx = time * f_NOx / 1000  # Convert to g to kg
        m_CO = time * f_CO / 1000  # Convert to g to kg

        # m_fl   = (time/3600 * P_GT) / e_fl
        m_batt = (time / 3600 * P_EC) / e_bat

        # print('power', P_req, P_GT, P_EC)
        # print('ff', time, f_flow)

        P_tot = P_GT / gt_eff + P_EC
        eff = P_req / (P_GT / gt_eff + P_EC)
        # eff = P_req / (P_GT + P_EC)
    else:
        m_batt = 0
        m_fl = 0
        eff = 0
        P_req = 0
        P_tot = 0
        m_NOx = 0
        m_CO = 0

    return m_batt * 2, m_fl * 2, eff, P_req * 2, P_tot * 2, m_NOx * 2, m_CO * 2


def calculate_segment_Pem(m, alt, V, Vv, time, LD, P_EM, eta_e, eta_gbp, e_bat):
    # Required Power

    P_req = V * 9.81 * (m / LD) + Vv * m * 9.81

    # Power splits
    # P_GT  = (P_req * (1-p_doh)) / eta_gbp
    P_GT = (P_req / eta_gbp) - P_EM
    P_EC = P_EM / (eta_e * eta_gbp)

    mach = Ma(alt, V)
    f_flow, gt_eff = engine_map(alt, mach, P_GT)

    # Calculate fuel and battery mass
    # e_fl   = 11900 #Wh/kg

    m_fl = time * f_flow
    # m_fl   = (time/3600 * P_GT) / e_fl
    m_batt = (time / 3600 * P_EC) / e_bat

    # print('power', P_req, P_GT, P_EC)
    # print('ff', time, f_flow)

    P_tot = P_GT / gt_eff + P_EC
    eff = P_req / (P_GT / gt_eff + P_EC)
    # eff = P_req / (P_GT + P_EC)

    return m_batt, m_fl, eff, P_req, P_tot


def postpro_run(architecture, results):
    # Calculate the lost energy of each electric component and GT

    eta_e = architecture["eta_e"]
    eta_e_comp = architecture["eta_e_comp"]  # Dictionary with eff of elec components
    eta_gbp = architecture["eta_gb"]

    P_elec, P_loss = [], []
    P_GT = []

    for i in range(len(results.iloc[1:-1])):
        # print(i)
        P_req = results.iloc[i].P_req
        p_doh = results.iloc[i].pDOH
        alt = results.iloc[i].alt
        V = results.iloc[i].V
        ma = Ma(alt, V)

        p_c, p_l = calculate_segment_Pe_loss(
            P_req / 2, eta_gbp, eta_e_comp, p_doh=p_doh
        )
        p_gt = calculate_segment_Pgt_loss(P_req / 2, eta_gbp, ma, alt, p_doh=p_doh)
        P_elec.append(p_c)
        P_loss.append(p_l)
        P_GT.append(p_gt)

    ## Build output
    df_Pelec = pd.DataFrame(P_elec, columns=["P_{}".format(k) for k in eta_e_comp])
    df_Ploss = pd.DataFrame(P_loss, columns=["P_loss_{}".format(k) for k in eta_e_comp])
    df_P_GT = pd.DataFrame(P_GT, columns=["P_GT_out", "P_GT_in", "P_GT_loss", "eta_GT"])

    df_output = pd.concat([results, df_Pelec, df_Ploss, df_P_GT], axis=1)

    return df_output


def main_run(mission, architecture, verbose=False):
    # Unpack Data

    eta_e = architecture["eta_e"]
    MTOM = architecture["MTOM"]
    OEM = architecture["OEM"]
    M_py = architecture["M_py"]
    e_bat = architecture["e_bat"]
    eta_gb = architecture["eta_gb"]

    ## Calculation loop
    # if verbose:
    #     print('GT Fuel Interpolation Score: {:.3f}'.format())

    ## assumed
    m_fl = 2000
    m_bt = 1000

    m_old = 0

    # To break out in case of infinite loop
    iteration = 0

    while abs(m_old - (m_fl + m_bt)) > 0.01:
        if verbose:
            print("Mass Error: {:.4f}".format(abs(m_old - (m_fl + m_bt))))

        m_old = m_fl + m_bt
        M, P = [], []
        TOM = OEM + M_py + m_fl + m_bt

        M.append([TOM, 0, 0, 0, 0, OEM, M_py])
        P.append([0, 0, 0])

        for i in mission.index:
            # print(i)
            segment = mission.iloc[i]
            # print(M[-1][0])

            if "time" in segment.index:
                dt = segment.time
            else:
                dt = segment.range / (segment.V)

            m_batt, m_fuel, eff, P_req, P_tot, m_NOx, m_CO = calculate_segment_doh(
                M[-1][0],
                segment.alt,
                segment.V,
                segment.Vz,
                dt,
                segment.LD,
                segment.pDOH,
                eta_e,
                eta_gb * segment.eff_p,
                e_bat,
            )

            M.append([M[-1][0] - m_fuel, m_fuel, m_batt, m_NOx, m_CO, OEM, M_py])
            P.append([eff, P_req, P_tot])

        m_fl = sum([x[1] for x in M])
        m_bt = sum([x[2] for x in M])

        iteration += 1
        if iteration > 30:
            print("Did not converge")
            break

    ## Build output
    df_output = pd.DataFrame(
        [[0 for i in range(len(mission.columns))]], columns=mission.columns
    )

    # df_output = df_output.append(mission, ignore_index=True)
    df_output = pd.concat([df_output, mission], ignore_index=True)

    df_mass = pd.DataFrame(
        M, columns=["mass", "m_fl", "m_bat", "m_NOx", "m_CO", "oem", "m_py"]
    )
    df_eff = pd.DataFrame(P, columns=["eff", "P_req", "P_tot"])

    df_output = pd.concat([df_output, df_mass, df_eff], axis=1)

    # Build summary row

    summary = df_output.iloc[0].copy()
    summary.name = "total"

    if "time" in summary.index:
        summary.time = df_output.time.sum()

    summary.range = df_output.range.sum()
    summary.m_fl = df_output.m_fl.sum()
    summary.m_bat = df_output.m_bat.sum()
    summary.m_NOx = df_output.m_NOx.sum()
    summary.m_CO = df_output.m_CO.sum()

    summary.eff = df_output.P_req.sum() / df_output.P_tot.sum()
    summary.P_req = df_output.P_req.sum()
    summary.P_tot = df_output.P_tot.sum()

    # df_output = df_output.append(summary)
    df_output = pd.concat([df_output, pd.DataFrame([summary])])

    return df_output


def generate_doh(energy_mgm, mission):
    ## Define the energy strategy as a piecewise polynomial for each
    ## mission segments.
    # Input Format
    # e_mgm = pd.DataFrame([['climbout', 1, 0.5],
    #                       ['climb', 0, 0.44],
    #                       ['climb', 1, 0.22],
    #                       ['cruise', 0, 0.4],
    #                       ['cruise', 0.3, 0.6],
    #                       ['cruise', 1, 0.2]],
    #                      columns=['segment','x','doh'])

    # mission = pd.read_csv('mission_stutt.csv')

    for s in set(energy_mgm.segment):
        mgm = energy_mgm.loc[energy_mgm["segment"] == s]
        seg = mission.loc[mission["tag"] == s]

        t = seg["time"].cumsum() / seg["time"].sum()

        ## TO BE SURE
        t_np = t.to_numpy()
        t_np[t_np < 0] = 0
        t_np[t_np > 1] = 1.0

        # number of points in segment
        n_s = len(seg)

        if len(mgm) == 1:
            # uniform
            tmp = mgm.iloc[0, 2] * np.ones(n_s)

        else:
            # picewise linear
            # tmp = interp1d(list(mgm.x), list(mgm.doh))(np.linspace(0,1,n_s))
            tmp = interp1d(list(mgm.x), list(mgm.doh))(t_np)

        mission.loc[mission["tag"] == s, "pDOH"] = None
        mission.loc[mission["tag"] == s, "pDOH"] = tmp.astype(float)

    return mission


def model(
    energy_management,
    mission_file="data/mission.csv",
    architecture_file="data/architecture.json",
    architecture_data=None,
    verbose=False,
):
    mission = pd.read_csv(mission_file)

    if architecture_data is None:
        architecture = json.load(open(architecture_file, "r"))
    else:
        architecture = architecture_data

    # If there is no energy_management, then skip and use the baseline mission
    if energy_management is not None:
        m = generate_doh(energy_management, mission)
        result = main_run(m, architecture, verbose=verbose)
    else:
        result = main_run(mission, architecture, verbose=verbose)
    return result


if __name__ == "__main__":
    # Calculate Baseline Quantities for evaluating the relative reduction
    baseline = model(None, mission_file="data/mission_original.csv")
    # baseline.to_csv('data/baseline.csv', index=False)

    mission_m055 = model(None, mission_file="data/mission_m055.csv")
    mission_hot = model(None, mission_file="data/mission_hothigh.csv")
    mission_srwy = model(None, mission_file="data/mission_shortRwy.csv")
    mission_r621 = model(None, mission_file="data/mission_r621.csv")

    # baseline.iloc[-1].to_json('data/baseline.json')

    plt.figure(figsize=(10, 5))
    d = mission_hot.iloc[1:-1, 3].cumsum().to_numpy() / 1000
    h = mission_hot.iloc[1:-1, 2].to_numpy()
    d_t, h_t = [], []
    for tag in mission_hot.tag.unique()[1:]:
        idx = mission_hot.loc[mission_hot.tag == tag].first_valid_index()
        d_t.append(d[idx - 1])
        h_t.append(h[idx - 1])

    plt.plot(d, h, "k")
    plt.plot(d_t, h_t, "ok")
    plt.text(5 + 0.5 * (d_t[1] + d_t[0]), 0.5 * (h_t[1] + h_t[0]) - 150, "TO")
    plt.text(5 + 0.5 * (d_t[2] + d_t[1]), 0.5 * (h_t[2] + h_t[1]) - 80, "Climbout")
    plt.text(0.5 * (d_t[3] + d_t[2]) - 5, 0.5 * (h_t[3] + h_t[2]) - 150, "Climb")
    plt.text(0.5 * (d_t[4] + d_t[3]) - 15, 0.5 * (h_t[4] + h_t[3]) - 400, "Cruise")
    plt.text(0.5 * (d_t[5] + d_t[4]) - 35, 0.5 * (h_t[5] + h_t[4]) - 150, "Descent")
    plt.text(0.5 * (d_t[6] + d_t[5]) - 25, 0.5 * (h_t[6] + h_t[5]) - 150, "Final")
    plt.grid()
    plt.xlabel("Distance [km]"), plt.ylabel("Altitude [m]")
    plt.tight_layout()
    plt.savefig("hothigh_mission.eps", dpi=300, bbox_inches="tight")

    plt.figure(figsize=(10, 5))
    d = mission_r621.iloc[1:-1, 4].cumsum().to_numpy() / 1000
    h = mission_r621.iloc[1:-1, 3].to_numpy()
    d_t, h_t = [], []

    for tag in mission_r621.tag.unique()[1:]:
        idx = mission_r621.loc[mission_r621.tag == tag].first_valid_index()
        d_t.append(d[idx - 1])
        h_t.append(h[idx - 1])

    plt.plot(d, h, "k")
    plt.plot(d_t, h_t, "ok")
    plt.text(5 + 0.5 * (d_t[1] + d_t[0]), 0.5 * (h_t[1] + h_t[0]) - 150, "TO")
    plt.text(3 + 0.5 * (d_t[2] + d_t[1]), 0.5 * (h_t[2] + h_t[1]) - 130, "Climbout")
    plt.text(0.5 * (d_t[3] + d_t[2]) - 5, 0.5 * (h_t[3] + h_t[2]), "Climb")
    plt.text(0.5 * (d_t[4] + d_t[3]) - 25, 0.5 * (h_t[4] + h_t[3]) - 400, "Cruise")
    plt.text(0.5 * (d_t[5] + d_t[4]) - 65, 0.5 * (h_t[5] + h_t[4]) + 100, "Descent")
    plt.text(0.5 * (d_t[6] + d_t[5]) - 35, 0.5 * (h_t[6] + h_t[5]) - 150, "Final")
    plt.grid()
    plt.xlabel("Distance [km]"), plt.ylabel("Altitude [m]")
    plt.tight_layout()
    plt.savefig("european_mission.eps", dpi=300, bbox_inches="tight")

    # sanity check

    # m_cran = pd.read_csv('data/mission.csv')
    # a_cran = json.load(open('data/architecture.json','r'))

    # df_cr0  = main_run(m_cran, a_cran, verbose=True)

    # mgm1 = pd.DataFrame([['climb',  1, 0],
    #                       ['cruise', 1, 0.3]],
    #                       columns=['segment','x','doh'])

    # m1 = generate_doh(mgm1, m_cran)
    # df_cr1  = main_run(m1, a_cran, verbose=True)

    # mgm2 = pd.DataFrame([['climb',  1, 0.3],
    #                       ['cruise', 1, 0]],
    #                       columns=['segment','x','doh'])

    # m2 = generate_doh(mgm2, m_cran)
    # df_cr2  = main_run(m2, a_cran, verbose=True)

    # mgm3 = pd.DataFrame([['cruise',  0, 0.2],
    #                      ['cruise', 0.3, 0.1],
    #                      ['cruise', 1, 0.3]],
    #                       columns=['segment','x','doh'])

    # m3 = generate_doh(mgm3, m_cran)
    # df_cr3  = main_run(m3, a_cran, verbose=True)

    # tom = [x.loc['total', 'mass'] for x in [df_cr0, df_cr1, df_cr2, df_cr3]]
    # m_f = [x.loc['total', 'm_fl'] for x in [df_cr0, df_cr1, df_cr2, df_cr3]]
    # m_b = [x.loc['total', 'm_bat'] for x in [df_cr0, df_cr1, df_cr2, df_cr3]]
    # m_nox = [x.loc['total', 'm_NOx'] for x in [df_cr0, df_cr1, df_cr2, df_cr3]]

    # postpro_run(a_cran, df_cr3).to_csv('temp.csv')

    ## visualisation

    # plt.figure()
    # plt.plot([0, 0.3], tom[0:2], 'o-r', label='CR')
    # plt.plot([0, 0.3], [tom[0], tom[2]], 'o-b', label='CL')
    # plt.plot([0, 0.3], [tom[0], tom[3]], 'o-g', label='CL+CR')
    # plt.legend(), plt.title('TOM vs DOH')
    # plt.ylabel('TOM [kg]'), plt.xlabel('DOH')

    # plt.figure()
    # plt.plot([0, 0.3], m_f[0:2], 'o-r', label='CR')
    # plt.plot([0, 0.3], [m_f[0], m_f[2]], 'o-b', label='CL')
    # plt.plot([0, 0.3], [m_f[0], m_f[3]], 'o-g', label='CL+CR')
    # plt.legend(), plt.title('M_F vs DOH')
    # plt.ylabel('M Fuel [kg]'), plt.xlabel('DOH')

    # plt.figure()
    # plt.plot([0, 0.3], m_b[0:2], 'o-r', label='CR')
    # plt.plot([0, 0.3], [m_b[0], m_b[2]], 'o-b', label='CL')
    # plt.plot([0, 0.3], [m_b[0], m_b[3]], 'o-g', label='CL+CR')
    # plt.legend(), plt.title('M_B vs DOH')
    # plt.ylabel('M Battery [kg]'), plt.xlabel('DOH')

    # plt.figure()
    # plt.plot([0, 0.3], m_nox[0:2], 'o-r', label='CR')
    # plt.plot([0, 0.3], [m_nox[0], m_nox[2]], 'o-b', label='CL')
    # plt.plot([0, 0.3], [m_nox[0], m_nox[3]], 'o-g', label='CL+CR')
    # plt.legend(), plt.title('M_NOx vs DOH')
    # plt.ylabel('M NOx [kg]'), plt.xlabel('DOH')
