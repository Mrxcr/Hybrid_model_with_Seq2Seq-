import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
from scipy import optimize
from scipy.optimize import NonlinearConstraint, LinearConstraint
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from datetime import datetime


# Mechanism model parameters
RHO = np.array([])
R = np.array([])
CPCRHOC = np.array([])
E = np.array([])
TF_BZ = np.array([])
TF_FC = np.array([])

# Initial values for Mechanism model variables 
beta1, c1, b1, a1, cpV1, gamma1 = []
beta2, c2, b2, a2, cpV2, gamma2 = []
beta3, c3, b3, a3, cpV3, gamma3 = []
beta4, c4, b4, a4, cpV4, gamma4 = []
beta5, c5, b5, a5, cpV5, gamma5 = []
beta6, c6, b6, a6, cpV6, gamma6 = []
beta7, c7, b7, a7, cpV7, gamma7 = []

BETA = [beta1, beta2, beta3, beta4, beta5, beta6, beta7]
C = [c1, c2, c3, c4, c5, c6, c7]
B = [b1, b2, b3, b4, b5, b6, b7]
A = [a1, a2, a3, a4, a5, a6, a7]
CPV = [cpV1, cpV2, cpV3, cpV4, cpV5, cpV6, cpV7]
GAMMA = [gamma1, gamma2, gamma3, gamma4, gamma5, gamma6, gamma7]

# vars = np.vstack([BETA, C, B, A, CPV, GAMMA])
paras = np.vstack([RHO, R, E, CPCRHOC, TF_BZ, GAMMA, TF_FC])

# Bounds
LB = []
UB = []
list_con = [(item1, item2) for item1, item2, in zip(LB, UB)]

list_TR = ["TR1", "TR2", "TR3", "TR4", "TR5", "TR6", "TR7"]
list_Tcin, list_BZ = ['Tcin'], ['BZ']
list_Fc = ['Fc1', 'Fc2', 'Fc3', 'Fc4', 'Fc5', 'Fc6', 'Fc7']
list_Tcout = ['Tcout1', 'Tcout2', 'Tcout3', 'Tcout4', 'Tcout5', 'Tcout6', 'Tcout7']
list_DNA, list_SNA, list_SSA = ['F_DNA3', 'F_DNA4'], ['F_SNA3','F_SNA4', 'F_SNA5', 'F_SNA6', 'F_SNA7'], ['F_SSA5', 'F_SSA7']
list_TR_next = ["TR1_next", "TR2_next", "TR3_next", "TR4_next", "TR5_next", "TR6_next", "TR7_next"]

def plot_variables(df, list_var, height=300):
    n_var = len(list_var)
    fig = make_subplots(rows=n_var, cols=1, subplot_titles=list_var, vertical_spacing=0.01, shared_xaxes=True)
    for i in range(n_var):
        fig.add_trace(go.Scatter(x=df.index, y=df[list_var[i]], showlegend=False), col=1, row=i+1)
    fig.update_layout(height=height*n_var, width=1300)
    fig.show()

def pred_KA_quad(Fc, para):
    '''
    Calculate KA
    '''
    return

def Cal_Qsh(Tb, F_BZ_s, F_DNA_s, F_SNA_s, F_SSA_s):
    '''
    Calculate sensible heat Qsh
    Tb: n_sam * 7 array; F_BZ_s: n_sam * 1 BZ flowrate with 1s 
    ''' 
    return

def DynModel_singleOut(Tb, Fc, BZ, Tcin, Tcout_ahead, F_DNA, F_SNA, F_SSA, idx_R, paras, vars):
    '''
    Predict temperature of one reactor
    '''
    return 
    
def DynModel_MulOut(Tb, FC_hat, FC, BZ_hat, BZ, Tcin, Tcout_ahead, F_DNA, F_SNA, F_SSA, paras, vars):
    '''
    Predict temperatures of the seven reactors
    Input:
    Tb: temperatures at timestep t, 1*7;  FC: flow rates of cooling water at timestep t, 1*7;  
    BZ: flow rates of BZ at timestep t; Tcin: inlet temperature of the cooling water;
    F_DNA: flow rates of dilute nitric acid at timestep t, 1*2; F_SNA: flow rates of concentrated nitric acid at timestep t 1*5;
    F_SSA: flow rates of concentrated sulfuric acid at timestep t 1*2;
    Output:
    Tb_next: temperatures at timestep t+1, 1*7;
    '''
    return                

def TransitionFunc(state, dt, FC, BZ, Tcin, Tcout_ahead, F_DNA, F_SNA, F_SSA, paras):
    '''
    Transition function for UKF
    '''
    num = 7
    TR = state[:num][np.newaxis, :]
    BETA, CPV, C = state[num:2*num], state[num*2:3*num], state[num*3:4*num]
    B, A, BZ_hat, FC_hat = state[num*4:5*num], state[num*5:6*num], state[num*6:7*num], state[num*7:8*num]
    
    vars = np.vstack([BETA, C, B, A, CPV])
    Tb_next, BZ_hat_next, FC_hat_next, _, _, _ = DynModel_MulOut(TR, FC_hat, FC, BZ_hat, BZ, Tcin, Tcout_ahead, F_DNA, F_SNA, F_SSA, paras, vars)

    return np.concatenate([Tb_next.flatten(), BETA.flatten(), CPV.flatten()
                           , C.flatten(), B.flatten(), A.flatten()
                           , BZ_hat_next.flatten(), FC_hat_next.flatten()])

def TransitionFunc_plus(state, dt, FC, BZ, Tcin, Tcout_ahead, F_DNA, F_SNA, F_SSA, paras):
    '''
    Transition function used for multistep prediction of temperature and heat
    '''    
    num = 7
    TR = state[:num][np.newaxis, :]
    BETA, CPV, C = state[num:2*num], state[num*2:3*num], state[num*3:4*num]
    B, A, BZ_hat, FC_hat = state[num*4:5*num], state[num*5:6*num], state[num*6:7*num], state[num*7:8*num]
    
    vars = np.vstack([BETA, C, B, A, CPV])
    Tb_next, BZ_hat_next, FC_hat_next, Qr, Qc, Qsh = DynModel_MulOut(TR, FC_hat, FC, BZ_hat, BZ, Tcin, Tcout_ahead, F_DNA, F_SNA, F_SSA, paras, vars)

    return np.concatenate([Tb_next.flatten(), BETA.flatten(), CPV.flatten()
                           , C.flatten(), B.flatten(), A.flatten()
                           , BZ_hat_next.flatten(), FC_hat_next.flatten()]), Qr, Qc, Qsh

def MeansurementFunc(state):
    '''
    Meansurement function for UKF
    '''
    num = 7
    TR = state[:num][np.newaxis, :]
    return TR.flatten()

def smooth(x, window_len, window="hamming"):
    # Smoothing signals
    s=np.r_[x[window_len:0:-1],x]
    w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='valid')[:len(x)]
    return y


def initialize_KF(InintialState):
    # Initialize the Kalman filter
    dt = 1

    # create sigma points to use in the filter. This is standard for Gaussian processes
    points = MerweScaledSigmaPoints(len(InintialState), alpha=1e-3, beta=2., kappa=0)    
    
    kf = UnscentedKalmanFilter(dim_x=len(InintialState), dim_z=7, dt=dt, fx=TransitionFunc, hx=MeansurementFunc, points=points)
    kf.x = InintialState # initial state
    kf.P = np.identity(56)

    # process error covariance and measurement error covariance
    kf.R = np.diag([])
    kf.Q = np.diag([])
    return kf

def SQP(vars, state_now, P_now):
    '''
    Quadratic programming
    '''
    res = np.dot((vars - state_now).T, np.dot(P_now, (vars - state_now)))
    return res

AddDim = lambda x: x[np.newaxis, :]

def KalmanUpdatePredict(kf, df_data, flag_StopUKF):
    '''
    Predict Multistep temperatures
    Input:
    kf: Kalman filter object
    df_data: a dataframe made up of data slices at timestep t and t-1
    flag_StopUKF: flag for determining whether to use Kalman filter to update the model parameters 
    Output:
    kf: Updated Kalman filter object
    arr_multistep: multistep predictions for temperatures of the seven reactor
    parameters: updated parameters
    arr_Qr, arr_Qc, arr_Qsh: estimated heat 
    '''
    # Preprocess
    # df_data.loc[:, "BZ"] = smooth(df_data["BZ"].values, 10, window="hamming")
    # for Fc in list_Fc:
    #     df_data.loc[:, Fc] = smooth(df_data[Fc].values, 5, window="hamming")
    # for F in list_DNA+list_SNA+list_SSA:
    #     df_data.loc[:, F] = smooth(df_data[F].values, 10, window="hamming") 

    # Lower Bound to flowrate
    df_data.loc[:, list_Fc+list_DNA+list_SNA+list_SSA+list_BZ] = df_data.loc[:, list_Fc+list_DNA+list_SNA+list_SSA+list_BZ] \
                    .applymap(lambda x: x if x>0 else 0.01)       

    arr_multistep  = np.zeros((len_win_output, 7))
    arr_Qr  = np.zeros((len_win_output, 7))
    arr_Qc  = np.zeros((len_win_output, 7))
    arr_Qsh  = np.zeros((len_win_output, 7))

    if not flag_StopUKF:
        # Predict
        kf.predict(FC=AddDim(df_data.loc[df_data.index[-2], list_Fc].values.astype(float))
                , BZ=AddDim(df_data.loc[df_data.index[-2], list_BZ].values.astype(float))
                , Tcin=AddDim(df_data.loc[df_data.index[-2], list_Tcin].values.astype(float))
                , Tcout_ahead=AddDim(df_data.loc[df_data.index[-2], list_Tcout].values.astype(float))
                , F_DNA=AddDim(df_data.loc[df_data.index[-2], list_DNA].values.astype(float))
                , F_SNA=AddDim(df_data.loc[df_data.index[-2], list_SNA].values.astype(float))
                , F_SSA=AddDim(df_data.loc[df_data.index[-2], list_SSA].values.astype(float))
                , paras=paras)

        # Correct
        measurements = df_data.loc[df_data.index[-1], list_TR].values
        kf.update(z=measurements)    

        # Parameter constraints
        state_now, P_now = kf.x.copy(), kf.P.copy()
        if np.any(kf.x[14:21] > np.array([20,30,30,30,30,30,30])) | np.any(kf.x[14:21] < np.array([8,8,8,8,8,8,8])):
            # Fix range for TB, FC_hat and BZ_hat
            opt_para = optimize.minimize(SQP, state_now, args=(state_now, P_now)
                                    , method='SLSQP'
                                    , tol=1e-6
                                    , bounds=tuple(list_con)
                                    #  , options={"disp":True}
                                    ).x[7:-14]
            kf.x = np.concatenate([kf.x[:7], opt_para, kf.x[-14:]])
    
    # Multistep prediction
    state = kf.x.copy()
    state[:7] = df_data.loc[df_data.index[-1], list_TR].values
    for k in range(len_win_output):
        # Keep constant
        state, Qr, Qc, Qsh = TransitionFunc_plus(state.astype(float), dt=1
                                , FC=AddDim(df_data.loc[df_data.index[-1], list_Fc].values.astype(float))
                                , BZ=AddDim(df_data.loc[df_data.index[-1], list_BZ].values.astype(float))
                                , Tcin=AddDim(df_data.loc[df_data.index[-1], list_Tcin].values.astype(float))
                                , Tcout_ahead=AddDim(df_data.loc[df_data.index[-1], list_Tcout].values.astype(float))
                                , F_DNA=AddDim(df_data.loc[df_data.index[-1], list_DNA].values.astype(float))
                                , F_SNA=AddDim(df_data.loc[df_data.index[-1], list_SNA].values.astype(float))
                                , F_SSA=AddDim(df_data.loc[df_data.index[-1], list_SSA].values.astype(float))
                                , paras=paras)     

        arr_multistep[k, :] = state[:7].copy()        
        arr_Qr[k, :], arr_Qc[k, :], arr_Qsh[k, :] = Qr, Qc, Qsh

    return kf, arr_multistep, state[7:7+35].copy(), arr_Qr, arr_Qc, arr_Qsh


df_data = pd.read_csv("改造数据.csv")
df_data.columns = ["Time"] + df_data.columns.tolist()[1:]
df_data.loc[:, "Time"] = pd.to_datetime(df_data.loc[:, "Time"])
df_data.set_index("Time", drop=True, inplace=True)

# number of steps for prediction
len_win_output = 5
# number of past samples
len_back_input = 2
# number of samples
n_sam = len(df_data)


if __name__ == "__main__":
    # creat a log file to sace running information
    with open('log.txt', 'a') as f:
        current_time = datetime.now().strftime("%y/%m/%d %H:%M:%S")
        f.write("\n")
        f.write("\n" + current_time + "Start...")

    # Initialize Kalman filter
    InintialState = np.concatenate([df_data[list_TR].iloc[0, :]
                                    , np.array(BETA), np.array(CPV), np.array(C), np.array(B), np.array(A)
                                    , np.tile(df_data[list_BZ].iloc[0, :].values, [7])
                                    , df_data[list_Fc].iloc[0, :]
                                    ])
    kf = initialize_KF(InintialState=InintialState)

    # Simulate the online application of the UKF-FP model
    list_pred, list_real, list_FCall = [], [], []
    list_para, list_Qr, list_Qc, list_Qsh = [], [], [], []
    for idx in tqdm(range(n_sam - len_back_input)):
        df_win = df_data.iloc[idx:(idx + len_back_input)].copy()

        # determine whether to update the model parameters 
        Flowrate_new_BZ = df_win.loc[:, ["BZ"]].values
        Flowrate_new_fc = df_win.loc[:, list_Fc].values
        flag_StopUKF = np.any((Flowrate_new_BZ<=100) | (Flowrate_new_fc<=0.01))

        if flag_StopUKF:
            with open('log.txt', 'a') as f:
                f.write("\n" + f"{df_win.index[-1]}, UKF disabled...")

        try:
            kf, arr_multistep_, arr_para, arr_Qr, arr_Qc, arr_Qsh = KalmanUpdatePredict(kf, df_win, flag_StopUKF)
        except:
            with open('log.txt', 'a') as f:
                f.write("\n" + f"\n{df_win.index[-1]}, Initialize Kalman fiter...")            
            kf = initialize_KF(InintialState=InintialState)
            kf, arr_multistep_, arr_para, arr_Qr, arr_Qc, arr_Qsh  = KalmanUpdatePredict(kf, df_win, flag_StopUKF)
    
    #     arr_multistep_[arr_multistep_<=10] = 10
    #     arr_multistep_[arr_multistep_>=100] = 100
    #     list_pred.append(arr_multistep_[np.newaxis, :, :])

    #     list_real.append(df_win.loc[df_win.index[-1], list_TR].values[np.newaxis, :])
    #     list_FCall.append(df_win.loc[df_win.index[-1], list_Fc].values[np.newaxis, :])

    #     list_para.append(arr_para[np.newaxis, :])
    #     list_Qc.append(arr_Qc[np.newaxis, :, :])
    #     list_Qr.append(arr_Qr[np.newaxis, :, :])
    #     list_Qsh.append(arr_Qsh[np.newaxis, :, :])
    
    # pred = np.concatenate(list_pred, axis=0)
    # real = np.concatenate(list_real, axis=0)
    # Fc_all = np.concatenate(list_FCall, axis=0)
    # model_para = np.concatenate(list_para, axis=0)
    
    # Qc = np.concatenate(list_Qc, axis=0)
    # Qr = np.concatenate(list_Qr, axis=0)
    # Qsh = np.concatenate(list_Qsh, axis=0)
    # arr_beta, arr_CPV, arr_C, arr_B, arr_A = model_para[:, :7], model_para[:, 7:14], model_para[:, 14:21], model_para[:, 21:28], model_para[:, 28:]