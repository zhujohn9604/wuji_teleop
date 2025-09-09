import math

MAX_ANGLES = 4
MAX_FINGERS = 6
NUM_FINGERS = 5
EXTRA_MOTORS = 1

LOGIC_POS_LIMIT = 65535
ENCODER_CPR = 128
GEAR_RATIO = 19.2

PI = 3.1415926535897932384626433832795
HALF_PI  = 1.5707963267948966192313216916397

ANGLE_SCALE = 100
MIN_POSITION = 0
MAX_POSITION = 65535
MIN_THUMBROOT_ANGLE = 0
MAX_THUMBROOT_ANGLE = 90

POS_NAGATIVE_CDF = -1.466

DPRS = [
    1.0, 1.4, 1.4, 1.4, 1.4
]

START_ABS_OFFSET_VALUE = [
    8749, 5530, 5530, 5530, 5530
]

LIMIT_RANGE_VALS =  [
    26500, 31000, 31500, 30000, 30500
]

def DEGREE_TO_RAD(angle):
    return (angle * PI / 180)
def START_LOGIC_OFFSET_VALUE(finger_id):
    return round(LOGIC_POS_LIMIT * START_ABS_OFFSET_VALUE[finger_id] / LIMIT_RANGE_VALS[finger_id])

def NORMALIZATION(value, mean, std):
    return ((value - mean) / std)

def LOGIC_OFFSET_TO_DISTANCE(finger_id, offset):
    return ((offset * DPRS[finger_id]) * LIMIT_RANGE_VALS[finger_id]/ (LOGIC_POS_LIMIT * ENCODER_CPR * GEAR_RATIO))

def DISTANCE_TO_LOGIC_OFFSET(finger_id, distance):
    return ((MAX_POSITION * ENCODER_CPR * GEAR_RATIO * distance) / (DPRS[finger_id] * LIMIT_RANGE_VALS[finger_id]))

def MAP(value, in_min, in_max, out_min, out_max):
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def CLAMP(value, min_value, max_value):
    return max(min(value, max_value), min_value)

def ANGLE_COS(L_a, L_b, L_c):
    return math.acos((L_b * L_b + L_c * L_c - L_a * L_a) / (2 * L_b * L_c))         #L_a是对边，L_b、L_c是邻边

def ANGLE_SIN(angle_a, L_a, L_b):
    return math.asin((L_b * math.sin(angle_a)) / L_a)                                    # L_b是所求角对边

def LENGTH_COS(angle, L_b, L_c):
    return math.sqrt(L_b *L_b + L_c * L_c - 2 * L_b * L_c * math.cos(angle))             # L_b 和 L_c 是所求角邻边

def LENGTH_SIN(angle_a, L_a, angle_b):
    return ((math.sin(angle_b) * L_a) / math.sin(angle_a))                               #angle_a是L_a的对角, angle_b是所求边的对角

#校验位参数:
ThumbParams = [
    #L_BP0, L_OP,  L_AB,  L_OA,  ∠OAR, L_CT0, L_OT,  ∠EOQ,  L_OE, L_CD,  L_ED,  ∠DEF,  L_EF,    XE, L_DF,   ∠EFD, L_CG
    11.48, 0.48, 10.98, 10.00, 0.5983, 43.78, 2.08, 0.2155, 52.52, 9.90, 11.30, 1.2976, 27.54, 51.30, 26.8, 0.4184, 13.31 # finger_0
]

FingerParams = [
    #X_A0, L_AP,  L_AC,  L_OC,  ∠BOQ,  ∠COD,  ∠EDF,  L_OD,  L_CD, L_OB, L_DE,  L_BE,  L_DF, 3rdJoint(°)
    [12.70, 8.60, 19.63, 14.25, 0.5353, 0.7624, 0.8496, 35.92, 27.44, 8.63, 7.69, 34.14, 50.23, 146.1], # finger_1
    [13.17, 8.60, 19.63, 14.31, 0.5353, 0.7763, 0.8496, 41.01, 32.39, 8.63, 7.69, 39.48, 50.23, 146.1], # finger_2
    [13.00, 8.60, 19.63, 14.25, 0.5353, 0.7624, 0.8496, 35.92, 27.44, 8.63, 7.69, 34.14, 50.23, 146.1], # finger_3
    [13.15, 8.60, 19.63, 14.27, 0.5353, 0.7502, 0.8141, 31.93, 23.59, 8.63, 7.53, 30.85, 35.53, 146.1]  # finger_4
]

#关节角度限制
Max_Joint_Angle= [
  #joint0, joint1, joint2, joint3
  [  36.76, 167.74, 191.13,  36.34 ],# finger0
  [ 180.91, 169.38, 146.10, 174.72 ],# finger1
  [ 178.52, 171.81, 146.10, 174.01 ],# finger2
  [ 179.04, 166.97, 146.10, 171.44 ],# finger3
  [ 177.33, 169.38, 146.10, 171.73 ],# finger4
  [  92.00,  92.00,  92.00,  92.00 ] # finger5
]

Min_Joint_Angle= [
  #joint0, joint1, joint2, joint3
  [-12.69, 143.96, 159.20, -32.06 ],# finger0
  [ 91.00,  64.11, 146.10,  18.20 ],# finger1
  [ 89.58,  68.00, 146.10,  25.06 ],# finger2
  [ 89.47,  61.74, 146.10,  14.81 ],# finger3
  [ 88.05,  63.39, 146.10,  24.81 ],# finger4
  [ -2.00,  -2.00,  -2.00,  -2.00 ] # finger5
]

#通过位置推导关节角度推导位置
#采用五阶多项式拟合，参数是在MATLAB的曲线拟合工具中求得的。
ThumbPosToAngle_PolyCoefs = [
    #      p0,       p1,      p2,     p3,     p4,    p5,   mean,  std
    [-0.06277, -0.14240, -0.27210, -0.9211,  -12.9, 16.12, 2.21, 3.336], # Thumb_joint0
    [0.009278,  0.01787,  0.06977, -2.6030, -7.165, 163.5, 2.21, 3.336], # Thumb_joint1
    [ 0.01048,  0.03519, -0.08826,  2.8090, -9.049, 166.5, 2.21, 3.336], # Thumb_joint2
    [-0.06371, -0.12510, -0.23050, -0.8751,  -18.5,  5.93, 2.21, 3.336]  # Thumb_joint3
]

FingerPosToAngle_PolyCoefs = [
    #          p0,      p1,      p2,    p3,    p4,    p5,  mean,  std
    [
        [-0.05468, -0.0183, -0.7806, 1.325, -23.14, 132.2, 6.47, 5.795], # indexFinger_joint0
        [-0.15440, -0.1848, -1.2850, 2.768, -25.14, 110.4, 6.47, 5.795], # indexFinger_joint1
        [ 0.00000,  0.0000,  0.0000, 0.000,   0.00, 146.1, 6.47, 5.795], # indexFinger_joint2(fixed angle)
        [-0.19720, -0.2277, -1.5390, 2.666, -38.79, 90.77, 6.47, 5.795], # indexFinger_joint3
    ],

    [
        [-0.06417, -0.04234, -0.7546, 1.141, -22.85, 131.1, 6.47, 5.795], # middleFinger_joint0
        [-0.15720, -0.17790, -1.1640, 2.414, -25.05, 114.5, 6.47, 5.795], # middleFinger_joint1
        [ 0.00000,  0.00000,  0.0000, 0.000,  0.000, 146.1, 6.47, 5.795], # middleFinger_joint2(fixed angle)
        [-0.17960, -0.19620, -1.4050, 2.311, -37.17, 94.58, 6.47, 5.795], # middleFinger_joint3
    ],

    [
        [-0.06154, -0.03494, -0.7673, 1.208, -23.02, 131.0, 6.47, 5.795], # thirdFinger_joint0
        [-0.20270, -0.24340, -1.1980, 2.600, -24.92, 109.1, 6.47, 5.795], # thirdFinger_joint1
        [ 0.00000,  0.00000,  0.0000,  0.00,  0.000, 146.1, 6.47, 5.795], # thirdFinger_joint2(fixed angle)
        [-0.24610, -0.29980, -1.4600, 2.463, -38.59, 88.76, 6.47, 5.795], # thirdFinger_joint3
    ],

    [
        [-0.06489, -0.04309, -0.7589, 1.150, -22.93, 129.7, 6.47, 5.795], # littleFinger_joint0
        [-0.20360, 0.017360, -1.3550, 2.994, -24.69, 107.4, 6.47, 5.795], # littleFinger_joint1
        [ 0.00000, 0.000000, 0.00000, 0.000,  0.000, 146.1, 6.47, 5.795], # littleFinger_joint2(fixed angle)
        [-0.18860, -0.06764, -1.4770, 2.634, -36.29, 91.14, 6.47, 5.795], # littleFinger_joint3
    ]
]

#通过关节角度推导位置
#采用四阶多项式拟合，参数是在MATLAB的曲线拟合工具中求得的。
ThumbAngleToPos_PolyCoefs = [
    #     p0,      p1,     p2,     p3,    p4,    mean,   std
    [0.006733, 0.05529, -0.3055, -3.495, 2.515, 14.94, 13.69],  # thumb_joint0
    [0.0     , 0.0    ,  0.0   ,  0.0  , 0.0  ,  0.0 ,  0.0 ],  # thumb_joint1
    [0.0     , 0.0    ,  0.0   ,  0.0  , 0.0  ,  0.0 ,  0.0 ],  # thumb_joint2
    [-0.01311, 0.09194, -0.319, -4.053, 3.339, -0.4461, 23.30]  # thumb_joint3
]

FingerAngleToPos_PolyCoefs = [
    [
        [-0.04835, 0.1884, 0.3872, -6.169, 6.151, 133.5, 24.79], # indexFinger_joint0
        [-0.12460, 0.3022, 0.7211, -6.410, 5.935, 112.8, 28.15], # indexFinger_joint1
        [ 0.0    , 0.0   , 0.0   ,  0.0  , 0.0  ,   0.0,  0.0 ], # indexFinger_joint2 (固定角度)
        [-0.07643, 0.2559, 0.4424, -6.296, 6.144, 93.03, 42.39], # indexFinger_joint3
    ],
    [
        [-0.04590, 0.1976, 0.3324, -6.179, 6.204, 132.1, 24.48], # middleFinger_joint0
        [-0.10930, 0.2909, 0.6280, -6.378, 6.006, 116.6, 27.84], # middleFinger_joint1
        [ 0.0    , 0.0   , 0.0   ,  0.0  , 0.0  ,   0.0,  0.0 ], # middleFinger_joint2 (固定角度)
        [-0.06831, 0.2463, 0.4001, -6.275, 6.174, 96.54, 40.45], # middleFinger_joint3
    ],
    [
        [-0.04703, 0.1953, 0.3516, -6.177, 6.186, 132.1, 24.67], # thirdFinger_joint0
        [-0.12350, 0.3215, 0.6650, -6.439, 5.994, 111.2, 27.96], # thirdFinger_joint1
        [ 0.0    , 0.0   , 0.0   ,  0.0  , 0.0  ,   0.0,  0.0 ], # thirdFinger_joint2 (固定角度)
        [-0.07468, 0.2684, 0.3946, -6.316, 6.194, 90.69, 42.23], # thirdFinger_joint3
    ],
    [
        [-0.04622, 0.1982, 0.3339, -6.181, 6.203, 130.8, 24.57], # littleFinger_joint0
        [-0.13230, 0.3147, 0.8628, -6.473, 5.790, 110.4, 28.09], # littleFinger_joint1
        [ 0.0    , 0.0   , 0.0   ,  0.0  , 0.0  ,   0.0,  0.0 ], # littleFinger_joint2 (固定角度)
        [-0.07518, 0.2527, 0.4987, -6.301, 6.078, 93.65, 39.77], # littleFinger_joint3
    ]
]

#
#计算大拇指位移到角度
def THUMB_OffsetToAngle(finger_id, offset_to_ref):
    pos = LOGIC_OFFSET_TO_DISTANCE(finger_id, offset_to_ref)

    Angle_Thumb = [0] * MAX_ANGLES#初始化数组

    for joint_id in range(MAX_ANGLES):
        p0 = ThumbPosToAngle_PolyCoefs[joint_id][0]
        p1 = ThumbPosToAngle_PolyCoefs[joint_id][1]
        p2 = ThumbPosToAngle_PolyCoefs[joint_id][2]
        p3 = ThumbPosToAngle_PolyCoefs[joint_id][3]
        p4 = ThumbPosToAngle_PolyCoefs[joint_id][4]
        p5 = ThumbPosToAngle_PolyCoefs[joint_id][5]
        mean = ThumbPosToAngle_PolyCoefs[joint_id][6]
        std = ThumbPosToAngle_PolyCoefs[joint_id][7]

        # 中心化并归一化位置
        pos_norm = NORMALIZATION(pos, mean, std)
        coef = pos_norm

        # 计算角度
        Angle_Thumb[joint_id] = p5
        Angle_Thumb[joint_id] += p4 * coef
        coef *= pos_norm
        Angle_Thumb[joint_id] += p3 * coef
        coef *= pos_norm
        Angle_Thumb[joint_id] += p2 * coef
        coef *= pos_norm
        Angle_Thumb[joint_id] += p1 * coef
        coef *= pos_norm
        Angle_Thumb[joint_id] += p0 * coef

    return Angle_Thumb


#
#计算除大拇指外其他手指的位移到角度
def FINGER_OffsetToAngle(finger_id, offset_to_ref):
    pos = LOGIC_OFFSET_TO_DISTANCE(finger_id, offset_to_ref)

    finger_id -= 1 #匹配数组序号

    Angle_Finger = [[0] * MAX_ANGLES for _ in range(MAX_FINGERS)] #初始化数组

    for joint_id in range(MAX_ANGLES):
        p0 = FingerPosToAngle_PolyCoefs[finger_id][joint_id][0]
        p1 = FingerPosToAngle_PolyCoefs[finger_id][joint_id][1]
        p2 = FingerPosToAngle_PolyCoefs[finger_id][joint_id][2]
        p3 = FingerPosToAngle_PolyCoefs[finger_id][joint_id][3]
        p4 = FingerPosToAngle_PolyCoefs[finger_id][joint_id][4]
        p5 = FingerPosToAngle_PolyCoefs[finger_id][joint_id][5]
        mean = FingerPosToAngle_PolyCoefs[finger_id][joint_id][6]
        std = FingerPosToAngle_PolyCoefs[finger_id][joint_id][7]

        # 中心化并归一化位置
        pos_norm = NORMALIZATION(pos, mean, std)
        coef = pos_norm

        # 计算角度
        Angle_Finger[finger_id][joint_id] = p5
        Angle_Finger[finger_id][joint_id] += p4 * coef
        coef *= pos_norm
        Angle_Finger[finger_id][joint_id] += p3 * coef
        coef *= pos_norm
        Angle_Finger[finger_id][joint_id] += p2 * coef
        coef *= pos_norm
        Angle_Finger[finger_id][joint_id] += p1 * coef
        coef *= pos_norm
        Angle_Finger[finger_id][joint_id] += p0 * coef

    return Angle_Finger[finger_id]


#
#推导手指位移到角度
def HAND_FingerPosToAngle(finger_id, pos):
    angles = [0] * MAX_ANGLES #初始化数组

    if finger_id >= NUM_FINGERS + EXTRA_MOTORS:
        return # 无效id
    elif finger_id >= NUM_FINGERS:
        angles = MAP(pos, MIN_POSITION, MAX_POSITION, 0, 90)

    else:
        offset_to_ref = pos - START_LOGIC_OFFSET_VALUE(finger_id)

        if finger_id == 0:
            angles = THUMB_OffsetToAngle(finger_id, offset_to_ref)
        else:
            angles = FINGER_OffsetToAngle(finger_id, offset_to_ref)

    return angles


#
#通过关节角度推导大拇指位置
def THUMB_AngleToOffset(joint_id, Angle_Thumb):
    pos_in_mm = 0.0
    angle = Angle_Thumb

    if joint_id == 0 or joint_id == 3:
        p0 = ThumbAngleToPos_PolyCoefs[joint_id][0]
        p1 = ThumbAngleToPos_PolyCoefs[joint_id][1]
        p2 = ThumbAngleToPos_PolyCoefs[joint_id][2]
        p3 = ThumbAngleToPos_PolyCoefs[joint_id][3]
        p4 = ThumbAngleToPos_PolyCoefs[joint_id][4]
        mean = ThumbAngleToPos_PolyCoefs[joint_id][5]
        std = ThumbAngleToPos_PolyCoefs[joint_id][6]
        angle_norm = NORMALIZATION(angle, mean, std)

        coef = angle_norm

        pos_in_mm = p4
        pos_in_mm += p3 * coef
        coef *= angle_norm
        pos_in_mm += p2 * coef
        coef *= angle_norm
        pos_in_mm += p1 * coef
        coef *= angle_norm
        pos_in_mm += p0 * coef

    else:
        L_BP0 = ThumbParams[0]
        L_OP = ThumbParams[1]
        L_AB = ThumbParams[2]
        L_OA = ThumbParams[3]
        Angle_OAR = ThumbParams[4]
        L_CT0 = ThumbParams[5]
        L_CD = ThumbParams[9]
        L_ED = ThumbParams[10]
        L_EF = ThumbParams[12]
        XE = ThumbParams[13]
        L_DF = ThumbParams[14]
        Angle_EFD = ThumbParams[15]
        L_CG = ThumbParams[16]

        if joint_id == 1:
            Angle_DCT = DEGREE_TO_RAD(angle)
            Angle_DCG = Angle_DCT - HALF_PI;
            L_DG = LENGTH_COS(Angle_DCG, L_CG, L_CD)
            Angle_DGC = ANGLE_COS(L_CD, L_CG, L_DG)
            Angle_DGE = HALF_PI - Angle_DGC
            Angle_GED = ANGLE_SIN(Angle_DGE, L_ED, L_DG)
            Angle_GDE = PI - Angle_GED - Angle_DGE
            L_EG = LENGTH_SIN(Angle_DGE, L_ED, Angle_GDE)
            L_CT = XE - L_EG
            pos_in_mm = L_CT - L_CT0

        elif joint_id == 2:
            Angle_CDF = DEGREE_TO_RAD(angle)

            if Angle_CDF > PI:
                Angle_CDF = 2 * PI - Angle_CDF
                L_CF = LENGTH_COS(Angle_CDF, L_CD, L_DF)
                Angle_DFC = ANGLE_COS(L_CD, L_DF, L_CF)
                Angle_EFC = Angle_EFD + Angle_DFC

            elif Angle_CDF == PI:
                L_CF = L_CD + L_DF
                Angle_EFC = Angle_EFD

            else:
                L_CF = LENGTH_COS(Angle_CDF, L_CD, L_DF)
                Angle_DFC = ANGLE_COS(L_CD, L_DF, L_CF)
                Angle_EFC = Angle_EFD - Angle_DFC

            L_EC = LENGTH_COS(Angle_EFC, L_EF, L_CF)
            L_EG = math.sqrt(L_EC * L_EC - L_CG * L_CG)

            L_CT = XE - L_EG
            pos_in_mm = L_CT - L_CT0

    return pos_in_mm


#
#计算除大拇指外的手指角度到位移
def FINGER_AngleToOffset(finger_id, joint_id, Angle_Finger):
    pos_in_mm = 0.0
    angle = Angle_Finger
    finger_id -= 1

    if joint_id == 2:
        return 0
        # 固定角度，无法写
    else:
        p0 = FingerAngleToPos_PolyCoefs[finger_id][joint_id][0]
        p1 = FingerAngleToPos_PolyCoefs[finger_id][joint_id][1]
        p2 = FingerAngleToPos_PolyCoefs[finger_id][joint_id][2]
        p3 = FingerAngleToPos_PolyCoefs[finger_id][joint_id][3]
        p4 = FingerAngleToPos_PolyCoefs[finger_id][joint_id][4]
        mean = FingerAngleToPos_PolyCoefs[finger_id][joint_id][5]
        std = FingerAngleToPos_PolyCoefs[finger_id][joint_id][6]

        angle_norm = NORMALIZATION(angle, mean, std)
        coef = angle_norm

        pos_in_mm = p4
        pos_in_mm += p3 * coef
        coef *= angle_norm
        pos_in_mm += p2 * coef
        coef *= angle_norm
        pos_in_mm += p1 * coef
        coef *= angle_norm
        pos_in_mm += p0 * coef

    return pos_in_mm


#
#计算手指角度到位移
def HAND_FingerAngleToPos(finger_id, angle_target, joint_id):
    high_limit_angle = Max_Joint_Angle[finger_id][joint_id]
    low_limit_angle = Min_Joint_Angle[finger_id][joint_id]
    angles = CLAMP(angle_target, low_limit_angle, high_limit_angle)

    if finger_id >= NUM_FINGERS + EXTRA_MOTORS:
        #错误ID
        return 0

    elif finger_id >= NUM_FINGERS:
        pos = round(MAP(angles, MIN_THUMBROOT_ANGLE, MAX_THUMBROOT_ANGLE, MIN_POSITION, MAX_POSITION))

    else:
        if finger_id == 0:
            pos_in_mm = THUMB_AngleToOffset(joint_id, angles)
        else:
            pos_in_mm = FINGER_AngleToOffset(finger_id, joint_id, angles)

    pos = round(DISTANCE_TO_LOGIC_OFFSET(finger_id, pos_in_mm)+ START_LOGIC_OFFSET_VALUE(finger_id))

    return pos


#
#计算由输入关节角度到目标关节角度
def calculate_joint_angle(finger_id, src_joint_id, src_angle, dst_joint_id):
    angle_target = [0] * MAX_ANGLES #初始化数组

    if finger_id >= NUM_FINGERS:
        #只支持手指ID 0-4
        return 0
    pos_src = HAND_FingerAngleToPos(finger_id, src_angle, src_joint_id)

    angle_target = HAND_FingerPosToAngle(finger_id, pos_src)

    return angle_target[dst_joint_id]
