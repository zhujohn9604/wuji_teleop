MIN_GRIP_WIDTH = 0.0
MAX_GRIP_WIDTH = 0.082  # 8.2cm for robotiq openness


TX_LEFT_RIGHT = [
    [ 1.        ,  0.        ,  0.        ,  0.01      ],
    [ 0.        ,  1.        ,  0.        , -0.927     ],
    [ 0.        ,  0.        ,  1.        , -0.01     ],
    [ 0.        ,  0.        ,  0.        ,  1.        ]
]

POS_DELTA_THRESHOLD = 0.083

WINDOW_SIZE = 16

ACTION_HORIZON = 24

QUASI_STATIC_DROPPED_FRAMES = 0
QUASI_STATIC_KEPT_FRAMES = ACTION_HORIZON
QUASI_STATIC_WAIT_AFTER_TRUNK = 0.0
