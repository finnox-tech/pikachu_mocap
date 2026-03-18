
# Humanoid Bones
00: NOSE
01: LEFT_EYE_INNER
02: LEFT_EYE
03: LEFT_EYE_OUTER
04: RIGHT_EYE_INNER
05: RIGHT_EYE
06: RIGHT_EYE_OUTER
07: LEFT_EAR
08: RIGHT_EAR
09: MOUTH_LEFT
10: MOUTH_RIGHT

11: LEFT_SHOULDER
12: RIGHT_SHOULDER
13: LEFT_ELBOW
14: RIGHT_ELBOW
15: LEFT_WRIST
16: RIGHT_WRIST
17: LEFT_PINKY
18: RIGHT_PINKY
19: LEFT_INDEX
20: RIGHT_INDEX
21: LEFT_THUMB
22: RIGHT_THUMB

23: LEFT_HIP
24: RIGHT_HIP
25: LEFT_KNEE
26: RIGHT_KNEE
27: LEFT_ANKLE
28: RIGHT_ANKLE
29: LEFT_HEEL
30: RIGHT_HEEL
31: LEFT_FOOT_INDEX
32: RIGHT_FOOT_INDEX


# Pikachu Bones
ear.L
ear.R

head
neck

chest
torso
shoulder.L
shoulder.R

upper_arm_fk.L
upper_arm_fk.R
forearm_fk.L
forearm_fk.R
hand_fk.R
hand_fk.L

hips
foot_ik.R
foot_ik.L
toe.R
toe.L

tail


   ┌─────────────────────────────────────────────────┐
   │ 上半部分：列表区域（4/5）                               │
   │  - Bone模式：显示Bone列表（带勾选框、按钮、角度）  │
   │  - URDF模式：显示所有URDF joint的滑动条         │
   │     （带勾选框、标题、滑动条、数值显示）                    │
   ├─────────────────────────────────────────────────┤
   │ 下半部分：joint panel区域（1/5）                        │
   │  - Bone模式：显示选中bone的XYZ三轴滑动条          │
   │  - URDF模式：显示简化的URDF信息              │
   └─────────────────────────────────────────────────┘


部位	URDF joint name	轴	MediaPipe 源
头	head_yaw_joint	Z	HEAD_LINK
头	head_pitch_joint	Y	HEAD_LINK
头	head_roll_joint	X	HEAD_LINK
左臂	left_arm_pitch_joint	Y	LEFT_SHOULDER
左臂	left_arm_roll_joint	X	LEFT_SHOULDER
左臂	left_arm_yaw_joint	Z	LEFT_SHOULDER
左肘	left_elbow_ankle_joint	Y	LEFT_ELBOW
右臂	right_arm_pitch_joint	Y	RIGHT_SHOULDER
右臂	right_arm_roll_joint	X	RIGHT_SHOULDER
右臂	right_arm_yaw_joint	Z	RIGHT_SHOULDER
右肘	right_elbow_ankle_joint	Y	RIGHT_ELBOW
左髋	left_hip_pitch_joint	Y	LEFT_HIP
左髋	left_hip_roll_joint	X	LEFT_HIP
左髋	left_hip_yaw_joint	Z	LEFT_HIP
左膝	left_knee_joint	Y	LEFT_KNEE
左踝	left_ankle_joint	X	LEFT_ANKLE
右髋	right_hip_pitch_joint	Y	RIGHT_HIP
右髋	right_hip_roll_joint	X	RIGHT_HIP
右髋	right_hip_yaw_joint	Z	RIGHT_HIP
右膝	right_knee_joint	Y	RIGHT_KNEE
右踝	right_ankle_joint	X	RIGHT_ANKLE