#!/usr/bin/env python3
"""render_widowx.py — WidowX 构图渲染工具（自包含，不依赖 src/）。

用途：
    按「env CameraSpec 相同的方式」手动指定相机(target/distance/yaw/pitch/roll/
    fov/resolution)渲染主图，并另存一张左视图(仅改 yaw)用于核对方位。
    场景与 env 完全一致：plane 地面 + 棕薄桌 + 红 cube_small + wx250。

设计约定：
    - 相机指定方式 == src 的 CameraSpec 球坐标：target/distance/yaw/pitch/roll/
      fov/resolution，全部数值由你直接给，脚本不做任何自动定位计算。
    - 每次运行输出一个日期时间戳命名的文件夹（如 20260905_202400/），
      内含两张 PNG：<ts>.png 主图 + side_yaw<角>.png 左视图。
    - 默认纯运动学构图（resetJointState 直设关节、不推物理），结果确定可复现；
      需要碰撞/物理检查时加 --settle。
    - wx250.urdf 关节布局：0 waist / 1 shoulder / 2 elbow / 3 wrist_angle /
      4 wrist_rotate（5 臂）+ 9/10 left/right_finger（夹爪两指）。
    - 默认目标姿态 (0,0,0,-pi/2,0)：第 3 段（wrist_angle 之后的腕段）折 -90° 垂直
      向下，夹爪随腕段自然下垂。⚠️ 实测 +pi/2 会把腕段折向上方，朝下取 -pi/2。
      最终以左视图（水平侧视）确认方向。

输出目录：data/scriptData/cameraTest/<日期时间戳>/（可用 --out-dir 覆盖根目录）。

用法示例：
    # 默认（腕段折下 + env 式相机）→ cameraTest/<ts>/<ts>.png + side_yaw0.png
    python scripts/camera/render_widowx.py

    # 相机照 env 方式调：target/dist/yaw/pitch 直接给数值
    python scripts/camera/render_widowx.py \
        --cam-target 0.30,0,0.15 --cam-dist 0.95 --cam-yaw 90 --cam-pitch -60

    # 侧视图从另一侧拍（yaw=180 → +y 右侧），看方位用
    python scripts/camera/render_widowx.py --side-yaw 180

    # 打印 URDF 真实关节表（名称/索引/限位）后退出
    python scripts/camera/render_widowx.py --print-joints

    # 物理落稳后再渲染（cube 从初始 z 自由下落）
    python scripts/camera/render_widowx.py --settle --cube-pos 0.30,0,0.15
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data  # plane.urdf / cube_small.urdf（pybullet 自带数据，无下载）
from PIL import Image

PROJ = Path(__file__).resolve().parents[2]
DEFAULT_URDF = PROJ / "robot" / "widowx" / "wx250.urdf"
MESH_PARENT = PROJ / "robot"  # urdf 内 package://widowx/... 在此父目录下可解析
DEFAULT_OUT_DIR = PROJ / "data" / "scriptData" / "cameraTest"

# wx250.urdf 关节索引（与 src/env/robot/widowx/widowx_robot.py 一致，勿凭记忆改）
ARM_JOINT_INDICES = (0, 1, 2, 3, 4)
GRIPPER_JOINT_INDICES = (9, 10)
ARM_JOINT_NAMES = ("waist", "shoulder", "elbow", "wrist_angle", "wrist_rotate")

# 场景几何（脚本自定，与 src/env/pybullet_env.py 的默认桌面同构）
TABLE_CENTER = (0.5, 0.0)     # 桌面中心 x/y
TABLE_HALF = 0.30             # 桌面半宽(总 0.6x0.6)
TABLE_THICKNESS = 0.04        # 桌面厚度
CUBE_HALF = 0.025             # cube_small 半边长(总 0.05)
# 颜色与 src/env/pybullet_env.py 对齐：cube 红(1,0,0)、桌面棕(0.55,0.42,0.30)
CUBE_RGBA = (1.0, 0.0, 0.0, 1.0)       # env color_map["red"]
TABLE_RGBA = (0.55, 0.42, 0.30, 1.0)   # env _create_table rgbaColor

# 默认目标姿态：第 3 段（腕段 wrist_angle）折 -90° 垂直向下，夹爪随腕段自然下垂。
# ⚠️ 实测 +π/2 会把腕段折向上方（urdf elbow rpy 翻转），朝下需取 -π/2。
DEFAULT_JOINT = (0.0, 0.0, 0.0, -np.pi / 2.0, 0.0)


def parse_vec(text: str, name: str, n: int | None = None) -> list[float]:
    """解析 "a,b,c"（容忍空格分隔）为 float 列表，可选校验长度。"""
    parts = [x for x in text.replace(",", " ").split() if x]
    vals = [float(x) for x in parts]
    if n is not None and len(vals) != n:
        raise SystemExit(f"--{name} 需要 {n} 个数，得到 {len(vals)}: {text!r}")
    return vals


def parse_res(text: str) -> tuple[int, int]:
    w, h = text.lower().replace("x", ",").split(",")
    return int(w), int(h)


def dump_joints(robot_id: int) -> None:
    """打印 URDF 全部关节表（名称/类型/限位/子 link），用于精确核对索引。"""
    print("[joints] idx | type | name | lower..upper | velocity | childLink")
    for i in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, i)
        jname, jtype = info[1].decode(), info[2]
        lower, upper = info[8], info[9]
        vel = info[11]
        child = info[12].decode()
        lo = f"{lower: .6f}" if lower < 1e6 else "     -inf"
        hi = f"{upper: .6f}" if upper < 1e6 else "      inf"
        print(f"  {i:>3}  | {jtype:>3} | {jname:<16} | {lo} .. {hi} | {vel:>5.2f} | {child}")


def find_link_index(robot_id: int, name_keyword: str) -> int:
    """按子 link 名关键词返回 link index（link index == 其父关节 index）。"""
    for i in range(p.getNumJoints(robot_id)):
        child = p.getJointInfo(robot_id, i)[12].decode()
        if name_keyword in child:
            return i
    raise SystemExit(f"[err] URDF 中找不到含 {name_keyword!r} 的 link")


def build_scene(
    table_top_z: float,
    cube_pos: tuple[float, float, float],
) -> tuple[int, int, int]:
    """按 env 场景逻辑建景：plane 地面 + 棕薄桌 + 红 cube_small(URDF)。

    对齐 src/env/pybullet_env.py.reset()：
      - loadURDF("plane.urdf") 灰色大地面
      - 桌面 = 静态 box，rgbaColor=(0.55,0.42,0.30) 棕，顶面 z=table_top_z
      - cube = loadURDF("cube_small.urdf") + changeVisualShape 红
    返回 (plane_id, table_id, cube_id)。
    """
    plane_id = p.loadURDF("plane.urdf")
    # 桌面（悬浮薄板，env 同款：顶面 z=table_top_z，中心 TABLE_CENTER）
    table_half_z = TABLE_THICKNESS / 2.0
    table_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=(TABLE_HALF, TABLE_HALF, table_half_z))
    table_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=(TABLE_HALF, TABLE_HALF, table_half_z),
                                    rgbaColor=TABLE_RGBA)
    table_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=table_col,
        baseVisualShapeIndex=table_vis,
        basePosition=(TABLE_CENTER[0], TABLE_CENTER[1], table_top_z - table_half_z),
    )
    # cube（env 逻辑：cube_small.urdf 载入后改红）
    cube_id = p.loadURDF("cube_small.urdf", basePosition=cube_pos)
    p.changeVisualShape(cube_id, -1, rgbaColor=CUBE_RGBA)
    p.changeDynamics(cube_id, -1, lateralFriction=0.8, spinningFriction=0.05)
    return plane_id, table_id, cube_id


def set_robot_pose(robot_id: int, joint_q: list[float], gripper_open: bool) -> list[float]:
    """直设 5 臂关节角 + 夹爪两指（不推物理）。返回两指实际目标位置。"""
    for idx, q in zip(ARM_JOINT_INDICES, joint_q):
        p.resetJointState(robot_id, idx, targetValue=q, targetVelocity=0.0)
    # 夹爪限位动态读取（left_finger: [0.015, 0.037]，right 对称负区间）
    limits = []
    for idx in GRIPPER_JOINT_INDICES:
        info = p.getJointInfo(robot_id, idx)
        limits.append((float(info[8]), float(info[9])))
    # 开 = 两指外极限；闭 = 两指内极限
    if gripper_open:
        targets = [limits[0][1], limits[1][0]]
    else:
        targets = [limits[0][0], limits[1][1]]
    for idx, tgt in zip(GRIPPER_JOINT_INDICES, targets):
        p.resetJointState(robot_id, idx, targetValue=tgt, targetVelocity=0.0)
    return targets


def settle(robot_id: int, cube_id: int, max_steps: int = 2400) -> int:
    """物理落稳：重力 + 5 臂关节 POSITION_CONTROL 锁定当前位置 + cube 下落。"""
    p.setGravity(0, 0, -9.8)
    # 读当前臂角做伺服目标，防止臂自由下坠
    cur = [p.getJointState(robot_id, i)[0] for i in ARM_JOINT_INDICES]
    p.setJointMotorControlArray(
        robot_id, list(ARM_JOINT_INDICES), p.POSITION_CONTROL,
        targetPositions=cur, forces=[500.0] * 5,
    )
    cube_z = p.getBasePositionAndOrientation(cube_id)[0][2]
    stable = 0
    for _ in range(max_steps):
        p.stepSimulation()
        z = p.getBasePositionAndOrientation(cube_id)[0][2]
        if abs(z - cube_z) < 1e-5:
            stable += 1
            if stable >= 60:
                break
        else:
            stable = 0
        cube_z = z
    p.setGravity(0, 0, 0)
    return stable >= 60


def eye_from_view_matrix(view_matrix: list[float]) -> np.ndarray:
    """从列主序 viewMatrix 反解相机 eye 世界坐标（用于回显/记录）。"""
    m = np.array(view_matrix, dtype=float).reshape(4, 4, order="F")
    R, t = m[:3, :3], m[:3, 3]
    return -R.T @ t


def wrist_segment_direction(robot_id: int) -> tuple[np.ndarray, float]:
    """腕段延伸方向单位向量（wrist_link 局部 x）与 -z(垂直向下) 的夹角 rad。

    腕段 = wrist_angle 之后的第 3 段臂（用户所述「腕段垂直朝下」指这一段）。
    0 rad = 正好垂直向下；方向正负以左视图为准校验。
    """
    link_idx = find_link_index(robot_id, "wrist_link")
    pos, orn = p.getLinkState(robot_id, link_idx, computeForwardKinematics=1)[:2]
    R = np.array(p.getMatrixFromQuaternion(orn), dtype=float).reshape(3, 3)
    x_dir = R @ np.array([1.0, 0.0, 0.0])  # 腕段延伸方向(link 局部 x)
    z_up = np.array([0.0, 0.0, 1.0])
    ang = float(np.arccos(np.clip(np.dot(x_dir, -z_up), -1.0, 1.0)))
    return x_dir, ang


def lowest_z(robot_id: int, keywords: tuple[str, ...]) -> float:
    """计算若干 link 的 mesh AABB 最低 z（≈夹爪指端最低点）。"""
    zmin = float("inf")
    for kw in keywords:
        link = find_link_index(robot_id, kw)
        aabb = p.getAABB(robot_id, link)
        zmin = min(zmin, float(aabb[0][2]))
    return zmin


def render_one(
    robot_id: int,
    out_png: Path,
    camera_kwargs: dict,
    resolution: tuple[int, int],
    renderer: int,
) -> dict:
    """按相机参数渲染一张并保存；返回 {eye, view, proj} 诊断信息。"""
    width, height = resolution
    if "eye" in camera_kwargs:
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_kwargs["eye"],
            cameraTargetPosition=camera_kwargs["look_at"],
            cameraUpVector=[0.0, 0.0, 1.0],
        )
    else:
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=list(camera_kwargs["target"]),
            distance=camera_kwargs["distance"],
            yaw=camera_kwargs["yaw"],
            pitch=camera_kwargs["pitch"],
            roll=camera_kwargs.get("roll", 0.0),
            upAxisIndex=2,
        )
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=camera_kwargs["fov"], aspect=width / height, nearVal=0.01, farVal=100.0,
    )
    _, _, px, _, _ = p.getCameraImage(
        width, height,
        viewMatrix=view_matrix, projectionMatrix=proj_matrix,
        renderer=renderer,
    )
    rgb = np.array(px, dtype=np.uint8).reshape((height, width, 4))[:, :, :3]
    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(out_png)
    return {
        "eye_world": [round(v, 4) for v in eye_from_view_matrix(view_matrix).tolist()],
        "png": str(out_png),
    }


def resolve_renderer(value: str) -> int:
    if value == "gpu":
        return p.ER_BULLET_HARDWARE_OPENGL
    if value == "cpu":
        return p.ER_TINY_RENDERER
    # auto：Mac 无 GPU 环境用 CPU tiny 渲染器（构图几何可辨即可）
    return p.ER_TINY_RENDERER


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="WidowX 构图渲染（自包含，不依赖 src/）。默认纯运动学摆位，不推物理。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- 场景 ---
    ap.add_argument("--urdf", default=str(DEFAULT_URDF), help="wx250 URDF 路径")
    ap.add_argument("--table-z", type=float, default=0.10,
                    help="桌面顶面高度 z（目标参数：0.10）")
    ap.add_argument("--cube-pos", default=None,
                    help="cube 世界坐标 'x,y[,z]'；z 省略则贴桌顶(table_z+半高)。"
                         "默认 x=0.30,y=0 即落稳位 z=0.125")
    ap.add_argument("--base-pos", default="0,0,0", help="机械臂基座世界坐标 'x,y,z'")
    ap.add_argument("--settle", action="store_true",
                    help="推物理等 cube 落稳后再渲染（默认不推，直接构图）")
    # --- 机器人姿态 ---
    ap.add_argument("--joint", default=",".join(f"{q:.8f}" for q in DEFAULT_JOINT),
                    help="5 臂关节角(rad)，逗号分隔；顺序=waist/shoulder/elbow/wrist_angle/wrist_rotate")
    ap.add_argument("--gripper", choices=("open", "closed"), default="open",
                    help="夹爪开合（开=两指外极限）")
    # --- 相机（与 env CameraSpec 完全一致：球坐标数值手动指定，无自动计算）---
    ap.add_argument("--cam-target", default="0.30,0,0.15", help="注视点 'x,y,z'（同 env cameras[].target）")
    ap.add_argument("--cam-dist", type=float, default=0.55, help="相机到注视点距离(米)（同 cameras[].distance）")
    ap.add_argument("--cam-yaw", type=float, default=-50.0, help="相机方位角(度)（同 cameras[].yaw）")
    ap.add_argument("--cam-pitch", type=float, default=-30.0, help="相机俯仰角(度)（同 cameras[].pitch）")
    ap.add_argument("--cam-roll", type=float, default=0.0, help="相机横滚角(度)（同 cameras[].roll）")
    ap.add_argument("--cam-fov", type=float, default=60.0, help="垂直视场角(度)（同 cameras[].fov）")
    ap.add_argument("--cam-res", default="640x480", help="分辨率 WxH（同 cameras[].resolution）")
    ap.add_argument("--side-yaw", type=float, default=0.0,
                    help="左视图 yaw(度)：水平正侧看时站在哪一侧——0 = 从 -y 侧看(画面朝 +y)，"
                         "180 = 从 +y 侧看")
    ap.add_argument("--side-pitch", type=float, default=0.0,
                    help="左视图 pitch(度)：默认 0 = 水平视线正侧看（非俯视）。需要略微俯瞰可设 -10~-20")
    # --- 输出 ---
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="输出根目录（每跑生成一个日期时间戳子目录）")
    ap.add_argument("--renderer", choices=("auto", "cpu", "gpu"), default="auto")
    ap.add_argument("--print-joints", action="store_true", help="打印 URDF 关节表后退出")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    proj_root = PROJ
    out_dir = Path(args.out_dir)
    out_dir = out_dir if out_dir.is_absolute() else proj_root / out_dir

    cube_pos = parse_vec(args.cube_pos, "cube-pos") if args.cube_pos else None
    if cube_pos is None:
        cube_pos = [0.30, 0.0]
    if len(cube_pos) == 2:
        cube_pos.append(args.table_z + CUBE_HALF)  # 贴桌顶落稳位
    if len(cube_pos) != 3:
        raise SystemExit("--cube-pos 需要 'x,y' 或 'x,y,z'")

    p.connect(p.DIRECT)
    p.resetSimulation()
    p.setAdditionalSearchPath(str(MESH_PARENT))
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # plane.urdf / cube_small.urdf
    p.setGravity(0, 0, 0)
    robot_id = p.loadURDF(
        str(args.urdf),
        basePosition=parse_vec(args.base_pos, "base-pos", 3),
        useFixedBase=True,
    )
    if args.print_joints:
        dump_joints(robot_id)
        return 0

    # 场景（env 同款：plane + 棕桌 + 红 cube_small）+ 姿态
    plane_id, table_id, cube_id = build_scene(args.table_z, tuple(cube_pos))
    joint_q = parse_vec(args.joint, "joint", 5)
    set_robot_pose(robot_id, joint_q, args.gripper == "open")

    if args.settle:
        ok = settle(robot_id, cube_id)
        print(f"[settle] cube 落稳={ok} 最终 z="
              f"{p.getBasePositionAndOrientation(cube_id)[0][2]:.4f}")

    # 诊断量（客观打印，不做自动判定——方位对不对以渲染图为准）
    x_dir, ang_to_down = wrist_segment_direction(robot_id)
    print(f"[pose ] arm_joint={[round(q, 4) for q in joint_q]}  gripper_open={args.gripper == 'open'}")
    print(f"[pose ] 腕段(wrist_link)延伸方向={np.round(x_dir, 3).tolist()}  "
          f"与垂直向下夹角={np.degrees(ang_to_down):.1f}°")
    tip_z = lowest_z(robot_id, ("left_finger", "right_finger"))
    cube_top_z = cube_pos[2] + CUBE_HALF
    print(f"[pose ] 夹爪指端最低 z≈{tip_z:.4f} | cube 顶 z={cube_top_z:.4f} | "
          f"间隙≈{tip_z - cube_top_z:+.4f} m")

    # 相机 = env CameraSpec 球坐标，数值全部手动指定（不做任何自动定位）
    target = parse_vec(args.cam_target, "cam-target", 3)
    res = parse_res(args.cam_res)
    renderer = resolve_renderer(args.renderer)

    def render_view(yaw: float, pitch: float, roll: float, out_png: Path) -> dict:
        cam = {"target": target, "distance": args.cam_dist, "yaw": yaw,
               "pitch": pitch, "roll": roll, "fov": args.cam_fov}
        info = render_one(robot_id, out_png, cam, res, renderer)
        print(f"[view ] yaw={yaw} pitch={pitch} dist={args.cam_dist} "
              f"fov={args.cam_fov} -> {out_png.name}  eye_world={info['eye_world']}")
        return info

    print(f"[cam  ] 主图 CameraSpec: target={target} distance={args.cam_dist} "
          f"yaw={args.cam_yaw} pitch={args.cam_pitch} roll={args.cam_roll} "
          f"fov={args.cam_fov} resolution={list(res)}")

    # 输出：日期时间戳文件夹，内两张——主图 + 左视图（水平视线正侧看）
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    render_view(args.cam_yaw, args.cam_pitch, args.cam_roll, run_dir / f"{ts}.png")
    render_view(args.side_yaw, args.side_pitch, 0.0, run_dir / f"side_yaw{args.side_yaw:g}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
