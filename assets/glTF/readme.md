既然你已经有 `pikachu_role.glb`，我们可以做一个 **完整的 Web Skeleton Viewer**，实现：

* 浏览器加载 `pikachu_role.glb`
* 3D 可旋转查看
* 自动读取骨骼
* 自动生成 **关节滑动条**
* 滑动条实时控制骨骼

只需要 **一个 HTML 文件**，在 VSCode 里即可运行。

---

# 目录结构

建议这样放：

```text
project/
 ├─ pikachu_role.glb
 └─ viewer.html
```

---

# viewer.html（完整脚本）


# 运行方法

在 VSCode 终端运行：

```bash
python -m http.server 8000
```

打开浏览器：

```
http://localhost:8000/viewer.html
```

即可看到：

* Pikachu 模型
* 右侧骨骼滑动条
* 拖动即可控制骨骼

---

# 现在这个系统支持

✔ 模型浏览
✔ 鼠标旋转视角
✔ 所有骨骼自动识别
✔ 滑动条控制关节

---

# 如果你愿意，我可以再帮你升级这个 viewer（非常推荐）

升级后会变成 **专业级 Skeleton Viewer**，增加：

### 1 骨骼可视化

类似机器人关节：

```
o---o---o
```

### 2 IK / FK 识别

只显示：

```
upper_arm_fk.L
forearm_fk.L
hand_fk.L
```

避免几百个骨骼。

### 3 自动关节树

```
arm.L
 ├ upper_arm
 ├ forearm
 └ hand
```

### 4 Python远程控制

你的 Python 脚本可以：

```
send joint angle
↓
浏览器模型实时动
```

这套其实就是 **MuJoCo / IsaacSim / Meshcat 的 Web Viewer 原理**。

