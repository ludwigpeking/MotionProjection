"""The Chinese version of the explainer: narration, the phrases the animations wait for, and
the on-screen texts. Names (Kinect, MediaPipe, Blender, Optoma, Manim) stay as they are, and
the main terms carry their English name in brackets.

NARRATION        the script, one entry per section, same keys and order as narration.py
CUES             per section: English cue phrase of explainer_video.py -> phrase of the Chinese narration
TEXTS            on-screen text -> Chinese
TEXT_FRAGMENTS   for texts assembled at run time: fragments replaced inside them
"""

NARRATION = [
    ("title",
     "这段视频讲的是：怎样把画面投到一张会动的脸上，并且让它一直贴在脸上。"
     "一台 Kinect 深度相机负责看，一台小型的 Optoma 投影仪负责画。"
     "关键在于两者之间的几何关系：相机画面里的一个像素，怎样对应到房间里的一个点；"
     "这个点，又怎样对应到投影仪的一个像素。"),

    ("equipment",
     "先看设备，一共两台。Kinect 里面有两个镜头。"
     "一个是普通的彩色相机，分辨率是 1920 乘 1080。"
     "另一个是红外深度相机，分辨率是 512 乘 424。"
     "它自己发出红外光，再测量光返回所用的时间，由此算出距离。"
     "投影仪是一台小型的 Optoma，分辨率也是 1920 乘 1080，当作电脑的第二块屏幕来用。"),

    ("rig",
     "这是从正上方看下去的布置。深度相机放在原点，投影仪放在它旁边。"
     "两者之间的距离叫基线，用字母 b 表示。"
     "人站在前方，到深度相机的距离用 d 表示。再往后是一面墙。"
     "相机和投影仪不在同一个位置，看东西的角度不同，"
     "所以相机画面里的某个像素，并不等于投影仪画面里的同一个像素。"
     "两者唯一共同的参照，就是这个房间本身。"),

    ("notation",
     "整个过程用到三幅画面，所以有三套像素坐标。"
     "深度相机的画面里，坐标写作 x 和 y，每个像素还带着一个距离 d。"
     "彩色相机的画面里，在字母右上角加一个小记号，和深度相机区分开。"
     "投影仪的画面里，加两个小记号。"
     "在每一幅画面里，x 表示从左往右的位置，y 表示从上往下的位置。"
     "房间里只用一套坐标，写成大写的 X、Y、Z，单位是米，以彩色相机为原点。"
     "大写 X 指向右边，大写 Y 指向上方，大写 Z 指向正前方。"),

    ("pixel_to_point",
     "一个像素怎么变成空间里的一个点？"
     "每个像素其实代表一个方向：从镜头中心出发，穿过这个像素，射向远处的一条线。"
     "先说焦距，用 f 表示。它是镜头中心到成像面的距离，这里用像素作单位。"
     "拿像素的横坐标，减去画面中心的横坐标，再除以焦距 f，得到的就是这条线的斜率。"
     "斜率再乘以距离，也就是大写的 Z，就得到这个点向右偏了多少米，这就是大写的 X。"
     "纵向也是同样的算法，得到大写的 Y。"
     "只是要加一个负号，因为画面里的 y 是往下增大的，而空间里的 Y 是向上的。"
     "所以，一个像素，加上一个距离，就确定了房间里的一个点。"),

    ("association",
     "不过这里有个问题：距离是另一个镜头测出来的，两个镜头相隔五厘米。"
     "那么深度相机的某个像素，对应彩色相机的哪个像素呢？分三步。"
     "第一步，用深度相机自己的焦距，把深度像素和它的距离，换算成房间里的一个点。"
     "第二步，两个镜头之间的相对位置，出厂时已经精确测好，按这个数值把点挪过去。"
     "第三步，把这个点投影到彩色相机的画面上，就找到了看着同一个位置的那个彩色像素。"
     "要注意，两幅画面错开多少，和距离有关：近的东西错开得多，远的东西错开得少。"
     "对每个深度像素都这样算一遍，每个彩色像素就都有了自己的距离。"),

    ("cloud",
     "这样一来，每个带有距离的彩色像素，都对应房间里的一个点。"
     "下面是这套装置实际采集到的数据。这是房间的三维点云。"
     "青色的框是深度相机，品红色的框是投影仪，它的位置是标定算出来的。"),

    ("projector",
     "再来看投影仪。可以把它看成一台倒过来用的相机："
     "相机是让光进来，投影仪是让光出去，光走的路线是一样的。"
     "所以描述相机的那几个参数，对投影仪同样适用。"
     "一个是焦距，一个是画面中心，还有它在空间里的位置和方向。"
     "位置和方向用旋转 R 和平移 t 来表示，说的是它相对于深度相机摆在哪里、对着哪边。"
     "具体怎么算呢？取房间里的一个点。先经过旋转和平移，换到投影仪自己的坐标系里。"
     "然后除以它到投影仪的距离，乘以焦距，再加上画面中心。"
     "算出来的，就是能照亮这个点的那个投影仪像素。"),

    ("coded_dots",
     "那么，投影仪的位置和方向怎么测出来？"
     "办法是让投影仪自己告诉我们，它的每个像素落在了哪里。"
     "投影仪先投出一片整齐的光点，每个光点在投影仪画面里的坐标都是已知的。"
     "接下来的十几帧里，每个光点按照自己的编号一亮一灭，就像在发一串二进制密码。"
     "相机拍到某个光点以后，根据它亮灭的顺序，就能认出它是投影仪的哪一个像素。"
     "同时，深度相机也测出了这个光点在房间里的位置。"
     "于是，每个光点都给出一对数据：房间里的一个点，和照亮它的那个投影仪像素。"),

    ("solve",
     "投影仪的模型里，一共有七个未知的参数。旋转占三个，平移占三个，焦距占一个。"
     "每一对数据可以列出两个方程，横坐标一个，纵坐标一个。"
     "几百对数据，远远多于七个参数所需要的。"
     "所以计算的目标，是找出这样一组参数："
     "让模型算出来的光点，和投影仪实际投出的光点，尽可能接近。"
     "个别认错了的光点，会被自动剔除。"
     "最后可以把结果写成一个矩阵：内参矩阵 K，乘以旋转和平移。"
     "这是一个三乘四的矩阵，它把房间里的点直接变成投影仪的像素。"),

    ("focal",
     "焦距为什么这么重要？因为它决定了画面的放大倍数。"
     "一个宽度为 s 的物体，放在距离 D 的地方，在画面上占多少像素？"
     "答案是焦距乘以宽度，再除以距离。"
     "举个例子：一张十五厘米宽的脸，离投影仪九十厘米，大约占三百七十个像素。"
     "如果模型里的焦距不准，投到脸上的图案，大小就会跟着出错。"),

    ("ambiguity",
     "这里藏着一个陷阱。如果所有光点都打在同一面墙上，计算就分不清下面两种情况："
     "是焦距比较小、投影仪离墙比较近，还是焦距比较大、投影仪离墙比较远。"
     "这两种情况在墙上投出的光点，位置完全相同。"
     "只有在离得近的物体上，两者才会显出差别。"
     "所以标定的时候，画面里一定要有近处的物体，让一部分光点落在上面。"
     "有了这些近处的光点，焦距才能定下来。"),

    ("face",
     "现在轮到会动的目标：人脸。"
     "相机每拍一帧，MediaPipe 这个人脸识别模型，就在脸上找出 468 个特征点，"
     "给出它们在彩色画面里的坐标。这些点描出了脸的形状。"
     "深度相机再补上距离，并且根据特征点上的深度，确定脸部起伏的深浅。"
     "这样，每个特征点都变成了房间里的一个点。"
     "把这些点逐个送进投影仪的模型，就得到投影仪应该画出的网格。"
     "其中背对着投影仪的三角形，就不画了。"),

    ("texture",
     "一张平面的图，怎么贴到凹凸不平的脸上？"
     "秘密在于，网格上的每个点都有两个地址。"
     "一个是它在脸上的位置，另一个是它在一张正方形平面图上的位置。"
     "这张平面图就是贴图，这种对应方式叫 UV 映射。可以把它想成一张摊平了的脸。"
     "在 Blender 里，把颜色画在网格上，颜色实际上就落在这张正方形的图里。"
     "运行的时候，三角形还是原来那些三角形。"
     "贴图上的每个三角形，都被拉伸到实时网格上对应的那个三角形上，然后画进投影仪的画面。"
     "变的只是三个顶点的位置，颜色跟着三角形一起走。"),

    ("error",
     "误差是从哪里来的？主要来自深度测量，而且会被基线放大。"
     "假设测出来的距离比实际远了一点，这个差值记作德尔塔 Z。"
     "系统就会以为，这个点在相机视线上更靠后的位置。"
     "于是它让投影仪对准这个它以为的位置。"
     "可是光还没走到那里，就先被真实的脸挡住了。结果，光落在了真实位置的旁边。"
     "偏了多少呢？大约是德尔塔 Z 乘以基线 b，再除以距离 d。"
     "当投影仪离深度相机七十厘米时，一厘米的深度误差，会让图案偏九毫米。"
     "如果把投影仪挪到深度相机旁边，同样的误差，只会偏一毫米左右。"),

    ("loop",
     "把这些环节连起来，就是一个围绕运动物体的反馈回路。"
     "脸动了。彩色相机拍下一帧。MediaPipe 找出特征点，这就是要追踪的目标。"
     "深度相机给每个特征点配上距离，它就成了房间里的一个点。"
     "投影仪模型再把这个点换算成投影仪的像素，光就打到了脸上。"
     "接着脸又动了，整个过程再来一遍，每秒钟十五次。"
     "不过走完一遍需要零点几秒，所以系统要提前预判："
     "它对准的不是脸现在的位置，而是光到达的那一刻，脸将要到的位置。"),

    ("closing",
     "从相机的像素，到房间里的点，再到投影仪的像素，最后回到脸上。"),
]

# The voice reads an English name differently every time, and sometimes wrongly. Every
# clause of the script that holds such a name is therefore spoken whole, several times
# (sentence_takes.py), and one take is chosen by ear. Each entry: (identifier, section,
# the clause exactly as it stands in NARRATION).
NAME_CLAUSES = [
    ("title_kinect", "title", "一台 Kinect 深度相机负责看，"),
    ("title_optoma", "title", "一台小型的 Optoma 投影仪负责画。"),
    ("equipment_kinect", "equipment", "Kinect 里面有两个镜头。"),
    ("equipment_optoma", "equipment", "投影仪是一台小型的 Optoma，分辨率也是 1920 乘 1080，当作电脑的第二块屏幕来用。"),
    ("face_mediapipe", "face", "相机每拍一帧，MediaPipe 这个人脸识别模型，就在脸上找出 468 个特征点，"),
    ("texture_blender", "texture", "在 Blender 里，把颜色画在网格上，颜色实际上就落在这张正方形的图里。"),
    ("loop_mediapipe", "loop", "MediaPipe 找出特征点，这就是要追踪的目标。"),
]

# The take chosen for each clause: identifier -> take number. A clause without a choice uses take 1.
def _chosen_takes():
    """The takes chosen by ear, saved by takes_page.py; empty until a choice was made."""
    import json
    import os
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "name_readings", "chosen_takes.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as file:
        return {identifier: int(take) for identifier, take in json.load(file).items()}


CHOSEN_TAKES = _chosen_takes()

CUES = {
    "title": {"A Kinect depth camera": "一台 Kinect 深度相机", "A small Optoma": "一台小型的 Optoma",
              "how a camera pixel": "相机画面里的一个像素", "and how that point": "这个点，又怎样"},
    "equipment": {"A colour camera": "一个是普通的彩色相机", "infrared depth camera": "另一个是红外深度相机",
                  "The projector is": "投影仪是一台"},
    "rig": {"Kinect sits": "深度相机放在原点", "projector stands": "投影仪放在它旁边", "baseline": "叫基线",
            "person is at": "人站在前方", "wall is further": "再往后是一面墙", "never share": "不在同一个位置",
            "only common ground": "唯一共同的参照"},
    "notation": {"x and y in the depth": "深度相机的画面里", "x prime and y prime": "彩色相机的画面里",
                 "x double prime": "投影仪的画面里", "x counts columns": "从左往右的位置",
                 "y counts rows": "从上往下的位置", "room has one set": "房间里只用一套坐标",
                 "X to the right": "大写 X 指向右边", "Y up": "大写 Y 指向上方", "Z straight ahead": "大写 Z 指向正前方"},
    "pixel_to_point": {"pixel names a direction": "每个像素其实代表一个方向", "focal length, f": "先说焦距",
                       "subtract the image centre": "减去画面中心", "Divide by f": "再除以焦距",
                       "Multiply by the distance": "再乘以距离", "rows work the same": "纵向也是同样的算法",
                       "with a minus sign": "加一个负号", "pixel plus a distance": "加上一个距离"},
    "association": {"other lens": "另一个镜头", "First, the depth pixel": "第一步",
                    "Second, the point is shifted": "第二步", "Third, the point is projected": "第三步",
                    "depends on distance": "和距离有关", "Done for every depth pixel": "对每个深度像素"},
    "cloud": {"depth cloud of the room": "房间的三维点云", "Kinect in cyan": "青色的框是",
              "projector in magenta": "品红色的框是"},
    "projector": {"camera running backwards": "倒过来用的相机", "A focal length": "一个是焦距",
                  "an image centre": "一个是画面中心", "and a pose": "还有它在空间里",
                  "Rotate and translate": "先经过旋转和平移", "Divide by its distance": "然后除以它到投影仪的距离",
                  "The result is": "算出来的"},
    "coded_dots": {"grid of dots": "一片整齐的光点", "known pixel": "都是已知的",
                   "blinks its own binary code": "一亮一灭", "camera sees a dot": "相机拍到某个光点",
                   "the code says": "就能认出", "depth gives": "深度相机也测出了",
                   "each dot is one pair": "都给出一对数据"},
    "solve": {"seven unknowns": "七个未知的参数", "Each pair gives two equations": "可以列出两个方程",
              "A few hundred pairs": "几百对数据", "the fit looks for": "所以计算的目标",
              "ignores dots": "个别认错了的光点", "Written as one matrix": "写成一个矩阵"},
    "focal": {"sets the magnification": "放大倍数", "An object of size s": "一个宽度为 s 的物体",
              "covers f times s": "焦距乘以宽度", "A face fifteen centimetres": "一张十五厘米宽的脸",
              "If the model's focal length": "如果模型里的焦距不准"},
    "ambiguity": {"every dot lands on the wall": "如果所有光点都打在同一面墙上", "short focal length": "是焦距比较小",
                  "from a long one": "还是焦距比较大", "Both draw the same": "这两种情况在墙上投出的光点",
                  "only disagree on a near object": "只有在离得近的物体上", "those dots pin": "有了这些近处的光点"},
    "face": {"MediaPipe finds": "MediaPipe 这个人脸识别模型", "Kinect adds the distance": "深度相机再补上距离",
             "depth at the landmarks": "根据特征点上的深度", "each landmark becomes": "每个特征点都变成了",
             "goes through the projector model": "送进投影仪的模型", "That gives the mesh": "投影仪应该画出的网格",
             "face away from the projector": "背对着投影仪的三角形"},
    "texture": {"second address": "都有两个地址", "That layout is the U V map": "这种对应方式叫",
                "the face, unfolded": "摊平了的脸", "At run time": "运行的时候",
                "Each triangle of the square": "贴图上的每个三角形", "Only the corners move": "变的只是三个顶点"},
    "error": {"Mostly from depth": "主要来自深度测量", "Suppose the measured distance": "假设测出来的距离",
              "further along the camera's ray": "更靠后的位置", "aims the projector": "让投影仪对准",
              "the light stops": "可是光还没走到那里", "It lands beside": "光落在了真实位置的旁边",
              "The sideways shift": "偏了多少呢", "With the projector seventy": "当投影仪离深度相机七十厘米",
              "Slide the projector": "如果把投影仪挪到"},
    "loop": {"The face moves": "脸动了", "colour camera captures": "彩色相机拍下一帧",
             "MediaPipe finds": "MediaPipe 找出特征点", "attaches a distance": "配上距离",
             "projector model turns": "投影仪模型再把这个点", "light lands on the face": "光就打到了脸上",
             "the face moves again": "接着脸又动了", "fifteen times a second": "每秒钟十五次",
             "One pass takes": "走完一遍需要"},
    "closing": {},
}

TEXTS = {
    # title and closing
    "Projecting onto a moving face": "在移动的脸上投影  Projection Mapping",
    "Kinect: watches": "Kinect：观察",
    "projector: paints": "投影仪：绘制",
    "camera pixel": "相机像素",
    "point in the room": "房间里的点",
    "projector pixel": "投影仪像素",
    "the face": "脸",
    "Animated with Manim, the mathematical animation library": "动画使用 Manim 制作：一个数学动画库",
    "created by Grant Sanderson of 3Blue1Brown  (Manim Community edition)":
        "由 3Blue1Brown 的 Grant Sanderson 创建  (Manim Community edition)",
    # equipment
    "The equipment": "设备",
    "colour camera   1920 × 1080": "彩色相机 (colour)   1920 × 1080",
    "depth camera   512 × 424\ninfrared, measures distance by timing light":
        "深度相机 (depth)   512 × 424\n红外，测量光的往返时间来测距 (time of flight)",
    "Optoma projector": "Optoma 投影仪",
    "1920 × 1080   second screen": "1920 × 1080   第二块屏幕",
    "photo: Evan-Amos, public domain": "照片：Evan-Amos，公有领域",
    # rig
    "The rig, from above": "装置俯视图",
    "projector": "投影仪",
    "baseline b = 0.7 m": "基线 (baseline) b = 0.7 m",
    "person": "人",
    "wall, 2.5 m": "墙，2.5 m",
    "no shared viewpoint:\npixels do not map to pixels": "视点不重合：\n像素不能直接对应像素",
    "common ground: the room": "共同基础：房间本身",
    # notation
    "Three images, one room": "三幅图像，一个房间",
    "depth camera  512 × 424": "深度相机  512 × 424",
    "colour camera  1920 × 1080": "彩色相机  1920 × 1080",
    "projector  1920 × 1080": "投影仪  1920 × 1080",
    "X  right": "X  向右",
    "Y  up": "Y  向上",
    "Z  ahead": "Z  向前",
    "the room: (X, Y, Z) in metres,\nmeasured from the colour camera": "房间：(X, Y, Z)，单位米，\n从彩色相机量起",
    # pixel to point
    "Colour pixel + distance = point   (seen from above)": "彩色像素 + 距离 = 点   （俯视）",
    "image": "图像",
    "pixel x′": "像素 x′",
    "f  (pixels)": "焦距 f（像素）",
    "slope = (x′ − c<sub>x</sub>) / f": "斜率 = (x′ − c<sub>x</sub>) / f",
    "point": "点",
    "lens centre": "镜头中心",
    "Z: straight ahead": "Z：正前方",
    "distance Z  (metres)": "距离 Z（米）",
    "minus sign: rows count downward,\nY points up": "负号：行向下数，\n而 Y 指向上方",
    "Z = distance": "Z = 距离",
    # association
    "Which colour pixel belongs to a depth pixel?   (seen from above)": "深度像素对应哪个彩色像素？   （俯视）",
    "depth lens": "深度镜头",
    "colour lens": "彩色镜头",
    "same depth pixel, different distance:\nthe colour pixel moves": "同一个深度像素，距离不同：\n彩色像素随之移动（视差 parallax）",
    "every colour pixel gets its own distance": "每个彩色像素都有了自己的距离",
    "1  lift:  (x, y), d  →  point": "1  抬升：(x, y), d  →  点",
    "2  shift by the lens offset": "2  平移镜头间的偏移量",
    "3  project:  point  →  (x′, y′)": "3  投影：点  →  (x′, y′)",
    # cloud
    "The room as the system sees it: real data from the rig": "系统眼中的房间：装置采集的真实数据",
    "depth cloud: one point (X, Y, Z) per pixel": "深度点云 (point cloud)：每个像素一个点 (X, Y, Z)",
    "projector and its beam": "投影仪及其光束",
    # projector
    "The projector: a camera running backwards": "投影仪：一台反向工作的相机",
    "projector lens": "投影仪镜头",
    "projector image": "投影仪图像",
    "point (X, Y, Z)": "点 (X, Y, Z)",
    "Kinect frame": "Kinect 坐标系",
    "projector frame": "投影仪坐标系",
    "focal length f": "焦距 f (focal length)",
    "image centre (cx, cy)": "图像中心 (cx, cy)",
    "pose: rotation R, translation t": "位置和方向 (pose)：旋转 R，平移 t",
    # coded dots
    "Calibration: the projector shows where its pixels land": "标定 (calibration)：投影仪显示它的像素落在哪里",
    "this dot, frame by frame: its code": "这个点逐帧的亮灭：它的编码",
    "colour camera image": "彩色相机图像",
    "seen at (x′, y′) = (1302, 418)": "出现在 (x′, y′) = (1302, 418)",
    "code 101101 → dot 32": "编码 101101 → 第 32 号点",
    "depth → (X, Y, Z) = (0.28, 0.05, 0.79) m": "深度 → (X, Y, Z) = (0.28, 0.05, 0.79) m",
    "one pair per dot": "每个点一对数据",
    # solve
    "Solving the projector model from the pairs": "由成对的数据求解投影仪模型",
    "∼ : equal after dividing\nby the third row (the distance)": "∼ ：除以第三行（距离）\n之后相等",
    "decoded wrongly: ignored": "解码错误：忽略",
    "rotation R        3 numbers": "旋转 R        3 个数",
    "translation t     3 numbers": "平移 t        3 个数",
    "focal length f    1 number": "焦距 f        1 个数",
    "7 unknowns": "7 个未知数",
    "each pair: 2 equations  (x″ and y″)": "每对数据：2 个方程  (x″ 和 y″)",
    "about 500 pairs: 1000 equations": "约 500 对：1000 个方程",
    "dots the projector drew": "投影仪实际画出的点",
    "where the model puts them": "模型算出的位置",
    # focal
    "Why the focal length matters: magnification": "焦距 (focal length) 为什么重要：放大倍率",
    "pixels it covers": "它占据的像素",
    "distance D": "距离 D",
    "pixels = f · s / D": "像素数 = f · s / D",
    "f wrong by 5%  →  image on the face 5% too big or too small": "f 偏差 5%  →  脸上的图像偏大或偏小 5%",
    "face, size s": "脸，大小 s",
    "2200 × 0.15 / 0.9 ≈ 370 pixels": "2200 × 0.15 / 0.9 ≈ 370 像素",
    # ambiguity
    "The trap: a wall cannot fix the focal length": "陷阱：只靠一面墙无法确定焦距",
    "wall": "墙",
    "short focal, close": "焦距短，离得近",
    "long focal, far back": "焦距长，离得远",
    "near object": "近处的物体",
    "same dots on the wall": "墙上的点完全相同",
    "different dots on a near object": "近处物体上的点不同",
    "near dots pin the focal length": "近处的点确定焦距",
    # face
    "The moving target: the face": "移动的目标：脸",
    "MediaPipe: 468 landmarks (x′, y′)": "MediaPipe：468 个特征点 (x′, y′)",
    "as the camera\nsees it": "相机\n看到的",
    "as the projector\nmust draw it": "投影仪\n应该画的",
    "projector\nmodel": "投影仪\n模型",
    "+ Kinect distance": "+ Kinect 距离",
    "+ depth at the landmarks: relief": "+ 特征点处的深度：起伏",
    "= 468 points (X, Y, Z)": "= 468 个点 (X, Y, Z)",
    # texture
    "From a flat painting to the face: the UV map": "从平面的画到脸上：UV 映射 (UV map)",
    "painting in Blender: the flat texture (left) and the mesh (right)": "在 Blender 里绘制：平面贴图（左）和网格 mesh（右）",
    "texture (U, V)": "贴图 texture (U, V)",
    "the face, unfolded": "展开的脸",
    "same triangles, moved corners: the paint follows": "三角形相同，顶点移动：颜料跟着走",
    # error
    "Where the error comes from   (seen from above)": "误差从哪里来   （俯视）",
    "the real face surface": "真实的脸部表面",
    "true point": "真实的点",
    "believed point": "以为的点",
    "shift": "偏移",
    "shift ≈ ΔZ · b / d": "偏移 ≈ ΔZ · b / d",
    "(ΔZ is drawn much larger than 1 cm)": "（图中的 ΔZ 远大于 1 cm）",
    "projector next to the Kinect:\ndepth errors barely show": "投影仪紧挨 Kinect：\n深度误差几乎看不出来",
    "light lands here": "光落在这里",
    # loop
    "the face moves": "脸在移动",
    "colour camera\ncaptures a frame": "彩色相机\n拍下一帧",
    "MediaPipe: landmarks\n(x′, y′) = the target": "MediaPipe：特征点\n(x′, y′) = 目标",
    "depth camera: distance\n→ point (X, Y, Z)": "深度相机：距离\n→ 点 (X, Y, Z)",
    "projector model\n→ pixel (x″, y″)": "投影仪模型\n→ 像素 (x″, y″)",
    "light lands\non the face": "光落在\n脸上",
    "The feedback loop": "反馈回路 (feedback loop)",
    "15 loops per second": "每秒 15 圈",
    "one pass takes a fraction of a second: aim where the face will be": "跑一圈要零点几秒：瞄准脸将要到的位置",
}

TEXT_FRAGMENTS = [
    ("baseline b = ", "基线 b = "),
    ("shift ≈ ", "偏移 ≈ "),
]
