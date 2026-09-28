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
     "这是一段跟随人脸移动的投影。Kinect 深度相机负责观察，一台小型 Optoma 投影仪负责绘制。"
     "这段视频讲解把两者联系起来的几何：相机的一个像素，怎样变成房间里的一个点；"
     "这个点，又怎样变成投影仪的一个像素。"),

    ("equipment",
     "两件设备。Kinect 里有两个传感器。一个是彩色相机，分辨率 1920 乘 1080。"
     "另一个是红外深度相机，分辨率 512 乘 424，它靠测量自身发出的光往返的时间来测距。"
     "投影仪是一台小型 Optoma，分辨率 1920 乘 1080，作为第二块屏幕接在电脑上。"),

    ("rig",
     "这是从上方看到的装置。Kinect 位于原点。投影仪放在它的旁边。"
     "两者之间的距离叫做基线，记作 b。人在前方，距离为 d，更远处是一面墙。"
     "两台设备的视点永远不重合，所以相机里的一个像素，并不对应投影仪里的同一个像素。"
     "它们唯一的共同基础，是房间本身。"),

    ("notation",
     "这里涉及三幅图像，所以有三套像素坐标。"
     "深度相机里是 x 和 y，每个像素还带有一个距离 d。"
     "彩色相机里是 x 撇和 y 撇。"
     "投影仪里是 x 两撇和 y 两撇。"
     "在每一幅图像里，x 向右数列，y 向下数行。"
     "房间只有一套坐标：大写的 X、Y、Z，单位是米，从彩色相机量起。"
     "X 向右，Y 向上，Z 指向正前方。"),

    ("pixel_to_point",
     "一个像素怎样变成一个点？一个像素代表一个方向：一条从镜头中心出发、穿过这个像素的射线。"
     "焦距 f，是镜头中心到图像的距离，以像素为单位。"
     "取像素所在的列 x 撇，减去图像中心。再除以 f，就得到这条射线的斜率。"
     "乘以距离 Z，就得到横向位置，也就是大写的 X，单位是米。"
     "行的算法相同，得到大写的 Y，只是多一个负号，因为行是向下数的，而 Y 指向上方。"
     "所以，一个像素加上一个距离，就是房间里的一个点。"),

    ("association",
     "但是，距离是由另一个镜头测得的，它在旁边五厘米处。"
     "深度像素怎样找到它对应的彩色像素？"
     "第一步，把深度像素 x、y 连同它的距离 d，用深度相机自己的焦距，抬升为房间里的一个点。"
     "第二步，把这个点平移两个镜头之间的偏移量，这个偏移量在出厂时已经测定。"
     "第三步，把这个点投影到彩色相机里，得到看见同一位置的彩色像素 x 撇、y 撇。"
     "两幅图像之间的错位取决于距离：近处的物体错位大，远处的错位小。"
     "对每一个深度像素都这样做，每个彩色像素就有了自己的距离。"),

    ("cloud",
     "结果是：每一个有距离的彩色像素，都对应一个点 X、Y、Z。"
     "这是装置采集到的真实数据。房间的深度点云，青色的是 Kinect，"
     "品红色的是投影仪，画在标定算出的位置上。"),

    ("projector",
     "再看投影仪。它是一台反向工作的相机：相机让光进来的地方，投影仪让光出去。"
     "所以可以用同样的方式描述它。一个焦距，一个图像中心，还有一个位姿："
     "旋转 R 和平移 t，说明它相对 Kinect 的位置和朝向。"
     "取房间里的一个点。通过旋转和平移，把它变换到投影仪自己的坐标系里。"
     "除以它的距离，乘以焦距，再加上图像中心。"
     "结果就是 x 两撇、y 两撇：照亮这个点的投影仪像素。"),

    ("coded_dots",
     "要求出位姿，投影仪必须告诉我们它的像素落在哪里。"
     "它画出一个点阵，每个点都在已知的像素 x 两撇、y 两撇上。"
     "在连续的若干帧里，每个点按自己的二进制编码闪烁。"
     "相机在 x 撇、y 撇处看到一个点时，编码就说明它来自哪个投影仪像素。"
     "而深度给出了这个点在房间里的位置。"
     "所以每个点就是一对数据：一个点 X、Y、Z，以及照亮它的投影仪像素。"),

    ("solve",
     "这个模型有七个未知数。旋转三个，平移三个，再加上焦距。"
     "每一对数据给出两个方程：一个关于 x 两撇，一个关于 y 两撇。"
     "几百对数据远远超过所需，所以拟合要找的，是让模型算出的点最接近投影仪实际画出的点的那组数值，"
     "并且忽略解码错误的点。"
     "写成一个矩阵，投影仪就是 K 乘以 R、t：三行四列，把房间里的点变成像素。"),

    ("focal",
     "焦距为什么重要？它决定放大倍率。"
     "一个大小为 s 的物体，在距离 D 处，占据 f 乘以 s 再除以 D 个像素。"
     "一张十五厘米宽的脸，离投影仪九十厘米，大约占三百七十个像素。"
     "如果模型里的焦距不对，投在脸上的图像大小就不对。"),

    ("ambiguity",
     "这里还有一个陷阱。如果所有的点都落在墙上，拟合就分不清两种情况："
     "焦距短而投影仪离得近，还是焦距长而投影仪离得远。两者在墙上画出的点完全一样。"
     "只有在近处的物体上，它们才不一致。"
     "所以扫描时需要有点落在近处的物体上，比如坐在光束里的人，这些点才能确定焦距。"),

    ("face",
     "现在来看移动的目标。在相机的每一帧里，MediaPipe 在脸上找到 468 个特征点，以彩色像素 x 撇、y 撇表示。"
     "这就是形状。Kinect 补上距离，特征点处的深度决定了起伏的深浅。"
     "于是每个特征点都变成一个点 X、Y、Z。"
     "每个点再经过投影仪模型，就得到投影仪应该画出的网格。"
     "背向投影仪的三角形不画。"),

    ("texture",
     "一幅平面的画，怎样贴到弯曲的脸上？"
     "网格上的每一个点，都有第二个地址：在一张平面的方形图像，也就是贴图上，有一个固定的位置。"
     "这种布局叫做 UV 映射。它就是展开的脸。"
     "在 Blender 里，颜料画在网格上，落在这个方形里。"
     "运行时，三角形还是同样的那些。方形里的每个三角形，被拉伸到实时网格上对应的三角形上，"
     "画进投影仪的图像里。移动的只是顶点，颜料跟着走。"),

    ("error",
     "误差从哪里来？主要来自深度，并且通过基线放大。"
     "假设测得的距离偏大了德尔塔 Z。系统就会以为这个点在相机射线上更远的地方。"
     "它让投影仪瞄准这个它以为的点。但是光在到达那里之前，就被真实的表面挡住了。"
     "光落在真实点的旁边。横向偏移大约是德尔塔 Z 乘以基线 b，再除以距离 d。"
     "投影仪离开七十厘米时，一厘米的深度误差会让图像偏移九毫米。"
     "把投影仪移到 Kinect 旁边，同样的误差只造成大约一毫米的偏移。"),

    ("loop",
     "合在一起，这是一个围绕移动物体的反馈回路。"
     "脸在移动。彩色相机拍下它。MediaPipe 找到特征点：这就是目标。"
     "深度相机给每个特征点加上距离，让它成为房间里的一个点。"
     "投影仪模型把这个点变成投影仪像素，光就落在脸上。"
     "然后脸又动了，回路再跑一遍，每秒十五次。"
     "跑一遍需要零点几秒，所以系统瞄准的，是光到达时脸将会在的位置。"),

    ("closing",
     "相机像素，到房间里的点，到投影仪像素，再回到脸上。"),
]

CUES = {
    "title": {"A Kinect depth camera": "Kinect 深度相机", "A small Optoma": "一台小型 Optoma",
              "how a camera pixel": "相机的一个像素", "and how that point": "这个点，又怎样"},
    "equipment": {"A colour camera": "一个是彩色相机", "infrared depth camera": "红外深度相机",
                  "The projector is": "投影仪是一台"},
    "rig": {"Kinect sits": "Kinect 位于原点", "projector stands": "投影仪放在", "baseline": "叫做基线",
            "person is at": "人在前方", "wall is further": "更远处是一面墙", "never share": "视点永远不重合",
            "only common ground": "唯一的共同基础"},
    "notation": {"x and y in the depth": "深度相机里是", "x prime and y prime": "彩色相机里是",
                 "x double prime": "投影仪里是", "x counts columns": "向右数列", "y counts rows": "向下数行",
                 "room has one set": "房间只有一套坐标", "X to the right": "X 向右，Y 向上", "Y up": "Y 向上，Z",
                 "Z straight ahead": "Z 指向正前方"},
    "pixel_to_point": {"pixel names a direction": "一个像素代表一个方向", "focal length, f": "焦距 f",
                       "subtract the image centre": "减去图像中心", "Divide by f": "再除以 f",
                       "Multiply by the distance": "乘以距离 Z", "rows work the same": "行的算法相同",
                       "with a minus sign": "多一个负号", "pixel plus a distance": "一个像素加上一个距离"},
    "association": {"other lens": "另一个镜头", "First, the depth pixel": "第一步",
                    "Second, the point is shifted": "第二步", "Third, the point is projected": "第三步",
                    "depends on distance": "错位取决于距离", "Done for every depth pixel": "对每一个深度像素"},
    "cloud": {"depth cloud of the room": "房间的深度点云", "Kinect in cyan": "青色的是 Kinect",
              "projector in magenta": "品红色的是投影仪"},
    "projector": {"camera running backwards": "反向工作的相机", "A focal length": "一个焦距",
                  "an image centre": "一个图像中心", "and a pose": "还有一个位姿",
                  "Rotate and translate": "通过旋转和平移", "Divide by its distance": "除以它的距离",
                  "The result is": "结果就是"},
    "coded_dots": {"grid of dots": "画出一个点阵", "known pixel": "已知的像素",
                   "blinks its own binary code": "二进制编码闪烁", "camera sees a dot": "看到一个点时",
                   "the code says": "编码就说明", "depth gives": "而深度给出了",
                   "each dot is one pair": "每个点就是一对数据"},
    "solve": {"seven unknowns": "七个未知数", "Each pair gives two equations": "给出两个方程",
              "A few hundred pairs": "几百对数据", "the fit looks for": "拟合要找的",
              "ignores dots": "忽略解码错误", "Written as one matrix": "写成一个矩阵"},
    "focal": {"sets the magnification": "决定放大倍率", "An object of size s": "一个大小为 s 的物体",
              "covers f times s": "占据 f 乘以 s", "A face fifteen centimetres": "一张十五厘米宽的脸",
              "If the model's focal length": "如果模型里的焦距"},
    "ambiguity": {"every dot lands on the wall": "所有的点都落在墙上", "short focal length": "焦距短而",
                  "from a long one": "还是焦距长", "Both draw the same": "两者在墙上画出的点",
                  "only disagree on a near object": "只有在近处的物体上", "those dots pin": "这些点才能确定焦距"},
    "face": {"MediaPipe finds": "MediaPipe 在脸上找到", "Kinect adds the distance": "Kinect 补上距离",
             "depth at the landmarks": "特征点处的深度", "each landmark becomes": "每个特征点都变成",
             "goes through the projector model": "经过投影仪模型", "That gives the mesh": "投影仪应该画出的网格",
             "face away from the projector": "背向投影仪的三角形"},
    "texture": {"second address": "第二个地址", "That layout is the U V map": "这种布局叫做",
                "the face, unfolded": "展开的脸", "At run time": "运行时",
                "Each triangle of the square": "方形里的每个三角形", "Only the corners move": "移动的只是顶点"},
    "error": {"Mostly from depth": "主要来自深度", "Suppose the measured distance": "假设测得的距离",
              "further along the camera's ray": "相机射线上更远的地方", "aims the projector": "让投影仪瞄准",
              "the light stops": "但是光在到达那里之前", "It lands beside": "光落在真实点的旁边",
              "The sideways shift": "横向偏移大约是", "With the projector seventy": "投影仪离开七十厘米",
              "Slide the projector": "把投影仪移到"},
    "loop": {"The face moves": "脸在移动", "colour camera captures": "彩色相机拍下它",
             "MediaPipe finds": "MediaPipe 找到特征点", "attaches a distance": "加上距离",
             "projector model turns": "投影仪模型把这个点", "light lands on the face": "光就落在脸上",
             "the face moves again": "然后脸又动了", "fifteen times a second": "每秒十五次",
             "One pass takes": "跑一遍需要"},
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
    "pose: rotation R, translation t": "位姿 (pose)：旋转 R，平移 t",
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
    "MediaPipe: 468 landmarks (x′, y′)": "MediaPipe：468 个特征点 landmarks (x′, y′)",
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
