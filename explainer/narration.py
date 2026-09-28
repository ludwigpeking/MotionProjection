"""Narration for the explainer video, one entry per scene section, in playing order.
build_audio.py turns each entry into audio/<key>.mp3, records its length and the time
of every spoken word, and cuts the text into subtitle lines.

Notation used throughout, in the voice and on screen:
    depth camera pixel      (x, y) with distance d
    colour camera pixel     (x', y')
    projector pixel         (x'', y'')
    point in the room       (X, Y, Z) in metres, measured from the colour camera
"""

NARRATION = [
    ("title",
     "This is a projection that follows a moving face. A Kinect depth camera watches. "
     "A small Optoma projector paints. "
     "This video explains the geometry that links them: how a camera pixel becomes a point in the room, "
     "and how that point becomes a projector pixel."),

    ("equipment",
     "Two pieces of equipment. The Kinect holds two sensors. A colour camera, 1920 by 1080 pixels. "
     "And an infrared depth camera, 512 by 424 pixels, which measures distance by timing its own light. "
     "The projector is a small Optoma, 1920 by 1080 pixels, connected to the computer as a second screen."),

    ("rig",
     "Here is the rig from above. The Kinect sits at the origin. The projector stands to its side. "
     "The distance between them is the baseline, b. The person is at a distance d in front, and a wall is further back. "
     "The two devices never share a viewpoint, so a pixel in the camera is not a pixel in the projector. "
     "The only common ground is the room itself."),

    ("notation",
     "Three images are involved, so there are three sets of pixel coordinates. "
     "x and y in the depth camera, with a distance d at every pixel. "
     "x prime and y prime in the colour camera. "
     "x double prime and y double prime in the projector. "
     "In every image, x counts columns to the right, and y counts rows downward. "
     "The room has one set of coordinates: capital X, Y and Z, in metres, measured from the colour camera. "
     "X to the right, Y up, and Z straight ahead."),

    ("pixel_to_point",
     "How does a pixel become a point? A pixel names a direction: a ray that leaves the lens centre and passes through that pixel. "
     "The focal length, f, is the distance from the lens centre to the image, counted in pixels. "
     "Take the pixel's column x prime, and subtract the image centre. Divide by f, and you have the slope of the ray. "
     "Multiply by the distance Z, and you have the sideways position, capital X, in metres. "
     "The rows work the same way and give capital Y, with a minus sign, because rows count downward while Y points up. "
     "So a pixel plus a distance is a point in the room."),

    ("association",
     "But the distance was measured by the other lens, five centimetres to the side. "
     "How does a depth pixel find its colour pixel? "
     "First, the depth pixel x, y, with its distance d, is lifted to a point in the room, using the depth camera's own focal length. "
     "Second, the point is shifted by the offset between the two lenses, which was measured at the factory. "
     "Third, the point is projected into the colour camera. That gives the colour pixel, x prime, y prime, that sees the same spot. "
     "The shift between the two images depends on distance: near things shift more than far things. "
     "Done for every depth pixel, this gives every colour pixel its own distance."),

    ("cloud",
     "The result is a point X, Y, Z for every colour pixel that has a distance. "
     "This is the real data from the rig. The depth cloud of the room, with the Kinect in cyan, "
     "and the projector in magenta, drawn where the calibration placed it."),

    ("projector",
     "Now the projector. It is a camera running backwards: light leaves through the lens where a camera would take it in. "
     "So it has the same description. A focal length, an image centre, and a pose: "
     "a rotation R and a translation t, which say where it sits relative to the Kinect. "
     "Take a point in the room. Rotate and translate it into the projector's own frame. "
     "Divide by its distance, multiply by the focal length, and add the image centre. "
     "The result is x double prime, y double prime: the projector pixel that lights that point."),

    ("coded_dots",
     "To find the pose, the projector has to show where its pixels land. "
     "It draws a grid of dots, each at a known pixel, x double prime, y double prime. "
     "Over a sequence of frames, every dot blinks its own binary code. "
     "When the camera sees a dot at x prime, y prime, the code says which projector pixel it came from. "
     "And the depth gives that dot's position in the room. "
     "So each dot is one pair: a point X, Y, Z, and the projector pixel that lit it."),

    ("solve",
     "The model has seven unknowns. Three for the rotation, three for the translation, and the focal length. "
     "Each pair gives two equations: one for x double prime, one for y double prime. "
     "A few hundred pairs are far more than enough, so the fit looks for the values that bring the model's dots "
     "closest to the dots the projector really drew, and it ignores dots that were decoded wrongly. "
     "Written as one matrix, the projector is K times R, t: three rows, four columns, taking room points to pixels."),

    ("focal",
     "Why does the focal length matter? It sets the magnification. "
     "An object of size s, at distance D, covers f times s over D pixels. "
     "A face fifteen centimetres wide, ninety centimetres from the projector, covers about three hundred and seventy pixels. "
     "If the model's focal length is wrong, the image on the face is the wrong size."),

    ("ambiguity",
     "And there is a trap. If every dot lands on the wall, the fit cannot tell a short focal length with the projector close "
     "from a long one with the projector far back. Both draw the same dots on the wall. "
     "They only disagree on a near object. "
     "So the scan needs dots on something near, such as the person sitting in the beam, and those dots pin the focal length."),

    ("face",
     "Now the moving target. In every camera frame, MediaPipe finds 468 landmarks on the face, as colour pixels x prime, y prime. "
     "That is the shape. The Kinect adds the distance, and the depth at the landmarks sets how deep the relief is. "
     "So each landmark becomes a point X, Y, Z. "
     "Each point goes through the projector model. That gives the mesh as the projector must draw it. "
     "Triangles that face away from the projector are not drawn."),

    ("texture",
     "How does a flat painting end up on a curved face? "
     "Every point of the mesh has a second address: a fixed place on a flat, square image, the texture. "
     "That layout is the U V map. It is the face, unfolded. "
     "In Blender, the paint goes onto the mesh, and lands in that square. "
     "At run time, the triangles are the same ones. Each triangle of the square is stretched onto the matching triangle of the live mesh, "
     "in the projector's image. Only the corners move, and the paint follows."),

    ("error",
     "Where does error come from? Mostly from depth, through the baseline. "
     "Suppose the measured distance is too long by delta Z. The system then believes the point lies further along the camera's ray. "
     "It aims the projector at that believed point. But the light stops at the real surface, before it gets there. "
     "It lands beside the true point. The sideways shift is about delta Z, times the baseline b, over the distance d. "
     "With the projector seventy centimetres away, one centimetre of depth error moves the image nine millimetres. "
     "Slide the projector next to the Kinect, and the same error moves it about one millimetre."),

    ("loop",
     "Put together, this is a feedback loop around a moving object. "
     "The face moves. The colour camera captures it. MediaPipe finds the landmarks: that is the target. "
     "The depth camera attaches a distance to each landmark, which makes it a point in the room. "
     "The projector model turns the point into a projector pixel, and light lands on the face. "
     "Then the face moves again, and the loop runs again, fifteen times a second. "
     "One pass takes a fraction of a second, so the system aims at where the face will be when the light arrives."),

    ("closing",
     "Camera pixel, to a point in the room, to a projector pixel, and back onto the face."),
]
