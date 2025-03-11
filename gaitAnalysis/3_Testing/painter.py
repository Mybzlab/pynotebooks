import cv2


class Painter:
    """
    class for adding visual elements to a frame of a video
    """
    def __init__(self, frame):
        self.frame = frame
        self.frame_height, self.frame_width, _ = frame.shape

    def paint_com(self, com, color=(255, 255, 255)):
        """
        Paint the center of mass on the frame.

        :param com: The center of mass.
        :param color: The color of the center of mass.
        """
        x, y = com
        self.frame = cv2.circle(self.frame, (int(x), int(y)), 5, color, -1)

    def paint_cor(self, cor, color=(255, 255, 255)):
        """
        Paint the center of rotation on the frame.

        :param cor: The center of rotation.
        :param color: The color of the center of rotation.
        """
        x, y = cor
        self.frame = cv2.circle(self.frame, (int(x), int(y)), 5, color, -1)

    def paint_line(self, start, end, color=(255, 255, 255)):
        """
        Paint a line on the frame.

        :param start: The start of the line.
        :param end: The end of the line.
        :param color: The color of the line.
        """
        x1, y1 = start
        x2, y2 = end
        self.frame = cv2.line(self.frame, (int(x1), int(y1)),
                              (int(x2), int(y2)), color, 2)
        
    def paint_circle(self, center, radius, color=(255, 255, 255)):
        """
        Paint a circle on the frame.

        :param center: The center of the circle.
        :param radius: The radius of the circle.
        :param color: The color of the circle.
        """
        x, y = center
        self.frame = cv2.circle(self.frame, (int(x), int(y)), radius, color, 2)

    def paint_percentage(self, location, value, color=(255, 255, 255), half_size=7):
        """
        Paint a bodypart on the frame.

        :param location: The location of the bodypart(x,y).
        :param color: The color of the bodypart.
        """

        x, y = location
        end_angle = int(360 * (value / 100.0))
        self.frame = cv2.ellipse(
            self.frame, (x, y), (half_size, half_size), 90, 0, end_angle, color, thickness=2)
        
    def _remove_substrings(self, original_string, substrings):
        for substring in substrings:
            original_string = original_string.replace(substring, "")
        return original_string
    
    def write_percentage(self, bodypart, location, value, color=(255, 255, 255), text=True, half_size=7):
        """
        Write a percentage on the frame.

        :param bodypart: The bodypart of the percentage(string).
        :param location: The location of the percentage(x,y).
        :param value: The value of the percentage(float).
        :param color: The color of the percentage(rgb).
        
        """
        
        # Add bodypart abbrevation
        text_1 = bodypart[0]
        text_2 = self._remove_substrings(bodypart, ['front', 'mid', 'back'])[0]
        text = (text_1 + text_2 + f" {round(value):.0f}%").upper()

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.3
        thickness = 1
        line_type = cv2.LINE_AA

        # Add percentage text
        x, y = location
        value = f"{value:.0f}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.3
        thickness = 1
        line_type = cv2.LINE_AA
        self.frame = cv2.putText(self.frame, text, (x-half_size, y+half_size*2+2), font, font_scale, color, thickness, line_type)

    def paint_angle(self, angle, cor, color=(0, 255, 255), half_size=7):
        """
        Paint an angle on the frame.

        :param angle: The angle to be painted.
        :param cor: The center of rotation.
        :param color: The color of the angle.
        """
        x, y = cor
        # self.frame = cv2.ellipse(self.frame, (x, y), (half_size, half_size), 90, 0, angle, color, thickness=2)
        self.frame = cv2.putText(self.frame, f"theta = {angle:.0f}", (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    def paint_arrow(self, start, end, color=(255, 255, 255)):
        """
        Paint an arrow on the frame.

        :param start: The start of the arrow.
        :param end: The end of the arrow.
        :param color: The color of the arrow.
        """
        x1, y1 = start
        x2, y2 = end
        self.frame = cv2.arrowedLine(self.frame, (int(x1), int(y1)),
                                     (int(x2), int(y2)), color, 2)
