from PIL import ImageFont, ImageDraw


def fill_in_img(resized, original_digits, solved_digits):

    draw = ImageDraw.Draw(resized)

    font = ImageFont.truetype("../fonts/Helvetica-Bold-Font.ttf", 40)

    w, h = resized.size
    w9, h9 = w/9, h/9

    offset_x = w9/2 - 15
    offset_y = h9/2 - 15

    # print(original_digits)
    # print(solved_digits)

    for i in range(9):
        for j in range(9):

            if original_digits[i, j] == 0:
                d = str(solved_digits[i, j])

                x = j*w9 + offset_x
                y = i*w9 + offset_y

                draw.text(
                    (x, y), d, color=0, font=font, align="center")

    return resized
