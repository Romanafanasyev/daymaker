from PIL import Image, ImageDraw
from network import NeuralNetwork
import math

image1 = Image.open("images/color.jpg")
image2 = Image.open("images/blackwhite.jpg")
draw = ImageDraw.Draw(image1)
width = image1.size[0]
height = image1.size[1]
pix1 = image1.load()
pix2 = image2.load()

print(pix1[0,0])
print(pix2[0,0])

my_net = NeuralNetwork()

for num in range(1):
    for i in range(width):
        for j in range(height):
            my_net.forward_prop(pix1[i,j])
            my_net.back_prop(pix1[i,j], pix2[i,j])
            print(num, ':', i*j, ':  ', my_net.output[0]*255, ', ',my_net.output[1]*255, ', ',my_net.output[2]*255, ' | mse = ', my_net.e0)


for x in range(width):
    for y in range(height):
        my_net.forward_prop(pix1[x, y])
        r = math.floor(my_net.output[0]*255)
        g = math.floor(my_net.output[1]*255)
        b = math.floor(my_net.output[2]*255)
        draw.point((x, y), (r, g, b))

image1.save("black_my2.jpg", "JPEG")