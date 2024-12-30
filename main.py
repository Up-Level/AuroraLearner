import numpy as np
import cdflib
import png

cdf = cdflib.CDF("im_k0_wic_20010913_v01.cdf")

cdf_vars = ["WIC_PIXELS", "EPOCH", "RADIUS", "ORB_X", "ORB_Y", "ORB_Z", "VFOV"]
data = {}
for var in cdf_vars:
    data[var] = cdf.varget(var)

def cart2polar(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(z / r)
    phi = np.arccos(x / (np.sqrt(x ** 2 + y ** 2)))

    return r, theta, phi

earth_radius = 6371

wic_pixels = data["WIC_PIXELS"]
# Ensure all values >0 for log
wic_pixels = np.vectorize(lambda p: 1 if p < 1 else p)(wic_pixels)

fov = np.deg2rad(data["VFOV"])
fov_vector = np.array([
    np.cos(fov) * np.sin(fov),
    np.sin(fov),
    np.cos(fov) ** 2
])

for i in range(wic_pixels.shape[0]):
    pos = np.array([data["ORB_X"][i], data["ORB_Y"][i], data["ORB_Z"][i]])
    #pos = np.array([0, 7000, 20000])

    # c is constant per pixel
    c = np.dot(pos, pos) - (earth_radius ** 2)

    centre_dir = -pos / np.sqrt(np.dot(pos, pos))
    
    current_image = wic_pixels[i]
    pixel_locs = np.zeros((current_image.shape[0], current_image.shape[1], 3))

    for x in range(current_image.shape[0]):
        for y in range(current_image.shape[1]):
            pixel_offset = np.array([x / current_image.shape[0] - 0.5, 1, y / current_image.shape[1] - 0.5])
            direction = pixel_offset * fov_vector + centre_dir

            a = np.dot(direction, direction)
            b = 2 * np.dot(pos, direction)

            #print(pixel_offset, direction, centre_dir, pos, centre_pos)

            discriminant = (b ** 2) - (4 * a * c)
            if discriminant >= 0:
                #roots = np.roots((a, b, c))
                root = (-b + np.sqrt(discriminant)) / (2 * a)
                pixel_locs[x][y] = pos + root * direction
                #print(pixel_locs[x][y])
                
    image_locs = ((pixel_locs - pixel_locs.min()) / (pixel_locs.max() - pixel_locs.min() + 0.00001) * 255).astype(int)
    image_locs = image_locs.reshape((image_locs.shape[0], image_locs.shape[1] * image_locs.shape[2]))

    image = ((current_image - current_image.min()) / (current_image.max() - current_image.min() + 0.00001) * 255).astype(int)

    print(f"""{i}
locs: {pixel_locs.min()} {pixel_locs.max()} image locs: {image_locs.min()} {image_locs.max()}
wic: {current_image.min()} {current_image.max()} image wic: {image.min()} {image.max()}
""")

    writer = png.Writer(256, 256, greyscale=False)
    writer_gs = png.Writer(256, 256, greyscale=True)

    with open(f"locs/{i}.png", "wb") as file:
        writer.write(file, image_locs.tolist())
    with open(f"images/{i}.png", "wb") as file:
        writer_gs.write(file, image.tolist())