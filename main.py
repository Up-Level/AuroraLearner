import numpy as np
import cdflib
import png
import wic_look_angle

cdf = cdflib.CDF("im_k0_wic_20010913_v01.cdf")

cdf_vars = ["WIC_PIXELS", "EPOCH", "ORB_X", "ORB_Y", "ORB_Z", "VFOV", "SV_X", "SV_Y", "SV_Z", "SCSV_X", "SCSV_Y", "SCSV_Z", "SPINPHASE", "INST_AZIMUTH", "INST_CO_ELEVATION", "INST_ROLL"]
data = {}
for var in cdf_vars:
    data[var] = cdf.varget(var)

earth_radius = 6371

wic_pixels = data["WIC_PIXELS"]
# Ensure all values >0 for log
wic_pixels = np.vectorize(lambda p: 1 if p < 1 else p)(wic_pixels)

fov = np.deg2rad(data["VFOV"])

offset = np.deg2rad(np.array([
    data["INST_AZIMUTH"],
    data["INST_CO_ELEVATION"],
    data["INST_ROLL"]
]))

for i in range(wic_pixels.shape[0]):
    pos = np.array([data["ORB_X"][i], data["ORB_Y"][i], data["ORB_Z"][i]])
    #pos = np.array([0, 7000, 30000])

    # c is constant per pixel
    c = np.dot(pos, pos) - (earth_radius ** 2)

    scsv = np.array([
        data["SCSV_X"][i],
        data["SCSV_Y"][i],
        data["SCSV_Z"][i]
    ])
    sc = np.array([
        data["SV_X"][i],
        data["SV_Y"][i],
        data["SV_Z"][i]
    ])
    psi = np.deg2rad(data["SPINPHASE"][i])

    matrix = wic_look_angle.transformation_matrix(offset, scsv, sc, psi)
    centre_dir = np.matmul(matrix, np.array([0, 0, -1]))
    #centre_dir = -pos / np.linalg.norm(pos)

    u = np.cross(centre_dir, np.array([1, 0, 0]))
    u /= np.linalg.norm(u)

    w = np.cross(centre_dir, u)
    w /= np.linalg.norm(w)

    current_image = wic_pixels[i]
    pixel_locs = np.zeros((current_image.shape[0], current_image.shape[1], 3))

    for x in range(current_image.shape[0]):
        for y in range(current_image.shape[1]):
            offset_x = (x / current_image.shape[0] - 0.5) * fov
            offset_y = (y / current_image.shape[1] - 0.5) * fov
            #print(centre_dir, np.rad2deg(offset_x), np.rad2deg(offset_y))

            direction = offset_x * u + offset_y * w + centre_dir

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