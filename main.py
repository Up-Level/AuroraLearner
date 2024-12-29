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

for i in range(50, 51):
    earth_radius = 6371
    pos = np.array([data["ORB_X"][i], data["ORB_Y"][i], data["ORB_Z"][i]])
    pos = np.array([0, 7000, 20000])
    data["RADIUS"][i] = 3.139

    centre_pos = pos / data["RADIUS"][i]
    dir_vector = centre_pos / np.sqrt(np.dot(centre_pos, centre_pos))

    centre_dir = np.array([
        np.arcsin(dir_vector[2]),
        np.arctan2(dir_vector[0], dir_vector[1])
    ])
    """centre_dir = np.array([
        np.arctan2(centre_pos[1], centre_pos[0]),
        np.arccos(centre_pos[2] / (data["RADIUS"][i] * earth_radius)),
    ])"""
    
    pixel_data = data["WIC_PIXELS"][i]
    pixel_locs = np.zeros((pixel_data.shape[0], pixel_data.shape[1], 3))

    fov = np.deg2rad(data["VFOV"])
    fov_vector = np.array([
        np.cos(fov) * np.sin(fov),
        np.sin(fov),
        np.cos(fov) ** 2
    ])
    print(fov_vector)

    for x in range(pixel_data.shape[0]):
        for y in range(pixel_data.shape[1]):
            pixel_offset = (np.array([x / pixel_data.shape[0], y / pixel_data.shape[1]]) - 0.5) * fov
            angles = pixel_offset + centre_dir

            direction = -np.array([
                np.sin(angles[0]) * np.cos(angles[1]),
                np.cos(angles[0]) * np.cos(angles[1]),
                np.sin(angles[0]),
            ])

            a = np.dot(direction, direction)
            b = 2 * np.dot(pos, direction)
            c = np.dot(pos, pos) - (earth_radius ** 2)

            #print(a, b, c)
            #print(direction, pos, centre_pos, np.rad2deg(angles), np.rad2deg(centre_dir), b)
            #print(b ** 2, 4 * a * c)

            discriminant = (b ** 2) - (4 * a * c)
            if discriminant >= 0:
                roots = np.roots((a, b, c))
                pixel_locs[x][y] = pos + roots[1] * direction
                print(pixel_locs[x][y])

    """image = (pixel_locs.min() + pixel_locs * (np.abs(pixel_data.min()) + pixel_data.max())) * 255
    #image = (255 * (pixel_data / pixel_data.max())).astype(int)
    print(image.min(), image.max(), image.shape)

    writer = png.Writer(256, 256, greyscale=True)
    with open("out.png", "wb") as file:
        writer.write(file, image.tolist())"""