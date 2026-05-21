# src/geospatial/pixel_projection.py

import math


def meters_to_latlon_offset(dx_m, dy_m, origin_lat):
    """
    Metre cinsinden offset'i yaklaşık lat/lon offset'e çevirir.

    dx_m: doğu-batı yönü metre
    dy_m: kuzey-güney yönü metre
    """

    meters_per_degree_lat = 111_320.0
    meters_per_degree_lon = 111_320.0 * math.cos(math.radians(origin_lat))

    if abs(meters_per_degree_lon) < 1e-6:
        meters_per_degree_lon = 1e-6

    dlat = dy_m / meters_per_degree_lat
    dlon = dx_m / meters_per_degree_lon

    return dlat, dlon


def rotate_offset(dx_m, dy_m, yaw_deg):
    """
    Kamera/görüntü koordinatındaki offset'i drone yaw açısına göre döndürür.

    Basit kabul:
    - DJI yaw 0 ise görüntü yukarısı kuzeye bakıyor kabul edilir.
    - DJI yaw saat yönünde artar: 90 derece doğu yönüdür.
    - dx: sağ yön
    - dy: yukarı yön

    Not:
    Kamera yaklaşık nadir (-90 gimbal pitch) kabul edilir.
    """

    yaw_rad = math.radians(yaw_deg)

    world_dx = dx_m * math.cos(yaw_rad) + dy_m * math.sin(yaw_rad)
    world_dy = -dx_m * math.sin(yaw_rad) + dy_m * math.cos(yaw_rad)

    return world_dx, world_dy


def pixel_to_geo_point(
    pixel_x,
    pixel_y,
    image_width,
    image_height,
    drone_lat,
    drone_lon,
    altitude_m,
    horizontal_fov_deg=73.7,
    vertical_fov_deg=53.1,
    drone_yaw_deg=0.0
):
    """
    Görüntüdeki bir piksel noktasını yaklaşık lat/lon noktasına çevirir.

    Varsayımlar:
    - Kamera yaklaşık yere bakıyor.
    - Zemin düz kabul ediliyor.
    - Drone GPS görüntü merkezine karşılık geliyor.
    - FOV değerleri yaklaşık.
    """

    if altitude_m is None or altitude_m <= 0:
        return drone_lat, drone_lon

    horizontal_fov_rad = math.radians(horizontal_fov_deg)
    vertical_fov_rad = math.radians(vertical_fov_deg)

    ground_width_m = 2 * altitude_m * math.tan(horizontal_fov_rad / 2)
    ground_height_m = 2 * altitude_m * math.tan(vertical_fov_rad / 2)

    meters_per_pixel_x = ground_width_m / image_width
    meters_per_pixel_y = ground_height_m / image_height

    dx_pixel = pixel_x - (image_width / 2)

    # Görüntü koordinatında y aşağı doğru artar.
    # Harita yönünde yukarı/kuzey pozitif olsun diye ters çeviriyoruz.
    dy_pixel = (image_height / 2) - pixel_y

    dx_m = dx_pixel * meters_per_pixel_x
    dy_m = dy_pixel * meters_per_pixel_y

    world_dx_m, world_dy_m = rotate_offset(
        dx_m,
        dy_m,
        drone_yaw_deg
    )

    dlat, dlon = meters_to_latlon_offset(
        world_dx_m,
        world_dy_m,
        drone_lat
    )

    fire_lat = drone_lat + dlat
    fire_lon = drone_lon + dlon

    return fire_lat, fire_lon


def estimate_area_from_pixel_area(
    pixel_area,
    image_width,
    image_height,
    altitude_m,
    horizontal_fov_deg=73.7,
    vertical_fov_deg=53.1
):
    """
    Pixel area değerinden yaklaşık m² hesaplar.
    """

    if altitude_m is None or altitude_m <= 0:
        return None

    horizontal_fov_rad = math.radians(horizontal_fov_deg)
    vertical_fov_rad = math.radians(vertical_fov_deg)

    ground_width_m = 2 * altitude_m * math.tan(horizontal_fov_rad / 2)
    ground_height_m = 2 * altitude_m * math.tan(vertical_fov_rad / 2)

    meters_per_pixel_x = ground_width_m / image_width
    meters_per_pixel_y = ground_height_m / image_height

    pixel_area_m2 = meters_per_pixel_x * meters_per_pixel_y

    return pixel_area * pixel_area_m2
