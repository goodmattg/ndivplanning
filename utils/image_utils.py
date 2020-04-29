def denorm(image):
    return ((image + 1.0) / 2.0) * 255.0


def norm(image):
    return (image / 255.0 - 0.5) * 2.0
