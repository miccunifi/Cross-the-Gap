from pathlib import Path

PROJECT_ROOT = Path(__file__).absolute().parents[2].absolute()


def is_image_file(filename: str) -> bool:
    '''
    Check whether a file has an image extension
    :param filename: name of file
    '''
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])
