"""
This driver is the good
"""
from rose.common import obstacles, actions # NOQA

driver_name = 'Good driver'

def drive(world):
    x = world.car.x
    y = world.car.y
    obstacle = world.get((x, y - 1))

    if obstacle == obstacles.PENGUIN:
        return actions.PICKUP
    elif obstacle == obstacles.WATER:
        return actions.BRAKE
    elif obstacle == obstacles.CRACK:
        return actions.JUMP
    elif obstacle == obstacles.NONE:
        return actions.NONE
    else:
        return actions.RIGHT if (x % 3) == 0 else actions.LEFT

