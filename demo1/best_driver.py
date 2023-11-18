"""
This driver is the best
"""
from rose.common import obstacles, actions # NOQA

driver_name = 'Best'

def return_direction_when_barrier(x, y, world):
    if x == 0 or x == 3:
        return actions.RIGHT
    elif x == 2 or x == 5:
        return actions.LEFT
    rightobstacle = world.get((x + 1, y - 2))
    leftobstacle = world.get((x - 1, y - 2))
    if rightobstacle == obstacles.PENGUIN:
        return actions.RIGHT
    elif leftobstacle == obstacles.PENGUIN:
        return actions.LEFT
    elif rightobstacle == obstacles.CRACK:
        return actions.RIGHT
    elif leftobstacle == obstacles.CRACK:
        return actions.LEFT
    elif rightobstacle == obstacles.WATER:
        return actions.RIGHT
    elif leftobstacle == obstacles.WATER:
        return actions.LEFT
    else:
        return actions.LEFT


def find_place_none(x, y, world):
    if x == 0 or x == 3:
        rightobstacle = world.get((x + 1, y - 2))
        if rightobstacle == obstacles.PENGUIN or rightobstacle == obstacles.CRACK or rightobstacle == obstacles.WATER:
            return actions.RIGHT
        else:
            return actions.NONE
    elif x == 2 or x == 5:
        leftobstacle = world.get((x - 1, y - 2))
        if leftobstacle == obstacles.PENGUIN or leftobstacle == obstacles.CRACK or leftobstacle == obstacles.WATER:
            return actions.LEFT
        else:
            return actions.NONE
    else:
        rightobstacle = world.get((x + 1, y - 2))
        leftobstacle = world.get((x - 1, y - 2))
        if rightobstacle == obstacles.PENGUIN:
            return actions.RIGHT
        elif leftobstacle == obstacles.PENGUIN:
            return actions.LEFT
        elif rightobstacle == obstacles.CRACK:
            return actions.RIGHT
        elif leftobstacle == obstacles.CRACK:
            return actions.LEFT
        elif rightobstacle == obstacles.WATER:
            return actions.RIGHT
        elif leftobstacle == obstacles.WATER:
            return actions.LEFT
        else:
            return actions.NONE


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
    elif obstacle == obstacles.TRASH or obstacle == obstacles.BIKE or obstacle == obstacles.BARRIER:
        return return_direction_when_barrier(x, y, world)
    elif obstacle == obstacles.NONE:
        return find_place_none(x, y, world)
    else:
        return actions.NONE

